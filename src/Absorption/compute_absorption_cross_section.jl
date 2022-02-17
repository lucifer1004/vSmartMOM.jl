#=
 
This file contains functions that perform the core absorption cross section calculations. 
While `compute_absorption_cross_section` *can* be called directly by users, it's 
wrapped by `absorption_cross_section`, in autodiff_handler.jl to allow autodiff users
to call the function seamlessly alike non-autodiff users. 

`compute_absorption_cross_section` is implemented both from scratch (::HitranModel) and as 
an interpolation (::InterpolationModel). For the former, there are separate line_shape 
kernel functions that allow the calculation to run in parallel, and a qoft! function to 
interpolate partition sums. 

=#

"""
Given the hitran data and necessary parameters, calculate an absorption cross-section at the
given pressure, temperature, and grid of wavenumbers (or wavelengths)
"""
function compute_absorption_cross_section(
                # Required
                model::HitranModel,          # Model to use in this cross section calculation 
                                             # (Calculation from Hitran data vs. using Interpolator)
                # Wavelength [nm] or wavenumber [cm-1] grid (modify using wavelength_flag)
                grid::Union{AbstractRange{<:Real}, AbstractArray},  # Can be range OR array
                pressure::Real,              # actual pressure [hPa]
                temperature::Real;           # actual temperature [K]    
                # Optionals
                wavelength_flag::Bool=false  # Use wavelength in nm (true) or wavenumber cm-1 units (false)    
                )
    
    @unpack hitran, broadening, wing_cutoff, vmr, CEF, architecture = model

    # Notify user of wavelength grid
    if (wavelength_flag)
        @info """
        Note: Cross-section reported to wavelength grid
        (will internally convert to wavenumber grid for calculations)
        """  maxlog = 5
    end

    # Convert T to float type (ex. if Int)
    if !(temperature isa ForwardDiff.Dual)
        temperature = AbstractFloat(temperature)
    end
    mols = unique(hitran.mol)
    isos = unique(hitran.iso)
    @assert length(mols)==1 "Use only one molecule at a time"
    @assert length(isos)==1 "Use only one molecule at a time"
    # Get temperature grid
    TT = get_TT(mols[1], isos[1])
    TQ = get_TQ(mols[1], isos[1])

    # Store results here to return
    result = array_type(architecture)(zeros(eltype(temperature), length(grid)))
    fill!(result, 0);

    # Calculate the minimum and maximum grid bounds, including the wing cutoff
    grid_max = maximum(grid) + wing_cutoff
    grid_min = minimum(grid) - wing_cutoff

    # Convert to wavenumber from [nm] space if necessary
    grid = wavelength_flag ? reverse(nm_per_m ./ grid) : grid
    grid_min, grid_max = wavelength_flag ? (nm_per_m /grid_max, nm_per_m/grid_min) : (grid_min, grid_max)

    # Interpolators from grid bounds to index values
    if length(grid)>1
        grid_idx_interp_low  = LinearInterpolation(grid, 1:1:length(grid), extrapolation_bc=1)
        grid_idx_interp_high = LinearInterpolation(grid, 1:1:length(grid), extrapolation_bc=length(grid))
    end

    # Temporary storage array for output of qoft!. Compiler/speed issues when returning value in qoft
    rate = zeros(eltype(temperature), 1)

    # Declare the device being used
    device = devi(architecture)

    grid = array_type(architecture)(grid)

    # Loop through all transition lines:
    for j in eachindex(hitran.Sᵢ)

        # Test that this ν lies within the grid
        if grid_min < hitran.νᵢ[j] < grid_max

            # Apply pressure shift
            ν   = hitran.νᵢ[j] + pressure / p_ref * hitran.δ_air[j]

            # Compute Lorentzian HWHM
            γ_l = (hitran.γ_air[j] *
                  (1 - vmr) * pressure / p_ref + hitran.γ_self[j] *
                  vmr * pressure / p_ref) *
                  (t_ref / temperature)^hitran.n_air[j]

            # Compute Doppler HWHM
            γ_d = ((cSqrt2Ln2 / cc_) * sqrt(cBolts_ / cMassMol) * sqrt(temperature) * 
            hitran.νᵢ[j] / sqrt(mol_weight(hitran.mol[j], hitran.iso[j])))

            # Ratio of widths
            y = sqrt(cLn2) * γ_l / γ_d

            # Apply line intensity temperature corrections
            S = hitran.Sᵢ[j]
            # Just compute if nothing changed
            if (j==1) || (hitran.mol[j] != hitran.mol[j-1]) || (hitran.iso[j] != hitran.iso[j-1])
                qoft!(hitran.mol[j], hitran.iso[j], temperature, t_ref, rate)
            end
            if hitran.E″[j] != -1
                #@show rate
                S = S * rate[1] *
                        exp(c₂ * hitran.E″[j] * (1 / t_ref - 1 / temperature)) *
                        (1 - exp(-c₂ * hitran.νᵢ[j] / temperature)) / (1 - exp(-c₂ * hitran.νᵢ[j] / t_ref));

            end

            if length(grid)>1
                # Calculate index range that this ν impacts
                ind_start = Int64(round(grid_idx_interp_low(ν - wing_cutoff)))
                ind_stop  = Int64(round(grid_idx_interp_high(ν + wing_cutoff)))
                
                # Create views from the result and grid arrays
                result_view   = view(result, ind_start:ind_stop);
                grid_view     = view(grid, ind_start:ind_stop);
            else
                result_view   = view(result, 1);
                grid_view     = view(grid, 1);
            end

            # Kernel for performing the lineshape calculation
            kernel! = line_shape!(device)

            # Run the event on the kernel 
            # That this, this function adds to each element in result, the contribution from this transition
            event = kernel!(result_view, grid_view, ν, γ_d, γ_l, y, S, broadening, CEF, ndrange=length(grid_view))
            wait(device, event)
            synchronize_if_gpu()
        end
    end

    # Return the resulting lineshape
    return (wavelength_flag ? reverse(result) : result)
end

"""
    $(FUNCTIONNAME)(model::InterpolationModel, grid::AbstractRange{<:Real}, pressure::Real, temperature::Real; wavelength_flag::Bool=false)

Given an Interpolation Model, return the interpolated absorption cross-section at the given pressure, 
temperature, and grid of wavenumbers (or wavelengths)

"""
function compute_absorption_cross_section(
    # Required
    model::InterpolationModel,      # Model to use in this cross section calculation 
                                    # (Calculation from Interpolator vs. Hitran Data)
    # Wavelength [nm] or wavenumber [cm-1] grid
    grid::Union{AbstractRange{<:Real}, AbstractArray},  # Can be range OR array
    pressure::Real,                 # actual pressure [hPa]
    temperature::Real;              # actual temperature [K]  
    # Optionals 
    wavelength_flag::Bool=false,    # Use wavelength in nm (true) or wavenumber cm-1 units (false)            
    )

    # Convert to wavenumber from [nm] space if necessary
    grid = wavelength_flag ? reverse(nm_per_m ./ collect(grid)) : collect(grid)

    # Scale the interpolation to match the model grids
    sitp = scale(model.itp, model.ν_grid, model.p_grid, model.t_grid)

    # Perform the interpolation and return the resulting grid
    return sitp(grid, pressure, temperature)
end

#=

Line-shape kernel functions that are called by absorption_cross_section

=# 

@kernel function line_shape!(A, @Const(grid), ν, γ_d, γ_l, y, S, ::Doppler, CEF)
    FT = eltype(ν)
    I = @index(Global, Linear)
    @inbounds A[I] += FT(S) * FT(cSqrtLn2divSqrtPi) * exp(-FT(cLn2) * ((FT(grid[I]) - FT(ν)) / FT(γ_d))^2) / FT(γ_d)
end

@kernel function line_shape!(A, @Const(grid), ν, γ_d, γ_l, y, S, ::Lorentz, CEF)
    FT = eltype(ν)
    I = @index(Global, Linear)
    @inbounds A[I] += FT(S) * FT(γ_l) / (FT(pi) * (FT(γ_l)^2 + (FT(grid[I]) - FT(ν))^2))
end

@kernel function line_shape!(A, grid, ν::FT, γ_d::FT, γ_l::FT, y::FT, S::FT, ::Voigt, CEF) where FT
#    FT = eltype(ν)
    I = @index(Global, Linear)
    @inbounds A[I] += S * cSqrtLn2divSqrtPi / γ_d * real(w(CEF, cSqrtLn2 / γ_d * (grid[I] - ν) + im * y))
end

@kernel function line_shape32!(A, @Const(grid), ν, γ_d, γ_l, y, S, broadening, CEF)
    line_shape!(A, grid, Float32(ν), Float32(γ_d), Float32(γ_l), Float32(y), Float32(S), broadening, CEF)
end

#=

Function to interpolate partition sum for specified isotopologue

=# 

"Given molecule and isotopologue numbers (M, I), target temperature (T), and reference 
temperature (T_ref), store the ratio of interpolated partition sums in `result`"
function qoft!(M, I, T, T_ref, result)

    # Get temperature grid
    TT = get_TT(M, I)
    TQ = get_TQ(M, I)
    #@show TT
    # Error if out of temperature range
    Tmin = minimum(TT); Tmax = maximum(TT)
    @assert (Tmin < T < Tmax) "TIPS2017: T ($T) must be between $Tmin K and $Tmax K."

    # Interpolate partition sum for specified isotopologue
    interp = DI_CS(TQ, TT)
    Qt  = interp(T)
    Qt2 = interp(T_ref) 

    # Save the ratio result 
    result[1] = Qt2/Qt
    nothing
end