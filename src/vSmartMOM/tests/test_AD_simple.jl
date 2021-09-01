using Revise
using Plots
using Pkg
# Pkg.activate(".")
using RadiativeTransfer
using RadiativeTransfer.Architectures
using RadiativeTransfer.Absorption
using RadiativeTransfer.Scattering
using RadiativeTransfer.vSmartMOM
using RadiativeTransfer.SolarModel
using InstrumentOperator
using Interpolations
using Polynomials
using ForwardDiff 
using Distributions
using NCDatasets

## Atmospheric Radiative Transfer

# Load parameters from file
parameters = vSmartMOM.parameters_from_yaml("RadiativeTransfer/test/helper/O2Parameters.yaml")
FT = Float64

oco_file = "/net/fluo/data1/group/oco2/L1bSc/oco2_L1bScGL_15258a_170515_B10003r_200214061601.h5"
oco_met_file = "/net/fluo/data1/group/oco2/L2Met/oco2_L2MetGL_26777a_190715_B10003r_200429213029.h5"
ils_file = "/home/rjeyaram/RadiativeTransfer/src/vSmartMOM/ils_oco2.json"

fp = 5
band = 1

function conv_spectra_local(m::VariableKernelInstrument, ν, spectrum; stride=1)
    # FT = eltype(m.ν_out)
    # Define grid where to perform convolution:
    
    # Padding at both sides required:
    off = ceil(Int, size(m.kernel, 1) / 2)
    ind = off:stride:(length(ν) - off)
    
    # knots where convolution will be applied to
    knots = view(ν, ind)
    te = LinearInterpolation(m.ν_out, Float32.(m.ind_out))
    spec_out = zeros(Real, length(knots));
    for i in eachindex(knots)
        # Simple first, nearest neighbor ILS
        ind_fraction = round(Int, te(knots[i]));
        kernel = view(m.kernel, :, ind_fraction)
        for j in eachindex(kernel)
            spec_out[i] += kernel[j] * spectrum[ind[i] + j] 
        end
    end
    # Change this later to only perform conv around output grid!
    fin = LinearInterpolation(ν[ind], spec_out; extrapolation_bc=Interpolations.Flat())
    return fin(m.ν_out)
end;

# Runner is used to set AD fields as duals
function runner!(y, x, parameters=parameters, oco_file=oco_file, 
                                              oco_met_file=oco_met_file, 
                                              ils_file=ils_file)

    # Set parameters fields as the dual numbers
    parameters.brdf = [LambertianSurfaceScalar(x[1])]

    parameters.scattering_params.rt_aerosols[1].τ_ref = x[2];

    parameters.scattering_params.rt_aerosols[1].aerosol.size_distribution = LogNormal(log(x[3]), log(x[4]), check_args=false)

    parameters.scattering_params.rt_aerosols[1].aerosol.nᵣ = x[5];
    parameters.scattering_params.rt_aerosols[1].aerosol.nᵢ = x[6];

    parameters.scattering_params.rt_aerosols[1].p₀ = x[7];
    parameters.scattering_params.rt_aerosols[1].σp = x[8];

    # Set profiles properly
    met = Dataset(oco_met_file);
    T_met = met.group["Meteorology"]["temperature_profile_met"][:,fp,iOrbit];
    ak = met.group["MeteorologyDiagnostics"]["ak"][:,fp,iOrbit];
    bk = met.group["MeteorologyDiagnostics"]["bk"][:,fp,iOrbit];
    p_surf = met.group["Meteorology"]["surface_pressure_met"][fp,iOrbit];
    p_half = (ak + bk * p_surf);
    p_half = vcat(p_half, p_surf);
    q = met.group["Meteorology"]["specific_humidity_profile_met"][:,fp,iOrbit];
    parameters.p = p_half / 100
    parameters.q = q
    parameters.T = T_met

    model = model_from_parameters(parameters);

    # Run the model to obtain reflectance matrix
    R = vSmartMOM.rt_run(model, i_band=1);

    # Produce black-body in wavenumber range
    T = 5777
    λ_grid = reverse(1e4 ./ parameters.spec_bands[1]) #collect(757:0.01:777)
    black_body = planck_spectrum_wl(T, λ_grid) * 2.1629e-05 * pi
    black_body = SolarModel.watts_to_photons(λ_grid, black_body)

    # Get solar transmittance spectrum 
    solar_transmission = solar_transmission_from_file("/home/rjeyaram/RadiativeTransfer/src/solar_merged_20160127_600_26316_100.out", parameters.spec_bands[1])

    # Get outgoing solar radiation
    sun_out = reverse(solar_transmission) .* black_body

    # Apply Earth reflectance matrix 
    earth_out = sun_out .* reverse(R[:])

    # y[:] = earth_out[:]

    # Set up InstrumentOperator
    # oco_file = "/home/cfranken/oco2_L1bScND_18688a_180105_B8100r_180206190633.h5"
    # ils_file = "/home/rjeyaram/RadiativeTransfer/src/vSmartMOM/ils_oco2.json"
    ils_Δ, ils_in, dispersion = InstrumentOperator.read_ils_table(oco_file, ils_file);

    # Define model grid:
    res = 0.001

    # Just consider the ILS within ± 0.35nm
    grid_x = -0.35e-3:res*1e-3:0.35e-3
    
    extended_dims = [fp,band] # Footprint, band

    # Re-interpolate I from ν_grid to new grid/resolution
    interp_I = LinearInterpolation(λ_grid, earth_out);
    wl = 757.5:res:771.0
    I_wl = interp_I(wl/1000)

    # Pixels to be used
    ind_out = collect(0:1015); 

    # Eventual grid of OCO-2 for Band 1, FP 5:
    dispPoly = Polynomial(view(dispersion, :, extended_dims...))
    ν = Float32.(dispPoly.(1:1016))

    # Prepare ILS table:
    ils_pixel   = InstrumentOperator.prepare_ils_table(grid_x, ils_in, ils_Δ,extended_dims)
    oco2_kernel = VariableKernelInstrument(ils_pixel, ν, ind_out)

    # Convolve input spectrum with variable kernel
    I_conv = conv_spectra_local(oco2_kernel, wl./1e3, I_wl)

    y[:] = I_conv[:]

end

x = FT[0.05,
       0.05,
       1.3,
       2.0,
       1.3,
       0.00000001,
       90000,
       5000.0]

# Run FW model:
# @time runner(x);
I_conv = zeros(1016)
@time dfdx = ForwardDiff.jacobian(runner!, I_conv, x);

# model = model_from_parameters(parameters); 
# R = rt_run(model);

## Solar Model 

# Produce black-body in wavenumber range
# T = 5777
# λ_grid = reverse(1e4 ./ parameters.spec_bands[1]) #collect(757:0.01:777)
# black_body_watts = planck_spectrum_wl(T, λ_grid)
# black_body = SolarModel.watts_to_photons(λ_grid, black_body_watts)

# # Get solar transmittance spectrum 
# solar_transmission = solar_transmission_from_file("/home/rjeyaram/RadiativeTransfer/src/solar_merged_20160127_600_26316_100.out", parameters.spec_bands[1])

# # Get outgoing solar radiation
# sun_out = solar_transmission .* black_body

# ## Apply Earth reflectance matrix 

# earth_out = sun_out .* reverse(R[:])

## Set up Instrument Operator
ils_Δ, ils_in, dispersion = InstrumentOperator.read_ils_table(oco_file, ils_file);

# Define model grid:
res = 0.001

# Just consider the ILS within ± 0.35nm
grid_x = -0.35e-3:res*1e-3:0.35e-3
fp = 5
band = 1
extended_dims = [fp,band] # Footprint, band

# # Re-interpolate I from ν_grid to new grid/resolution
# interp_I = LinearInterpolation(λ_grid, earth_out);
# wl = 757.5:res:771.0
# I_wl = interp_I(wl/1000)

# # Pixels to be used
# ind_out = collect(0:1015); 

# # Eventual grid of OCO-2 for Band 1, FP 5:
# dispPoly = Polynomial(view(dispersion, :, extended_dims...))
# ν = Float32.(dispPoly.(1:1016))

# # Prepare ILS table:
# ils_pixel   = InstrumentOperator.prepare_ils_table(grid_x, ils_in, ils_Δ,extended_dims)
# oco2_kernel = VariableKernelInstrument(ils_pixel, ν, ind_out)

# # Convolve input spectrum with variable kernel
# I_conv = conv_spectra(oco2_kernel, wl./1e3, I_wl)

## Getting an OCO spectrum for fun to fit:
ocoData = Dataset(oco_file)
o2AbandSpectra = ocoData.group["SoundingMeasurements"]["radiance_o2"]
SoundingGeometry = ocoData.group["SoundingGeometry"]
vza = SoundingGeometry["sounding_zenith"]
sza = SoundingGeometry["sounding_solar_zenith"]
lat = SoundingGeometry["sounding_latitude"]
lon = SoundingGeometry["sounding_longitude"]
# Pick an index along the orbit track
iOrbit = 5300
sza_ = sza[fp,iOrbit]
vza_ = vza[fp,iOrbit]
oco2_Aband = o2AbandSpectra[:,fp,iOrbit]

# function applyInstrument(Fin)
#     interp_I = CubicSplineInterpolation(range(ν_grid[1], ν_grid[end], length=length(ν_grid)), Fin);
#     conv_spectra(oco2_kernel, wl./1e3,interp_I(1e7./wl))
# end

# Get cludgy Jacobian with convolution
# K = zeros(1016,6)
# 101543.37f0
# for i=1:6
#     K[:,i] = applyInstrument(dfdx[:,i])
# end

# Measurement vector y
# y = oco2_Aband #./ 4.3e15 * 2.02e4;
# y = oco2_Aband*1.6e4 #oco2_Aband #./ 4.3e15 * 2.02e4;

#F(x)
y = oco2_Aband;
Fx = I_conv;
K = dfdx;

# Regular least squares:
dx = K \ (y-Fx)

