#=
 
This file implements rt_kernel!, which performs the core RT routines (elemental, doubling, interaction)
 
=#
#No Raman (default)
# Perform the Core RT routines (elemental, doubling, interaction)
#=
function rt_kernel_multisensor!(RS_type::noRS, 
                                sensor_levels, 
                                pol_type, 
                                SFI, 
                                added_layer, 
                                composite_layer, 
                                computed_layer_properties, 
                                m, 
                                quad_points, 
                                I_static, 
                                architecture, 
                                qp_μN, 
                                iz, 
                                arr_type) 

    @unpack τ_λ, ϖ_λ, τ, ϖ, Z⁺⁺, Z⁻⁺, dτ_max, dτ, ndoubl, dτ_λ, expk, scatter, τ_sum, scattering_interface = computed_layer_properties

    # If there is scattering, perform the elemental and doubling steps
    if scatter
        
        @timeit "elemental" elemental!(pol_type, SFI, τ_sum, dτ_λ, dτ, ϖ_λ, ϖ, Z⁺⁺, Z⁻⁺, m, ndoubl, scatter, quad_points,  added_layer,  I_static, architecture)
        #println("Elemental done...")
        @timeit "doubling"   doubling!(pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
        #println("Doubling done...")
    else # This might not work yet on GPU!
        # If not, there is no reflectance. Assign r/t appropriately
        added_layer.r⁻⁺[:] .= 0;
        added_layer.r⁺⁻[:] .= 0;
        added_layer.J₀⁻[:] .= 0;
        temp = Array(exp.(-τ_λ./qp_μN'))
        #added_layer.t⁺⁺, added_layer.t⁻⁻ = (Diagonal(exp(-τ_λ / qp_μN)), Diagonal(exp(-τ_λ / qp_μN)))   
        for iλ = 1:length(τ_λ)
            added_layer.t⁺⁺[:,:,iλ] = Diagonal(temp[iλ,:]);
            added_layer.t⁻⁻[:,:,iλ] = Diagonal(temp[iλ,:]);
        end
    end

    # @assert !any(isnan.(added_layer.t⁺⁺))
    
    # If this TOA, just copy the added layer into the bottom composite layer
    if (iz == 1)
        for ims = 1:length(sensor_levels)
            if(ims==0)
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻)
            else
                composite_layer.topT⁺⁺[ims][:], composite_layer.topT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.topR⁻⁺[ims][:], composite_layer.topR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.topJ₀⁺[ims][:], composite_layer.topJ₀⁻[ims][:] = 
                    Array((added_layer.J₀⁺), Array(added_layer.J₀⁻)
            end
        end
    # If this is not the TOA, perform the interaction step
    else
        #@timeit "interaction_multisensor" interaction_multisensor!(RS_type, sensor_levels, scattering_interface, SFI, composite_layer, added_layer, I_static)
        for ims=1:length(sensor_levels)
            if sensor_levels[ims]==(iz+1) #include ims==Nz with ims==0
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻)
            elseif sensor_levels[ims]>(iz+1) 
                @timeit "interaction_multisensor" interaction_bot!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)
            elseif sensor_levels[ims]<=iz 
                @timeit "interaction_multisensor" interaction_top!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)                
            end
        end
    end
end
=#
#=
#
#Rotational Raman Scattering (default)
# Perform the Core RT routines (elemental, doubling, interaction)
function rt_kernel_multisensor!(RS_type::Union{RRS, VS_0to1, VS_1to0}, 
                                sensor_levels, 
                                pol_type, 
                                SFI, 
                                added_layer, 
                                composite_layer, 
                                computed_layer_properties, 
                                m, 
                                quad_points, 
                                I_static, 
                                architecture, 
                                qp_μN, 
                                iz, 
                                arr_type) 
    
    @unpack τ_λ, ϖ_λ, τ, ϖ, Z⁺⁺, Z⁻⁺, dτ_max, dτ, ndoubl, dτ_λ, expk, scatter, τ_sum, scattering_interface = computed_layer_properties
    @unpack Z⁺⁺_λ₁λ₀, Z⁻⁺_λ₁λ₀ = RS_type
    # If there is scattering, perform the elemental and doubling steps
    if scatter
        #@show τ, ϖ, RS_type.fscattRayl
        @timeit "elemental_inelastic" elemental_inelastic!(RS_type, 
                                                pol_type, SFI, 
                                                τ_sum, dτ_λ, ϖ_λ, 
                                                Z⁺⁺_λ₁λ₀, Z⁻⁺_λ₁λ₀, 
                                                m, ndoubl, scatter, 
                                                quad_points,  added_layer,  
                                                I_static, architecture)
        #println("Elemental inelastic done...")                                        
        @timeit "elemental" elemental!(pol_type, SFI, 
                                    τ_sum, dτ_λ, dτ, 
                                    ϖ_λ, ϖ, 
                                    Z⁺⁺, Z⁻⁺, 
                                    m, ndoubl, scatter, 
                                    quad_points,  added_layer,  
                                    I_static, architecture)
        #println("Elemental  done...")
        @timeit "doubling_inelastic" doubling_inelastic!(RS_type, pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
        #println("Doubling done...")
        #@timeit "doubling"   doubling!(pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
    else # This might not work yet on GPU!
        # If not, there is no reflectance. Assign r/t appropriately
        added_layer.r⁻⁺[:] .= 0;
        added_layer.r⁺⁻[:] .= 0;
        added_layer.J₀⁻[:] .= 0;
        added_layer.ier⁻⁺[:] .= 0;
        added_layer.ier⁺⁻[:] .= 0;
        added_layer.ieJ₀⁻[:] .= 0;
        added_layer.iet⁻⁻[:] .= 0;
        added_layer.iet⁺⁺[:] .= 0;
        added_layer.ieJ₀⁺[:] .= 0;
        temp = Array(exp.(-τ_λ./qp_μN'))
        #added_layer.t⁺⁺, added_layer.t⁻⁻ = (Diagonal(exp(-τ_λ / qp_μN)), Diagonal(exp(-τ_λ / qp_μN)))   
        for iλ = 1:length(τ_λ)
            added_layer.t⁺⁺[:,:,iλ] = Diagonal(temp[iλ,:]);
            added_layer.t⁻⁻[:,:,iλ] = Diagonal(temp[iλ,:]);
        end
    end

    # @assert !any(isnan.(added_layer.t⁺⁺))
    
    # If this TOA, just copy the added layer into the composite layer
    if (iz == 1)
        for ims=1:length(sensor_levels)
            if(ims==0)
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.botieT⁺⁺[ims][:], composite_layer.botieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.botieR⁻⁺[ims][:], composite_layer.botieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.botieJ₀⁺[ims][:], composite_layer.botieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            else
                composite_layer.topT⁺⁺[ims][:], composite_layer.topT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.topR⁻⁺[ims][:], composite_layer.topR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.topJ₀⁺[ims][:], composite_layer.topJ₀⁻[ims][:] = 
                    added_layer.J₀⁺, added_layer.J₀⁻)
                composite_layer.topieT⁺⁺[ims][:], composite_layer.topieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.topieR⁻⁺[ims][:], composite_layer.topieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.topieJ₀⁺[ims][:], composite_layer.topieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            end
        end
    # If this is not the TOA, perform the interaction step
    else
        #@timeit "interaction_multisensor" interaction_multisensor!(RS_type, sensor_levels, scattering_interface, SFI, composite_layer, added_layer, I_static)
        for ims=1:length(sensor_levels)
            if sensor_levels[ims]==(iz+1) #include ims==Nz with ims==0
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.botieT⁺⁺[ims][:], composite_layer.botieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.botieR⁻⁺[ims][:], composite_layer.botieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.botieJ₀⁺[ims][:], composite_layer.botieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            elseif sensor_levels[ims]>(iz+1) 
                @timeit "interaction_multisensor" interaction_bot!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)
            elseif sensor_levels[ims]<=iz 
                @timeit "interaction_multisensor" interaction_top!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)                
            end
        end
    end
end
=#
### New update:
function rt_kernel_multisensor!(RS_type::noRS{FT}, 
                    sensor_levels,
                    pol_type, SFI, 
                    added_layer, 
                    composite_layer, 
                    computed_layer_properties::CoreScatteringOpticalProperties, 
                    scattering_interface, 
                    τ_sum, 
                    m, quad_points, 
                    I_static, 
                    architecture, 
                    qp_μN, iz, arr_type) where {FT}

    @unpack qp_μ, μ₀ = quad_points
    # Just unpack core optical properties from 
    @unpack τ, ϖ, Z⁺⁺, Z⁻⁺ = computed_layer_properties
    # SUNITI, check? Also, better to write function here
    #@show "here", size(τ .* ϖ), size(qp_μ)
    #@show maximum(τ .* ϖ), minimum(qp_μ)
    dτ_max = minimum([maximum(τ .* ϖ), FT(0.001) * minimum(qp_μ)])
    _, ndoubl = doubling_number(dτ_max, maximum(τ .* ϖ))
    scatter = true # edit later
    arr_type = array_type(architecture)
    # Compute dτ vector
    dτ = τ ./ 2^ndoubl
    expk = arr_type(exp.(-dτ /μ₀))
    

    # If there is scattering, perform the elemental and doubling steps
    if scatter
        @timeit "elemental" elemental!(pol_type, SFI, 
                                        τ_sum, dτ, 
                                        computed_layer_properties, 
                                        m, ndoubl, scatter, quad_points,  
                                        added_layer,  architecture)
        #println("Elemental done...")
        @timeit "doubling"   doubling!(pol_type, SFI, 
                                        expk, ndoubl, 
                                        added_layer, 
                                        I_static, architecture)
        #println("Doubling done...")
    else # This might not work yet on GPU!
        # If not, there is no reflectance. Assign r/t appropriately
        added_layer.r⁻⁺[:] .= 0;
        added_layer.r⁺⁻[:] .= 0;
        added_layer.J₀⁻[:] .= 0;
        temp = Array(exp.(-τ_λ./qp_μN'))
        #added_layer.t⁺⁺, added_layer.t⁻⁻ = (Diagonal(exp(-τ_λ / qp_μN)), Diagonal(exp(-τ_λ / qp_μN)))   
        for iλ = 1:length(τ_λ)
            added_layer.t⁺⁺[:,:,iλ] = Diagonal(temp[iλ,:]);
            added_layer.t⁻⁻[:,:,iλ] = Diagonal(temp[iλ,:]);
        end
    end
    #M1 = Array(added_layer.t⁺⁺)
    #M2 = Array(added_layer.r⁺⁻)
    #M3 = Array(added_layer.J₀⁻)
    #M4 = Array(added_layer.J₀⁺)
    #@show M1[1,1,1], M2[1,1,1], M3[1,1,1], M4[1,1,1]
    
    # @assert !any(isnan.(added_layer.t⁺⁺))
    
    # If this TOA, just copy the added layer into the bottom composite layer
    if (iz == 1)
        for ims=1:length(sensor_levels)
            if(sensor_levels[ims]==0)
                #@show sensor_levels[ims], iz, (iz==1)
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻)
            else
                #@show sensor_levels[ims], iz, (iz==1)
                composite_layer.topT⁺⁺[ims][:], composite_layer.topT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.topR⁻⁺[ims][:], composite_layer.topR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.topJ₀⁺[ims][:], composite_layer.topJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻)
            end
        end
    # If this is not the TOA, perform the interaction step
    else
        #@timeit "interaction_multisensor" interaction_multisensor!(RS_type, sensor_levels, scattering_interface, SFI, composite_layer, added_layer, I_static)
        for ims=1:length(sensor_levels)
            if sensor_levels[ims]==0
                #@show sensor_levels[ims], iz, (iz!=1)
                @timeit "interaction_multisensor" interaction_bot!(ims, 
                RS_type, 
                scattering_interface, 
                SFI, 
                composite_layer, 
                added_layer, 
                I_static,
                arr_type)
            else
                if sensor_levels[ims]==(iz-1) #include ims==Nz with ims==0
                    #@show sensor_levels[ims], iz, (iz==sensor_levels[ims]+1)
                    composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                        Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                    composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                        Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                    composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                        Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                elseif sensor_levels[ims]<(iz-1) 
                    #@show sensor_levels[ims], iz, (iz>sensor_levels[ims]+1)
                    @timeit "interaction_multisensor" interaction_bot!(ims, 
                                                                    RS_type, 
                                                                    scattering_interface, 
                                                                    SFI, 
                                                                    composite_layer, 
                                                                    added_layer, 
                                                                    I_static,
                                                                    arr_type)
                elseif sensor_levels[ims]>=iz 
                    #@show sensor_levels[ims], iz, (iz<=sensor_levels[ims])
                    @timeit "interaction_multisensor" interaction_top!(ims, 
                                                                    RS_type, 
                                                                    scattering_interface, 
                                                                    SFI, 
                                                                    composite_layer, 
                                                                    added_layer, 
                                                                    I_static,
                                                                    arr_type)    
                end
                if iz==2
                    M1 = (composite_layer.botT⁺⁺[1]);
                    M2 = (composite_layer.botR⁺⁻[1]);
                    M3 = (composite_layer.botT⁻⁻[1]);
                    M4 = (composite_layer.botR⁻⁺[1]);
                    M5 = (composite_layer.botJ₀⁻[1]);
                    M6 = (composite_layer.botJ₀⁺[1]);
                    @show M1[1,1,1], M2[1,1,1], M3[1,1,1], M4[1,1,1], M5[1,1,1], M6[1,1,1]
                end
            end
        end
    end
end

function rt_kernel_multisensor!(RS_type::Union{RRS{FT}, RRS_plus{FT}, VS_0to1_plus{FT}, VS_1to0_plus{FT}}, 
                                sensor_levels, 
                                pol_type, 
                                SFI, 
                                added_layer, 
                                composite_layer, 
                                computed_layer_properties::CoreScatteringOpticalProperties, 
                                scattering_interface, 
                                τ_sum,
                                m, 
                                quad_points, 
                                I_static, 
                                architecture, 
                                qp_μN, 
                                iz, 
                                arr_type)  where {FT}
    @unpack qp_μ, μ₀ = quad_points
    # Just unpack core optical properties from 
    @unpack τ, ϖ, Z⁺⁺, Z⁻⁺ = computed_layer_properties
    # SUNITI, check? Also, better to write function here
    dτ_max = minimum([maximum(τ .* ϖ), FT(0.001) * minimum(qp_μ)])
    _, ndoubl = doubling_number(dτ_max, maximum(τ .* ϖ))
    scatter = true # edit later
    #arr_type = array_type(architecture)
    # Compute dτ vector
    dτ = τ ./ 2^ndoubl
    expk = arr_type(exp.(-dτ /μ₀))

    @unpack Z⁺⁺_λ₁λ₀, Z⁻⁺_λ₁λ₀ = RS_type
    # If there is scattering, perform the elemental and doubling steps
    if scatter
        #@show τ, ϖ, RS_type.fscattRayl
        
        @timeit "elemental_inelastic" elemental_inelastic!(RS_type, 
                                                pol_type, SFI, 
                                                τ_sum, dτ, ϖ, 
                                                Z⁺⁺_λ₁λ₀, Z⁻⁺_λ₁λ₀, 
                                                m, ndoubl, scatter, 
                                                quad_points,  added_layer,  
                                                I_static, architecture)
        #println("Elemental inelastic done...")                                        
        @timeit "elemental" elemental!(pol_type, SFI, τ_sum, dτ, computed_layer_properties, m, ndoubl, scatter, quad_points,  added_layer,  architecture)
        #println("Elemental  done...")
        @timeit "doubling_inelastic" doubling_inelastic!(RS_type, pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
        #println("Doubling done...")
        #@timeit "doubling"   doubling!(pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
    else # This might not work yet on GPU!
        # If not, there is no reflectance. Assign r/t appropriately
        added_layer.r⁻⁺[:] .= 0;
        added_layer.r⁺⁻[:] .= 0;
        added_layer.J₀⁻[:] .= 0;
        added_layer.ier⁻⁺[:] .= 0;
        added_layer.ier⁺⁻[:] .= 0;
        added_layer.ieJ₀⁻[:] .= 0;
        added_layer.iet⁻⁻[:] .= 0;
        added_layer.iet⁺⁺[:] .= 0;
        added_layer.ieJ₀⁺[:] .= 0;
        temp = Array(exp.(-τ_λ./qp_μN'))
        #added_layer.t⁺⁺, added_layer.t⁻⁻ = (Diagonal(exp(-τ_λ / qp_μN)), Diagonal(exp(-τ_λ / qp_μN)))   
        for iλ = 1:length(τ_λ)
            added_layer.t⁺⁺[:,:,iλ] = Diagonal(temp[iλ,:]);
            added_layer.t⁻⁻[:,:,iλ] = Diagonal(temp[iλ,:]);
        end
    end

    # @assert !any(isnan.(added_layer.t⁺⁺))
    
    # If this TOA, just copy the added layer into the composite layer
    if (iz == 1)
        for ims=1:length(sensor_levels)
            if(sensor_levels[ims]==0)
                # bottom composite layer for TOA/BOA sensors
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.botieT⁺⁺[ims][:], composite_layer.botieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.botieR⁻⁺[ims][:], composite_layer.botieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.botieJ₀⁺[ims][:], composite_layer.botieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            else
                composite_layer.topT⁺⁺[ims][:], composite_layer.topT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.topR⁻⁺[ims][:], composite_layer.topR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.topJ₀⁺[ims][:], composite_layer.topJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.topieT⁺⁺[ims][:], composite_layer.topieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.topieR⁻⁺[ims][:], composite_layer.topieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.topieJ₀⁺[ims][:], composite_layer.topieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            end
        end
    # If this is not the TOA, perform the interaction step
    else
        #@timeit "interaction_multisensor" interaction_multisensor!(RS_type, sensor_levels, scattering_interface, SFI, composite_layer, added_layer, I_static)
        for ims=1:length(sensor_levels)
            if sensor_levels[ims]==(iz-1) #include ims==Nz with ims==0
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.botieT⁺⁺[ims][:], composite_layer.botieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.botieR⁻⁺[ims][:], composite_layer.botieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.botieJ₀⁺[ims][:], composite_layer.botieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            elseif sensor_levels[ims]<(iz-1) 
                @timeit "interaction_multisensor" interaction_bot!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)
            elseif sensor_levels[ims]>=iz 
                @timeit "interaction_multisensor" interaction_top!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)                
            end
        end
    end
end
#=
function rt_kernel_multisensor!(
            RS_type::Union{RRS_plus{FT}, VS_0to1_plus{FT}, VS_1to0_plus{FT}}, 
            sensor_levels,              
            pol_type, 
            SFI, 
            added_layer, 
            composite_layer, 
            computed_layer_properties::CoreScatteringOpticalProperties, 
            scattering_interface, 
            τ_sum,
            m, 
            quad_points, 
            I_static, 
            architecture, 
            qp_μN, 
            iz, 
            arr_type)  where {FT}
    @unpack qp_μ, μ₀ = quad_points
    # Just unpack core optical properties from 
    @unpack τ, ϖ, Z⁺⁺, Z⁻⁺ = computed_layer_properties
    # SUNITI, check? Also, better to write function here
    dτ_max = minimum([maximum(τ .* ϖ), FT(0.001) * minimum(qp_μ)])
    _, ndoubl = doubling_number(dτ_max, maximum(τ .* ϖ))
    scatter = true # edit later
    #arr_type = array_type(architecture)
    # Compute dτ vector
    dτ = τ ./ 2^ndoubl
    expk = arr_type(exp.(-dτ /μ₀))

    @unpack Z⁺⁺_λ₁λ₀, Z⁻⁺_λ₁λ₀ = RS_type
    # If there is scattering, perform the elemental and doubling steps
    if scatter
        #@show τ, ϖ, RS_type.fscattRayl
        
        @timeit "elemental_inelastic" elemental_inelastic!(RS_type, 
                                                pol_type, SFI, 
                                                τ_sum, dτ, ϖ, 
                                                Z⁺⁺_λ₁λ₀, Z⁻⁺_λ₁λ₀, 
                                                m, ndoubl, scatter, 
                                                quad_points,  added_layer,  
                                                I_static, architecture)
        #println("Elemental inelastic done...")                                        
        @timeit "elemental" elemental!(pol_type, SFI, τ_sum, dτ, computed_layer_properties, m, ndoubl, scatter, quad_points,  added_layer,  architecture)
        #println("Elemental  done...")
        @timeit "doubling_inelastic" doubling_inelastic!(RS_type, pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
        #println("Doubling done...")
        #@timeit "doubling"   doubling!(pol_type, SFI, expk, ndoubl, added_layer, I_static, architecture)
    else # This might not work yet on GPU!
        # If not, there is no reflectance. Assign r/t appropriately
        added_layer.r⁻⁺[:] .= 0;
        added_layer.r⁺⁻[:] .= 0;
        added_layer.J₀⁻[:] .= 0;
        added_layer.ier⁻⁺[:] .= 0;
        added_layer.ier⁺⁻[:] .= 0;
        added_layer.ieJ₀⁻[:] .= 0;
        added_layer.iet⁻⁻[:] .= 0;
        added_layer.iet⁺⁺[:] .= 0;
        added_layer.ieJ₀⁺[:] .= 0;
        temp = Array(exp.(-τ./qp_μN'))
        #added_layer.t⁺⁺, added_layer.t⁻⁻ = (Diagonal(exp(-τ_λ / qp_μN)), Diagonal(exp(-τ_λ / qp_μN)))   
        for iλ = 1:length(τ)
            added_layer.t⁺⁺[:,:,iλ] = Diagonal(temp[iλ,:]);
            added_layer.t⁻⁻[:,:,iλ] = Diagonal(temp[iλ,:]);
        end
    end

    # @assert !any(isnan.(added_layer.t⁺⁺))
    
    # If this TOA, just copy the added layer into the composite layer
    if (iz == 1)
        for ims=1:length(sensor_levels)
            if(ims==0) # bottom composite layer for satellite/groundbased sensors
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.botieT⁺⁺[ims][:], composite_layer.botieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.botieR⁻⁺[ims][:], composite_layer.botieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.botieJ₀⁺[ims][:], composite_layer.botieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            else
                composite_layer.topT⁺⁺[ims][:], composite_layer.topT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.topR⁻⁺[ims][:], composite_layer.topR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.topJ₀⁺[ims][:], composite_layer.topJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.topieT⁺⁺[ims][:], composite_layer.topieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.topieR⁻⁺[ims][:], composite_layer.topieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.topieJ₀⁺[ims][:], composite_layer.topieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            end
        end
        
    else # If this is not the TOA, perform the interaction step
        for ims=1:length(sensor_levels)
            if sensor_levels[ims]==(iz+1) #include ims==Nz with ims==0
                composite_layer.botT⁺⁺[ims][:], composite_layer.botT⁻⁻[ims][:] = 
                    Array(added_layer.t⁺⁺), Array(added_layer.t⁻⁻)
                composite_layer.botR⁻⁺[ims][:], composite_layer.botR⁺⁻[ims][:] = 
                    Array(added_layer.r⁻⁺), Array(added_layer.r⁺⁻)
                composite_layer.botJ₀⁺[ims][:], composite_layer.botJ₀⁻[ims][:] = 
                    Array(added_layer.J₀⁺), Array(added_layer.J₀⁻ )
                composite_layer.botieT⁺⁺[ims][:], composite_layer.botieT⁻⁻[ims][:] = 
                    Array(added_layer.iet⁺⁺), Array(added_layer.iet⁻⁻)
                composite_layer.botieR⁻⁺[ims][:], composite_layer.botieR⁺⁻[ims][:] = 
                    Array(added_layer.ier⁻⁺), Array(added_layer.ier⁺⁻)
                composite_layer.botieJ₀⁺[ims][:], composite_layer.botieJ₀⁻[ims][:] = 
                    Array(added_layer.ieJ₀⁺), Array(added_layer.ieJ₀⁻ )
            elseif sensor_levels[ims]>(iz+1) 
                @timeit "interaction_multisensor" interaction_bot!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)
            elseif sensor_levels[ims]<=iz 
                @timeit "interaction_multisensor" interaction_top!(ims, 
                                                                RS_type, 
                                                                scattering_interface, 
                                                                SFI, 
                                                                composite_layer, 
                                                                added_layer, 
                                                                I_static,
                                                                arr_type)                
            end
        end
    end
end
=#