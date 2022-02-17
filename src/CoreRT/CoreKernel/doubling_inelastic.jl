#=
 
This file contains RT doubling-related functions
 
=#

"""
    $(FUNCTIONNAME)(pol_type, SFI, expk, ndoubl::Int, added_layer::AddedLayer, I_static::AbstractArray{FT}, 
                    architecture) where {FT}

Compute homogenous layer matrices from its elemental layer using Doubling 
"""
function doubling_helper!(RS_type::RRS,
    pol_type, 
    SFI, 
    expk, 
    ndoubl::Int, 
    added_layer::Union{AddedLayer,AddedLayerRS},
    I_static::AbstractArray{FT}, 
    architecture) where {FT}

    # Unpack the added layer
    @unpack i_λ₁λ₀ = RS_type 
    @unpack r⁺⁻, r⁻⁺, t⁻⁻, t⁺⁺, J₀⁺, J₀⁻ = added_layer
    @unpack ier⁺⁻, ier⁻⁺, iet⁻⁻, iet⁺⁺, ieJ₀⁺, ieJ₀⁻ = added_layer
    # Device architecture
    dev = devi(architecture)

    # Note: short-circuit evaluation => return nothing evaluated iff ndoubl == 0 
    ndoubl == 0 && return nothing
    nQuad, _, nSpec = size(r⁺⁻)
    nRaman = length(i_λ₁λ₀);
    # Geometric progression of reflections (1-RR)⁻¹
    gp_refl      = similar(t⁺⁺)
    tt⁺⁺_gp_refl = similar(t⁺⁺)

    if SFI
        # Dummy for source 
        J₁⁺ = similar(J₀⁺)
        # Dummy for J
        J₁⁻ = similar(J₀⁻)

        # Dummy for source 
        ieJ₁⁺ = similar(ieJ₀⁺)
        # Dummy for J
        ieJ₁⁻ = similar(ieJ₀⁻)
    end

    # Loop over number of doublings
    for n = 1:ndoubl

        # T⁺⁺(λ)[I - R⁺⁻(λ)R⁻⁺(λ)]⁻¹, for doubling R⁺⁻,R⁻⁺ and T⁺⁺,T⁻⁻ is identical
        batch_inv!(gp_refl, I_static .- r⁻⁺ ⊠ r⁻⁺)
        @views tt⁺⁺_gp_refl[:] = t⁺⁺ ⊠ gp_refl
        if SFI
            # J⁺₂₁(λ) = J⁺₁₀(λ).exp(-τ(λ)/μ₀)
            @views J₁⁺[:,1,:] = J₀⁺[:,1,:] .* expk'
            @views ieJ₁⁺[:,1,:,:] = ieJ₀⁺[:,1,:,:] .* expk'

            # J⁻₁₂(λ)  = J⁻₀₁(λ).exp(-τ(λ)/μ₀)
            @views J₁⁻[:,1,:] = J₀⁻[:,1,:] .* expk'
            @views ieJ₁⁻[:,1,:,:] = ieJ₀⁻[:,1,:,:] .* expk'
            #@show size(ieJ₁⁺)
            @timeit "precomp" tmp1 = gp_refl ⊠  (J₀⁺ + r⁻⁺ ⊠ J₁⁻)
            @timeit "precomp" tmp2 = gp_refl ⊠  (J₁⁻ + r⁻⁺ ⊠ J₀⁺)
            #@timeit "prep"    tmp3 = repeat(r⁻⁺,1,1,1,nRaman) ⊠ reshape(ieJ₁⁻, 
            for Δn = 1:nRaman
                n₀, n₁ = get_n₀_n₁(ieJ₁⁺,i_λ₁λ₀[Δn])
                #@timeit "mati" @views tmp3 = ier⁻⁺[:,:,n₁,Δn] ⊠ J₁⁻[:,:,n₀]
                #@timeit "mati" @views tmp4 = r⁻⁺[:,:,n₁] ⊠ ier⁻⁺[:,:,n₁,Δn] 
                #@timeit "mati" @views tmp5 = ier⁻⁺[:,:,n₁,Δn] ⊠ r⁻⁺[:,:,n₀]
                #@show size(tmp3)
                #@show length(n₁), length(n₀), length(n₁_), length(n₀_)
                @inbounds @views ieJ₀⁺[:,:,n₁,Δn] = 
                                ieJ₁⁺[:,:,n₁,Δn] + 
                                (tt⁺⁺_gp_refl[:,:,n₁] ⊠ 
                                (ieJ₀⁺[:,:,n₁,Δn] + 
                                r⁻⁺[:,:,n₁] ⊠ ieJ₁⁻[:,:,n₁,Δn] + 
                                ier⁻⁺[:,:,n₁,Δn] ⊠ J₁⁻[:,:,n₀] + 
                                (r⁻⁺[:,:,n₁] ⊠ ier⁻⁺[:,:,n₁,Δn] + 
                                ier⁻⁺[:,:,n₁,Δn] ⊠ r⁻⁺[:,:,n₀]) ⊠ 
                                tmp1[:,:,n₀])) + 
                                iet⁺⁺[:,:,n₁,Δn] ⊠ tmp1[:,:,n₀]
                @inbounds @views ieJ₀⁻[:,:,n₁,Δn] = 
                                ieJ₀⁻[:,:,n₁,Δn] + 
                                (tt⁺⁺_gp_refl[:,:,n₁] ⊠ 
                                (ieJ₁⁻[:,:,n₁,Δn] + 
                                ier⁻⁺[:,:,n₁,Δn] ⊠ J₀⁺[:,:,n₀] +
                                r⁻⁺[:,:,n₁] ⊠ ieJ₀⁺[:,:,n₁,Δn] + 
                                (ier⁻⁺[:,:,n₁,Δn] ⊠ r⁻⁺[:,:,n₀] + 
                                r⁻⁺[:,:,n₁] ⊠ ier⁻⁺[:,:,n₁,Δn]) ⊠ 
                                tmp2[:,:,n₀])) +
                                iet⁻⁻[:,:,n₁,Δn] ⊠ tmp2[:,:,n₀]
            end
            
        
            # J⁻₀₂(λ) = J⁻₀₁(λ) + T⁻⁻₀₁(λ)[I - R⁻⁺₂₁(λ)R⁺⁻₀₁(λ)]⁻¹[J⁻₁₂(λ) + R⁻⁺₂₁(λ)J⁺₁₀(λ)] (see Eqs.8 in Raman paper draft)
            J₀⁻[:] = J₀⁻ + (tt⁺⁺_gp_refl ⊠ (J₁⁻ + r⁻⁺ ⊠ J₀⁺)) 

            # J⁺₂₀(λ) = J⁺₂₁(λ) + T⁺⁺₂₁(λ)[I - R⁺⁻₀₁(λ)R⁻⁺₂₁(λ)]⁻¹[J⁺₁₀(λ) + R⁺⁻₀₁(λ)J⁻₁₂(λ)] (see Eqs.8 in Raman paper draft)
            J₀⁺[:] = J₁⁺ + (tt⁺⁺_gp_refl ⊠ (J₀⁺ + r⁻⁺ ⊠ J₁⁻))

            expk[:] = expk.^2
        end  
        #println("Doubling part 1 done")
        for Δn = 1:nRaman
                n₀, n₁ = get_n₀_n₁(ieJ₁⁺,i_λ₁λ₀[Δn])
                #@show n₁, n₀
                #@show length(n₀)
                @timeit "n loop 2" @inbounds @views iet⁺⁺[:,:,n₁,Δn] = t⁺⁺[:,:,n₁] ⊠ gp_refl[:,:,n₁] ⊠ 
                        (iet⁺⁺[:,:,n₁,Δn] + 
                        (ier⁻⁺[:,:,n₁,Δn] ⊠ r⁻⁺[:,:,n₀] + r⁻⁺[:,:,n₁] ⊠ ier⁻⁺[:,:,n₁,Δn]) ⊠ 
                        gp_refl[:,:,n₀] ⊠ t⁺⁺[:,:,n₀]) + 
                        iet⁺⁺[:,:,n₁,Δn] ⊠ gp_refl[:,:,n₀] ⊠  t⁺⁺[:,:,n₀]
                @timeit "n loop 2" @inbounds @views ier⁻⁺[:,:,n₁,Δn] = ier⁻⁺[:,:,n₁,Δn] + 
                        t⁺⁺[:,:,n₁] ⊠ gp_refl[:,:,n₁] ⊠ r⁻⁺[:,:,n₁] ⊠  
                        (iet⁺⁺[:,:,n₁,Δn] + 
                        (ier⁻⁺[:,:,n₁,Δn] ⊠ r⁻⁺[:,:,n₀] + r⁻⁺[:,:,n₁] ⊠ ier⁻⁺[:,:,n₁,Δn]) ⊠ 
                        gp_refl[:,:,n₀] ⊠ t⁺⁺[:,:,n₀]) + 
                        (iet⁺⁺[:,:,n₁,Δn] ⊠ gp_refl[:,:,n₀] ⊠ r⁻⁺[:,:,n₀] + 
                        t⁺⁺[:,:,n₁] ⊠ gp_refl[:,:,n₁] ⊠ ier⁻⁺[:,:,n₁,Δn]) ⊠ t⁺⁺[:,:,n₀]
        end
        
        # R⁻⁺₂₀(λ) = R⁻⁺₁₀(λ) + T⁻⁻₀₁(λ)[I - R⁻⁺₂₁(λ)R⁺⁻₀₁(λ)]⁻¹R⁻⁺₂₁(λ)T⁺⁺₁₀(λ) (see Eqs.8 in Raman paper draft)
        r⁻⁺[:]  = r⁻⁺ + (tt⁺⁺_gp_refl ⊠ r⁻⁺ ⊠ t⁺⁺)

        # T⁺⁺₂₀(λ) = T⁺⁺₂₁(λ)[I - R⁺⁻₀₁(λ)R⁻⁺₂₁(λ)]⁻¹T⁺⁺₁₀(λ) (see Eqs.8 in Raman paper draft)
        t⁺⁺[:]  = tt⁺⁺_gp_refl ⊠ t⁺⁺
    end

    # After doubling, revert D(DR)->R, where D = Diagonal{1,1,-1,-1}
    # For SFI, after doubling, revert D(DJ₀⁻)->J₀⁻

    synchronize_if_gpu()

    apply_D_matrix!(pol_type.n, added_layer.r⁻⁺, added_layer.t⁺⁺, added_layer.r⁺⁻, added_layer.t⁻⁻)
    apply_D_matrix_IE!(RS_type, pol_type.n, added_layer.ier⁻⁺, added_layer.iet⁺⁺, added_layer.ier⁺⁻, added_layer.iet⁻⁻)
    SFI && apply_D_matrix_SFI!(pol_type.n, added_layer.J₀⁻)
    SFI && apply_D_matrix_SFI_IE!(RS_type, pol_type.n, added_layer.ieJ₀⁻)

    return nothing 
end


function doubling_helper!(RS_type::Union{VS_0to1_plus, VS_1to0_plus, RRS_VS_0to1_plus, RRS_VS_1to0_plus},
                        pol_type, 
                        SFI, 
                        expk, 
                        ndoubl::Int, 
                        added_layer::Union{AddedLayer,AddedLayerRS},
                        I_static::AbstractArray{FT}, 
                        architecture) where {FT}
    # Unpack the added layer
    @unpack i_λ₁λ₀_all, i_ref = RS_type 
    @unpack r⁺⁻, r⁻⁺, t⁻⁻, t⁺⁺, J₀⁺, J₀⁻ = added_layer
    @unpack ier⁺⁻, ier⁻⁺, iet⁻⁻, iet⁺⁺, ieJ₀⁺, ieJ₀⁻ = added_layer
    # Device architecture
    dev = devi(architecture)

    # Note: short-circuit evaluation => return nothing evaluated iff ndoubl == 0 
    ndoubl == 0 && return nothing
    nQuad, _, nSpec = size(r⁺⁻)
    # Geometric progression of reflections (1-RR)⁻¹
    gp_refl      = similar(t⁺⁺)
    tt⁺⁺_gp_refl = similar(t⁺⁺)
 
    if SFI
        # Dummy for source 
        J₁⁺ = similar(J₀⁺)
        # Dummy for J
        J₁⁻ = similar(J₀⁻)

        # Dummy for source 
        ieJ₁⁺ = similar(ieJ₀⁺)
        # Dummy for J
        ieJ₁⁻ = similar(ieJ₀⁻)
    end

    # Loop over number of doublings
    for n = 1:ndoubl
        
        # T⁺⁺(λ)[I - R⁺⁻(λ)R⁻⁺(λ)]⁻¹, for doubling R⁺⁻,R⁻⁺ and T⁺⁺,T⁻⁻ is identical
        batch_inv!(gp_refl, I_static .- r⁻⁺ ⊠ r⁻⁺)
        @views tt⁺⁺_gp_refl[:] = t⁺⁺ ⊠ gp_refl

        if SFI

            # J⁺₂₁(λ) = J⁺₁₀(λ).exp(-τ(λ)/μ₀)
            @views J₁⁺[:,1,:] = J₀⁺[:,1,:] .* expk'
            @views ieJ₁⁺[:,1,:,1] = ieJ₀⁺[:,1,:,1] .* expk'

            # J⁻₁₂(λ)  = J⁻₀₁(λ).exp(-τ(λ)/μ₀)
            @views J₁⁻[:,1,:]   = J₀⁻[:,1,:] .* expk'
            @views ieJ₁⁻[:,1,:,1] = ieJ₀⁻[:,1,:,1] .* expk'

            tmp1 = gp_refl ⊠  (J₀⁺ + r⁻⁺ ⊠ J₁⁻)
            tmp2 = gp_refl ⊠  (J₁⁻ + r⁻⁺ ⊠ J₀⁺)
            #for n₁ in eachindex ieJ₁⁺[1,1,:,1]
            for Δn in length(i_λ₁λ₀_all)
                n₁ = i_λ₁λ₀_all[Δn]
                n₀ = i_ref
                if n₁>0
                    # J⁺₂₀(λ) = J⁺₂₁(λ) + T⁺⁺₂₁(λ)[I - R⁺⁻₀₁(λ)R⁻⁺₂₁(λ)]⁻¹[J⁺₁₀(λ) + R⁺⁻₀₁(λ)J⁻₁₂(λ)] (see Eqs.16 in Raman paper draft)
                    @inbounds @views ieJ₀⁺[:,:,n₁,1] = 
                            ieJ₁⁺[:,:,n₁,1] + 
                            (tt⁺⁺_gp_refl[:,:,n₁] * 
                            (ieJ₀⁺[:,:,n₁,1] + 
                            r⁻⁺[:,:,n₁] * ieJ₁⁻[:,:,n₁,1] + 
                            ier⁻⁺[:,:,n₁,1] * J₁⁻[:,:,n₀] + 
                            (r⁻⁺[:,:,n₁] * ier⁻⁺[:,:,n₁,1] + 
                            ier⁻⁺[:,:,n₁,1] * r⁻⁺[:,:,n₀]) * 
                            tmp1[:,:,n₀])) + 
                            iet⁺⁺[:,:,n₁,1] * tmp1[:,:,n₀];  
            
                    # J⁻₀₂(λ) = J⁻₀₁(λ) + T⁻⁻₀₁(λ)[I - R⁻⁺₂₁(λ)R⁺⁻₀₁(λ)]⁻¹[J⁻₁₂(λ) + R⁻⁺₂₁(λ)J⁺₁₀(λ)] (see Eqs.17 in Raman paper draft)
                    @inbounds @views ieJ₀⁻[:,1,n₁,1] = 
                            ieJ₀⁻[:,1,n₁,1] + 
                            (tt⁺⁺_gp_refl[:,:,n₁] * 
                            (ieJ₁⁻[:,1,n₁,1] + 
                            ier⁻⁺[:,:,n₁,1] * J₀⁺[:,1,n₀] + 
                            r⁻⁺[:,:,n₁] * ieJ₀⁺[:,1,n₁,1] + 
                            (ier⁻⁺[:,:,n₁,1] * r⁻⁺[:,:,n₀] + 
                            r⁻⁺[:,:,n₁] * ier⁻⁺[:,:,n₁,1]) *
                            tmp2[:,:,n₀])) +
                            iet⁻⁻[:,:,n₁,1] * tmp2[:,:,n₀]
                end
            end            
            # J⁻₀₂(λ) = J⁻₀₁(λ) + T⁻⁻₀₁(λ)[I - R⁻⁺₂₁(λ)R⁺⁻₀₁(λ)]⁻¹[J⁻₁₂(λ) + R⁻⁺₂₁(λ)J⁺₁₀(λ)] (see Eqs.8 in Raman paper draft)
            J₀⁻[:] = J₀⁻ + (tt⁺⁺_gp_refl ⊠ (J₁⁻ + r⁻⁺ ⊠ J₀⁺)) 

            # J⁺₂₀(λ) = J⁺₂₁(λ) + T⁺⁺₂₁(λ)[I - R⁺⁻₀₁(λ)R⁻⁺₂₁(λ)]⁻¹[J⁺₁₀(λ) + R⁺⁻₀₁(λ)J⁻₁₂(λ)] (see Eqs.8 in Raman paper draft)
            J₀⁺[:] = J₁⁺ + (tt⁺⁺_gp_refl ⊠ (J₀⁺ + r⁻⁺ ⊠ J₁⁻))
             
            expk[:] = expk.^2
        end  

        #for n₁ in eachindex ieJ₁⁺[1,1,:,1]
        tmp1 = gp_refl ⊠ t⁺⁺
        for Δn = 1:length(i_λ₁λ₀_all)
            n₁ = i_λ₁λ₀_all[Δn]
            n₀ = i_ref
            if n₁>0
                # (see Eqs.12 in Raman paper draft)
                @inbounds @views iet⁺⁺[:,:,n₁,1] = tt⁺⁺_gp_refl[:,:,n₁] * 
                        (iet⁺⁺[:,:,n₁,1] + 
                        (ier⁻⁺[:,:,n₁,1] * r⁻⁺[:,:,n₀] + 
                        r⁻⁺[:,:,n₁] * ier⁻⁺[:,:,n₁,1]) * 
                        tmp1[:,:,n₀]) + 
                        iet⁺⁺[:,:,n₁,1] * tmp1[:,:,n₀]

                # (see Eqs.14 in Raman paper draft)
                @inbounds @views ier⁻⁺[:,:,n₁,1] = ier⁻⁺[:,:,n₁,1] + 
                        tt⁺⁺_gp_refl[:,:,n₁] * r⁻⁺[:,:,n₁] * 
                        (iet⁺⁺[:,:,n₁,1] + 
                        (ier⁻⁺[:,:,n₁,1] * r⁻⁺[:,:,n₀] + 
                        r⁻⁺[:,:,n₁] * ier⁻⁺[:,:,n₁,1]) * 
                        gp_refl[:,:,n₀] * t⁺⁺[:,:,n₀]) + 
                        (iet⁺⁺[:,:,n₁,1] * gp_refl[:,:,n₀] * r⁻⁺[:,:,n₀] + 
                        tt⁺⁺_gp_refl[:,:,n₁] * ier⁻⁺[:,:,n₁,1]) * t⁺⁺[:,:,n₀]
            end
        end
    
        # R⁻⁺₂₀(λ) = R⁻⁺₁₀(λ) + T⁻⁻₀₁(λ)[I - R⁻⁺₂₁(λ)R⁺⁻₀₁(λ)]⁻¹R⁻⁺₂₁(λ)T⁺⁺₁₀(λ) (see Eqs.8 in Raman paper draft)
        r⁻⁺[:]  = r⁻⁺ + (tt⁺⁺_gp_refl ⊠ r⁻⁺ ⊠ t⁺⁺)

        # T⁺⁺₂₀(λ) = T⁺⁺₂₁(λ)[I - R⁺⁻₀₁(λ)R⁻⁺₂₁(λ)]⁻¹T⁺⁺₁₀(λ) (see Eqs.8 in Raman paper draft)
        t⁺⁺[:]  = tt⁺⁺_gp_refl ⊠ t⁺⁺
    end

    # After doubling, revert D(DR)->R, where D = Diagonal{1,1,-1,-1}
    # For SFI, after doubling, revert D(DJ₀⁻)->J₀⁻

    synchronize_if_gpu()

    apply_D_matrix!(pol_type.n, 
        added_layer.r⁻⁺, added_layer.t⁺⁺, added_layer.r⁺⁻, added_layer.t⁻⁻)
    apply_D_matrix_IE!(RS_type, pol_type.n, 
        added_layer.ier⁻⁺, added_layer.iet⁺⁺, added_layer.ier⁺⁻, added_layer.iet⁻⁻)
    SFI && apply_D_matrix_SFI!(pol_type.n, added_layer.J₀⁻)
    SFI && apply_D_matrix_SFI_IE!(RS_type, pol_type.n, added_layer.ieJ₀⁻)
    
    return nothing 
end

function doubling_inelastic!(RS_type, 
                    pol_type, SFI, 
                    expk, ndoubl::Int, 
                    added_layer::Union{AddedLayer,AddedLayerRS},#{FT},
                    I_static::AbstractArray{FT}, 
                    architecture) where {FT}

    doubling_helper!(RS_type, 
                pol_type, SFI, 
                expk, ndoubl, 
                added_layer, 
                I_static, 
                architecture)

    synchronize_if_gpu()
end

@kernel function apply_D_IE_RRS!(i_λ₁λ₀,n_stokes,  
                        ier⁻⁺, iet⁺⁺, ier⁺⁻, iet⁻⁻)
    iμ, jμ, n, Δn  = @index(Global, NTuple)
    #@unpack i_λ₁λ₀ = RS_type 
    n₀  = n + i_λ₁λ₀[Δn]
    if 1 ≤ n₀ ≤ size(ier⁻⁺,4)
        i = mod(iμ, n_stokes)
        j = mod(jμ, n_stokes)
        if (i > 2)
            ier⁻⁺[iμ,jμ,n,n₀] = - ier⁻⁺[iμ, jμ, n, n₀]
        end
        
        if ((i <= 2) & (j <= 2)) | ((i > 2) & (j > 2))
            ier⁺⁻[iμ,jμ,n,n₀] = ier⁻⁺[iμ,jμ,n,n₀]
            iet⁻⁻[iμ,jμ,n,n₀] = iet⁺⁺[iμ,jμ,n,n₀]
        else
            ier⁺⁻[iμ,jμ,n,n₀] = - ier⁻⁺[iμ,jμ,n,n₀]
            iet⁻⁻[iμ,jμ,n,n₀] = - iet⁺⁺[iμ,jμ,n,n₀]
        end
    end
end

@kernel function apply_D_IE_VS!(i_λ₁λ₀_all, n_stokes,  
                        ier⁻⁺, iet⁺⁺, ier⁺⁻, iet⁻⁻)
    iμ, jμ, Δn  = @index(Global, NTuple)
    #@unpack i_λ₁λ₀ = RS_type 
    n  = i_λ₁λ₀_all[Δn]
    i = mod(iμ, n_stokes)
    j = mod(jμ, n_stokes)

    if (i > 2)
        ier⁻⁺[iμ,jμ,n,1] = - ier⁻⁺[iμ, jμ, n, 1]
    end
    
    if ((i <= 2) & (j <= 2)) | ((i > 2) & (j > 2))
        ier⁺⁻[iμ,jμ,n,1] = ier⁻⁺[iμ,jμ,n,1]
        iet⁻⁻[iμ,jμ,n,1] = iet⁺⁺[iμ,jμ,n,1]
    else
        ier⁺⁻[iμ,jμ,n,1] = - ier⁻⁺[iμ,jμ,n,1]
        iet⁻⁻[iμ,jμ,n,1] = - iet⁺⁺[iμ,jμ,n,1]
    end

end

#@kernel function apply_D_SFI!(n_stokes::Int, J₀⁻)
#    iμ, _, n = @index(Global, NTuple)
#    i = mod(iμ, n_stokes)
#
#    if (i > 2)
#        J₀⁻[iμ, 1, n] = - J₀⁻[iμ, 1, n] 
#    end
#end

# Kernel for RRS
@kernel function apply_D_SFI_IE_RRS!(i_λ₁λ₀, n_stokes::Int, ieJ₀⁻)
    iμ, n, Δn = @index(Global, NTuple)
    #@unpack i_λ₁λ₀ = RS_type
    n₀ = n + i_λ₁λ₀[Δn] 
    if 1 ≤ n₀ ≤ size(ieJ₀⁻,4)
        i = mod(iμ, n_stokes)

        if (i > 2)
            ieJ₀⁻[iμ, 1, n, n₀] = - ieJ₀⁻[iμ, 1, n, Δn] 
        end
    end
end

# Kernel for VRS
@kernel function apply_D_SFI_IE_VS!(i_λ₁λ₀_all, 
                                n_stokes::Int, ieJ₀⁻)
    iμ, _, Δn = @index(Global, NTuple)
    #@unpack i_λ₁λ₀ = RS_type
    n = i_λ₁λ₀_all[Δn] 
    i = mod(iμ, n_stokes)

    if (i > 2)
        ieJ₀⁻[iμ, 1, n, 1] = - ieJ₀⁻[iμ, 1, n, 1] 
    end
end

#Suniti: is it possible to  use the same kernel for the 3D elastic and 4D inelastic terms or do we need to call two different kernels separately? 
#function apply_D_matrix!(n_stokes::Int, r⁻⁺::CuArray{FT,3}, t⁺⁺::CuArray{FT,3}, r⁺⁻::CuArray{FT,3}, t⁻⁻::CuArray{FT,3}) where {FT}
#    
#    if n_stokes == 1
#        r⁺⁻[:] = r⁻⁺
#        t⁻⁻[:] = t⁺⁺    
#        
#        return nothing
#    else 
#        device = devi(architecture(r⁻⁺))
#        applyD_kernel! = apply_D!(device)
#        event = applyD_kernel!(n_stokes, r⁻⁺, t⁺⁺, r⁺⁻, t⁻⁻, ndrange=size(r⁻⁺)); #Suniti: is it possible to  use the same kernel for the 3D elastic and 4D inelastic terms or do we need to call two different kernels separately? 
#        wait(device, event);
#        synchronize_if_gpu();
#        return nothing
#    end
#end

function apply_D_matrix_IE!(RS_type::Union{VS_0to1_plus, VS_1to0_plus, RRS_VS_0to1_plus, RRS_VS_1to0_plus}, n_stokes::Int, ier⁻⁺::AbstractArray{FT,4}, iet⁺⁺::AbstractArray{FT,4}, ier⁺⁻::AbstractArray{FT,4}, iet⁻⁻::AbstractArray{FT,4}) where {FT}
    if n_stokes == 1
        ier⁺⁻[:] = ier⁻⁺
        iet⁻⁻[:] = iet⁺⁺  
        return nothing
    else 
        device = devi(architecture(ier⁻⁺))
        aType = array_type(architecture(ier⁻⁺))
        applyD_kernel_IE! = apply_D_IE_VS!(device)
        event = applyD_kernel_IE!(aType(RS_type.i_λ₁λ₀_all), n_stokes, 
            ier⁻⁺, iet⁺⁺, ier⁺⁻, iet⁻⁻, ndrange=getKernelDim(RS_type, ier⁻⁺));
        wait(device, event);
        synchronize();
        return nothing
    end
end

function apply_D_matrix_IE!(RS_type::RRS, n_stokes::Int, ier⁻⁺::AbstractArray{FT,4}, iet⁺⁺::AbstractArray{FT,4}, ier⁺⁻::AbstractArray{FT,4}, iet⁻⁻::AbstractArray{FT,4}) where {FT}
    if n_stokes == 1
        ier⁺⁻[:] = ier⁻⁺
        iet⁻⁻[:] = iet⁺⁺  
        return nothing
    else 
        device = devi(architecture(ier⁻⁺))
        aType = array_type(architecture(ier⁻⁺))
        applyD_kernel_IE! = apply_D_IE_RRS!(device)
        event = applyD_kernel_IE!(aType(RS_type.i_λ₁λ₀), n_stokes, 
            ier⁻⁺, iet⁺⁺, ier⁺⁻, iet⁻⁻, ndrange=getKernelDim(RS_type, ier⁻⁺));
        wait(device, event);
        synchronize();
        return nothing
    end
end

#function apply_D_matrix_SFI!(n_stokes::Int, J₀⁻::CuArray{FT,3}) where {FT}
#
#    n_stokes == 1 && return nothing
#    device = devi(architecture(J₀⁻)) #Suniti: how to do this so that ieJ₀⁻ can also be included?
#    applyD_kernel! = apply_D_SFI!(device)
#    event = applyD_kernel!(n_stokes, J₀⁻, ndrange=size(J₀⁻));
#    wait(device, event);
#    synchronize();
#    
#    return nothing
#end

# For RRS
function apply_D_matrix_SFI_IE!(RS_type::RRS, n_stokes::Int, ieJ₀⁻::AbstractArray{FT,4}) where {FT}
    n_stokes == 1 && return nothing
    device = devi(architecture(ieJ₀⁻))
    aType = array_type(architecture(ieJ₀⁻))
    applyD_kernel_IE! = apply_D_SFI_IE_RRS!(device)
    event = applyD_kernel_IE!(aType(RS_type.i_λ₁λ₀),n_stokes, 
                    ieJ₀⁻, ndrange=(size(ieJ₀⁻,1), size(ieJ₀⁻,3), size(ieJ₀⁻,4)));
    wait(device, event);
    synchronize_if_gpu()
    return nothing
end

# For S_0to1 and VS_1to0
function apply_D_matrix_SFI_IE!(RS_type::Union{VS_0to1_plus, VS_1to0_plus, RRS_VS_0to1_plus, RRS_VS_1to0_plus}, n_stokes::Int, ieJ₀⁻::AbstractArray{FT,4}) where {FT}
    n_stokes == 1 && return nothing
    device = devi(architecture(ieJ₀⁻))
    aType = array_type(architecture(ieJ₀⁻))
    applyD_kernel_IE! = apply_D_SFI_IE_VS!(device)
    event = applyD_kernel_IE!(aType(RS_type.i_λ₁λ₀_all), n_stokes, 
                    ieJ₀⁻, ndrange=getKernelDimSFI(RS_type, ieJ₀⁻));
    wait(device, event);
    synchronize_if_gpu()
    return nothing
end

