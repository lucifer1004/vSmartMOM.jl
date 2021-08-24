function batched_mul(A::CuArray{ForwardDiff.Dual{T,V,N},3}, B::CuArray{ForwardDiff.Dual{T,V,N},3}) where {T,V,N}
    # Extract values:
    Av = ForwardDiff.value.(A)
    Bv = ForwardDiff.value.(B)
    
    Cv = Av ⊠ Bv
    # Compute derivatives ∂(AB)/∂x = A * ∂B/∂x + ∂A/∂x * B;
    dABdx = [Av ⊠ ForwardDiff.partials.(B,i) + ForwardDiff.partials.(A,i) ⊠ Bv for i=1:N];
    #synchronize()
    #CUDA.@time batch_jacobian_mul(Av, Bv,dA, dB) 
    dABdx = ForwardDiff.Partials.(tuple.(dABdx...));
    return  eltype(A).(Cv,dABdx);
end

# Define batched matrix multiply for GPU and Duals:
function batched_mul2(A::CuArray{ForwardDiff.Dual{T,V,N},3}, B::CuArray{ForwardDiff.Dual{T,V,N},3}) where {T,V,N}
    # Extract values:
    Av = ForwardDiff.value.(A)
    Bv = ForwardDiff.value.(B)
    # Use strided batch for A*B (defined as gemm_strided_batched):
    Cv = Av ⊠ Bv
    # Compute derivatives ∂(AB)/∂x = A * ∂B/∂x + ∂A/∂x * B;
    println("Mat")
    CUDA.@time dA  = Tuple(ForwardDiff.partials.(A,i) for i in 1:N)
    CUDA.@time dB  = Tuple(ForwardDiff.partials.(B,i) for i in 1:N)
    Kout = deepcopy(dA)
    for i=1:length(dA)
        CUDA.@time multi_test_add!(Av, Bv, dA[i], dB[i],Kout[i])
    end
    #CUDA.@time dABdx = [Av ⊠ ForwardDiff.partials.(B,i) + ForwardDiff.partials.(A,i) ⊠ Bv for i=1:N];
    CUDA.@time dABdx = ForwardDiff.Partials.(tuple.(Kout...));
    return eltype(A).(Cv,dABdx);
end



# Overload of batch_inv! for Dual numbers
function batch_inv!(X::CuArray{ForwardDiff.Dual{T,V,N},3}, A::CuArray{ForwardDiff.Dual{T,V,N},3}) where {T,V,N}
    # AoS -> Array
    #@show T,V,N
    InArray  = reinterpret(V, A);
    OutArray = reinterpret(V, X);
    invA = OutArray[1:N+1:end,:,:]
    @inbounds batch_inv!(invA,InArray[1:N+1:end,:,:])
    @inbounds for i=1:N
        ∂A∂x = InArray[i+1:N+1:end,:,:]
        OutArray[i+1:N+1:end,:,:] = -∂A∂x ⊠ invA ⊠ ∂A∂x
    end
    
    X .= reinterpret(eltype(A), OutArray)
    #synchronize()
    return nothing
end

function batch_jacobian_mul(A::CuArray{FT,3}, B::CuArray{FT,3},dA, dB) where {FT}
    ∂AB = deepcopy(dA)
    for i=1:length(dA)
        #∂AB[i] = A ⊠ dB[i] + dA[i] ⊠ B
        multi_test_2!(A,B,dA[i],dB[i],∂AB[i])
    end
    return ∂AB
end

function batch_jacobian_inv(Fout::CuArray{FT,3},  Fin::CuArray{FT,3}, Kin, C) where {FT}
    #C = similar(Fin)
    #D = similar(Fin)

    Kout = deepcopy(Kin)
    batch_inv!(Fout,Fin)
    #-Fout ⊠ Kin[1] ⊠ Fout
    
    for i=1:length(Kin)
        if !all(==(0),Kin[i])
            multi_test!(Fout,Kin[i],C,Kout[i])
        else
            Kout[i] .= 0
        end
    end
    return Kout
end

# Overload of batch_inv! for Dual numbers
function batch_jacobian_inv(Fout::CuArray{FT,3},  Fin::CuArray{FT,3}, Kin) where {FT}
    RadiativeTransfer.vSmartMOM.batch_inv!(Fout,Fin)
    # Compute all partial derivatives here:
    Tuple(multi_test(Fout,Kin[i]) for i in 1:length(Kin))
    #Tuple(-Fout ⊠ Kin[i] ⊠ Fout for i in 1:length(Kin))
end

function multi_test!(A::CuArray{FT,3},B::CuArray{FT,3},C::CuArray{FT,3},D::CuArray{FT,3}) where {FT}
    CUBLAS.gemm_strided_batched!('N', 'N', FT(-1), A, B, FT(0), C)
    CUBLAS.gemm_strided_batched!('N', 'N', FT(1), C, A, FT(0), D)
    return nothing
end




