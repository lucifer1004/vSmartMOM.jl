
Nquad = 12;
Nspec = 20000;
FinCPU = randn(Nquad,Nquad,Nspec);
FoutCPU = randn(Nquad,Nquad,Nspec);
FT = Float64;
Fin = CuArray{FT}(FinCPU);
Fout = similar(Fin);
K = (CuArray{FT}(randn(Nquad,Nquad,Nspec)),CuArray{FT}(randn(Nquad,Nquad,Nspec)),CuArray{FT}(randn(Nquad,Nquad,Nspec)),CuArray{FT}(randn(Nquad,Nquad,Nspec)));
KCPU  = ((randn(Nquad,Nquad,Nspec)),(randn(Nquad,Nquad,Nspec)),(randn(Nquad,Nquad,Nspec)),(randn(Nquad,Nquad,Nspec)));
KoutCPU = deepcopy(KCPU);
Kout = deepcopy(K);
C = similar(Fin);

A_64 = CuArray{Float64}(FinCPU);
A_32 = CuArray{Float32}(FinCPU);
B_64 = CuArray{Float64}(FinCPU);
B_32 = CuArray{Float32}(FinCPU);
A = FinCPU
B = FinCPU

# Overload of batch_inv! for Dual numbers
function batch_jacobian_inv(Fout,  Fin, Kin, Kout) where {FT}
    #C = similar(Fin)
    #D = similar(Fin)
    #Kout = deepcopy(Kin)
    RadiativeTransfer.vSmartMOM.batch_inv!(Fout,Fin)
    #-Fout ⊠ Kin[1] ⊠ Fout
    
    for i=1:length(Kin)
        multi_test!(Fout,Kin[i],Kout[i])
    end
    # Compute all partial derivatives here:
    #Tuple(multi_test!(Fout,Kin[i],C,D) for i in 1:length(Kin))
    #Tuple(-Fout ⊠ Kin[i] ⊠ Fout for i in 1:length(Kin))
end

function multi_test!(A,B,D)
    #CUBLAS.gemm_strided_batched!('N', 'N', -1, A, B, 0, C)
    #CUBLAS.gemm_strided_batched!('N', 'N', 1, C, A, 0, D)
    D[:] = -A ⊠ B ⊠ A
    return nothing
end

A = CuArray{FT}(FinCPU);
B = CuArray{FT}(FinCPU);
C = similar(A);
# New bacthed mat mul
CUBLAS.cublasGemmStridedBatchedEx(CUBLAS.handle(),
                            'N',
                            'N',
                            size(A, 1 ),
                            size(B, 2 ),
                            size(A, 2 ),
                            FT(1),
                            A,
                            CUBLAS.Float64,
                            max(1,stride(A,2)),
                            size(A, 3) == 1 ? 0 : stride(A, 3),
                            B,
                            CUBLAS.Float64,
                            max(1,stride(B,2)),
                            size(B, 3) == 1 ? 0 : stride(B, 3),
                            FT(0),
                            C,
                            CUBLAS.Float64,
                            max(1,stride(C,2)),
                            stride(C, 3),
                            size(A, 3),
                            CUBLAS.CUBLAS_COMPUTE_64F,
                            CUBLAS.CUBLAS_GEMM_DEFAULT)

CUBLAS.cublasGemmBatchedEx(
                            CUBLAS.handle(),
                            'N',
                            'N',
                            size(A, 1 ),
                            size(B, 2 ),
                            size(A, 2 ),
                            FT(1),
                            CUBLAS.unsafe_strided_batch(A),
                            CUBLAS.Float64,
                            max(1,stride(A,2)),
                            CUBLAS.unsafe_strided_batch(B),
                            CUBLAS.Float64,
                            max(1,stride(B,2)),
                            FT(0),
                            CUBLAS.unsafe_strided_batch(C),
                            CUBLAS.Float64,
                            max(1,stride(C,2)),
                            size(A, 3),
                            CUBLAS.CUBLAS_COMPUTE_64F,
                            CUBLAS.CUBLAS_GEMM_DEFAULT)


map(Fin, eachrow(Kin_)) do v, p
    ForwardDiff.Dual{typeof(Fin)}(v, p...) # T is the tag
end