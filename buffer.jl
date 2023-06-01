using CUDA, LinearAlgebra

mx = collect(1:100) |> CuArray
mx = reshape(mx, (10,10))

function spatial_correlation(N_sg, replica_num)

    diag_zero = fill(1, (N_sg, N_sg)) |> CuArray
    diag_zero[diagind(diag_zero)] .= 0
    diag_zero = repeat(diag_zero, (replica_num,1))

    correlation = reshape(x_dir_sg, (N_sg, replica_num))'
    correlation = repeat(correlation, inner = (N_sg, 1))
    correlation = correlation .* diag_zero

    correlation = x_dir_sg .* correlation
    correlation = sum(correlation, dims=2)
end
