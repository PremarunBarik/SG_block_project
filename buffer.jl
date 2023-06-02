using LinearAlgebra

N_sg = 10
mx = collect(1:N_sg*N_sg) 
mx = reshape(mx, (N_sg,N_sg))

#Initilization inside the temp loop, before MC loop
global corltn_term1 = zeros(N_sg*replica_num, 1)                #<sigma_i*sigma_j>
global corltn_term2 = zeros(N_sg*replica_num, 1)                #<sigma_i>
global corltn_term3 = zeros(N_sg*replica_num, 1)                #<sigma_j>

#Put inside the MC loop, after the MC function
function spatial_correlation_terms(N_sg, replica_num)

    diag_zero = fill(1, (N_sg, N_sg)) 
    diag_zero[diagind(diag_zero)] .= 0
    diag_zero = repeat(diag_zero, (replica_num,1))

    corltn = reshape(x_dir_sg, (N_sg, replica_num))'
    corltn = repeat(corltn, inner = (N_sg, 1))
    corltn = corltn .* diag_zero

    corltn_term1 += x_dir_sg .* corltn
   
    term2 = repeat(x_dir_sg, (1, N_sg))
    corltn_term2 += term2 .* diag_zero

    corltn_term3 += corltn
end

#put outside the MC loop, just at the end of MC loop, Inside the temp loop
function spatial_correlation_claculation(MC_steps, N_sg, replica_num)
    sp_corltn = (corltn_term1/MC_steps) .- ((corltn_term2/MC_steps) .* (corltn_term3/MC_steps))
    sp_corltn = sum(sp_corltn)/((N_sg -1)*N_sg*replica_num)

    return sp_corltn
end
