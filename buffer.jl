using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2



rng = MersenneTwister(1234)

#NUMBER OF REPLICAS 
replica_num = 10

#NUMBER OF MC MC STEPS 
MC_steps = 10
MC_burns = 10000

#TEMPERATURE VALUES
min_Temp = 0.1
max_Temp = 1.5
Temp_step = 50
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = CuArray(collect(min_Temp:Temp_interval:max_Temp))

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 5
n_y = 5
n_z = 3

N_sg = n_x*n_y*n_z

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sg = collect(1:N_sg) |> CuArray
y_dir_sg = zeros(N_sg, 1)
z_dir_sg = zeros(N_sg, 1)

#for i in 1:N_sg
#    theta = pi/2
#    phi = 0
#    x_dir_sg[i] = sin(theta)cos(phi)
#    y_dir_sg[i] = sin(theta)sin(phi)
#    z_dir_sg[i] = cos(theta)
#end

x_dir_sg = repeat(x_dir_sg, replica_num, 1)
y_dir_sg = repeat(y_dir_sg, replica_num, 1)
z_dir_sg = repeat(z_dir_sg, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#creating a matrix with zero diagonal terms  to calculate the correlation terms
diag_zero = fill(1, (N_sg, N_sg)) |> CuArray 
diag_zero[diagind(diag_zero)] .= 0
global  diag_zero = repeat(diag_zero, replica_num, 1)

#Put inside the MC loop, after the MC function - Calculation of spatial correlation function terms
function spatial_correlation_terms(N_sg, replica_num)
    spin_mux = reshape(x_dir_sg, (N_sg, replica_num))' |>  Array
    spin_mux = repeat(spin_mux, inner = (N_sg, 1))                                  #Scalar indexing - less time consuming in CPU
    spin_mux = spin_mux |> CuArray

    global corltn_term1 += x_dir_sg .* spin_mux                                     #<sigma_i*sigma_j>
   
    global spin_sum += x_dir_sg                                                     #<sigma_i>
end

#------------------------------------------------------------------------------------------------------------------------------#

#put outside the MC loop, just at the end of MC loop, Inside the temp loop - Calculation of spatial coorelation function
function spatial_correlation_claculation(MC_steps, N_sg, replica_num)
    corltn_term1 = (corltn_term1/MC_steps) .* diag_zero

    corltn_term2 = repeat(spin_sum/MC_steps, 1, N_sg)
    corltn_term2 = corltn_term2 .*diag_zero                                         #<sigma_i>

    corltn_term3 = reshape(spin_mux/MC_steps, (N_sg, replica_num))' |>  Array
    corltn_term3 = repeat(corltn_term3, inner = (N_sg, 1))                          #Scalar indexing - less time consuming in CPU
    corltn_term3 = corltn_term3 |> CuArray
    corltn_term3 = corltn_term3 .* diag_zero                                        #<sigma_j>

    sp_corltn = corltn_term1 .- (corltn_term2 .* corltn_term3)

    return sp_corltn
end

#------------------------------------------------------------------------------------------------------------------------------#

#Initilization inside the temp loop, before MC loop - Calculation of spatial correlation function
global corltn_term1 = zeros(N_sg*replica_num, N_sg) |> CuArray                #<sigma_i*sigma_j>
global spin_sum = zeros(N_sg*replica_num, 1) |> CuArray                #<sigma_i>

for i in 1:MC_steps
    spatial_correlation_terms(N_sg, replica_num)
end
