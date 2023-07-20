using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2

#FERROMAGNETIC BLOCK FIELD INTENSITY
global field_intensity = 0.00

rng = MersenneTwister()

#NUMBER OF REPLICAS 
replica_num = 50

#NUMBER OF MC MC STEPS 
MC_steps = 100000
MC_burns = 100000

#TEMPERATURE VALUES
Temp = 0.4
#min_Temp = 0.1
#max_Temp = 2.0
#Temp_step = 50
#Temp_interval = (max_Temp - min_Temp)/Temp_step
#Temp_values = collect(min_Temp:Temp_interval:max_Temp)
#Temp_values = reverse(Temp_values)

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 20
n_y = 20
n_z = 1

N_sg = n_x*n_y

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sg = [(-1)^rand(rng, Int64) for i in 1:N_sg]

#y_dir_sg = zeros(N_sg, 1)
#z_dir_sg = zeros(N_sg, 1)

#spin initialization -- for heisenberg spins -- need to chnage in the CuArray section -- need to change in the dummbell energy function
#for i in 1:N_sg
#    theta = pi/2
#    phi = 0
#    x_dir_sg[i] = sin(theta)cos(phi)
#    y_dir_sg[i] = sin(theta)sin(phi)
#    z_dir_sg[i] = cos(theta)
#end
#spin initialization -- for ising spins

x_dir_sg = repeat(x_dir_sg, replica_num, 1)
#y_dir_sg = repeat(y_dir_sg, replica_num, 1)
#z_dir_sg = repeat(z_dir_sg, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sg = CuArray(collect(1:N_sg*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = fill(n_z, N_sg)

for i in 1:N_sg
    x_pos_sg[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sg[i] = ((i-1)%n_y)+1                         #1th position
end

x_pos_sg = repeat(x_pos_sg, replica_num, 1)
y_pos_sg = repeat(y_pos_sg, replica_num, 1)
z_pos_sg = repeat(z_pos_sg, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#INITIALIZATION OF FERROMAGNETIC BLOCKS
x_num = 5                                                       #number of blocks along X axis 
y_num = 5                                                       #number of blocks along Y axis
N_fm = x_num*y_num

x_dist = n_x/x_num                                              #distance between two blocks along x axis 
y_dist = n_y/y_num                                              #distance between two blocks along y axis 

#REFERENCE POSITION OF THE FERROMAGNETIC BLOCKS IN MATRIX
mx_fm = collect(1:N_fm) |> CuArray

#REFERENCE POSITION OF THE BLOCKS
x_pos_fm = zeros(N_fm, 1)
y_pos_fm = zeros(N_fm, 1)
z_pos_fm = fill(n_z + 1, N_fm) 

for i in 1:N_fm
    x_pos_fm[i] = trunc((i-1)/x_num)*(x_dist) + (x_dist/2)                  #10th position
    y_pos_fm[i] = ((i-1)%y_num)*(y_dist) + (y_dist/2)                       #1th position
end

#------------------------------------------------------------------------------------------------------------------------------#

#MAGNETIC ORIENTATION OF FERROMAGNETIC BLOCKS
x_dir_fm = fill(1, N_fm)                #for applie field along x-direction -- ising spins

#for aplied magnetic field along more than one direction -- need to change in the CuArray section -- and dummbell energy function
#y_dir_fm = fill(0, N_fm)
#z_dir_fm = fill(0, N_fm)

#------------------------------------------------------------------------------------------------------------------------------#

#ISING NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sg,1)
NN_n = zeros(N_sg,1)
NN_e = zeros(N_sg,1)
NN_w = zeros(N_sg,1)

for i in 1:N_sg                             #loop over all the spin ELEMENTS
        if x_pos_sg[i]%n_x == 0
            r_s =  (x_pos_sg[i]-n_x)*n_x + y_pos_sg[i]
        else
            r_s =  x_pos_sg[i]*n_x + y_pos_sg[i]
        end
        r_s = convert(Int64, trunc(r_s)) 
        NN_s[i] = r_s
        #-----------------------------------------------------------#
        if x_pos_sg[i]%n_x == 1
            r_n = (x_pos_sg[i]+n_x-2)*n_x + y_pos_sg[i]
        else
            r_n = (x_pos_sg[i]-2)*n_x + y_pos_sg[i]
        end
        r_n = convert(Int64, trunc(r_n)) 
        NN_n[i] = r_n
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 0
            r_e =  (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-n_y+1)
        else
            r_e = (x_pos_sg[i]-1)*n_x + y_pos_sg[i]+1
        end
        NN_e[i] = r_e
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 1
            r_w = (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]+n_y-1)
        else
            r_w = (x_pos_sg[i]-1)*n_x + y_pos_sg[i]-1
        end
        r_w = convert(Int64, trunc(r_w)) 
        NN_w[i] = r_w
end

NN_s = repeat(NN_s, replica_num, 1)
NN_n = repeat(NN_n, replica_num, 1)
NN_e = repeat(NN_e, replica_num, 1)
NN_w = repeat(NN_w, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#INTERACTION COEFFICIENT MATRIX
J_NN = zeros(N_sg,N_sg,replica_num)

for i in 1:N_sg
    for j in i:N_sg
        for k in 1:replica_num
            if i==j
                continue
            else
                J_NN[i,j,k] = J_NN[j,i,k] = (-1)^rand(rng, Int64)                                #for ising: 1, for spin glas: random
            end
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#REPLICA REFERENCE MATRIX OF SPIN ELEMENTS
spin_rep_ref = zeros(N_sg*replica_num,1)

for i in eachindex(spin_rep_ref)
    spin_rep_ref[i] = trunc((i-1)/N_sg)*N_sg
end

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref = zeros(replica_num, 1)

for i in eachindex(rand_rep_ref)
    rand_rep_ref[i] = (i-1)*N_sg
end

#------------------------------------------------------------------------------------------------------------------------------#

#CHANGING ALL THE MATRICES TO CU_ARRAY 
global x_dir_sg = CuArray(x_dir_sg)
#global y_dir_sg = CuArray(y_dir_sg)
#global z_dir_sg = CuArray(z_dir_sg)

global x_pos_sg = CuArray(x_pos_sg .- 0.5)                                  #fixing the exact position of 
global y_pos_sg = CuArray(y_pos_sg .- 0.5)
global z_pos_sg = CuArray(z_pos_sg)

global x_dir_fm = CuArray(x_dir_fm)
#global y_dir_fm = CuArray(y_dir_fm)
#global z_dir_fm = CuArray(z_dir_fm)

global x_pos_fm = CuArray(x_pos_fm)
global y_pos_fm = CuArray(y_pos_fm)
global z_pos_fm = CuArray(z_pos_fm)

NN_s = CuArray{Int64}(NN_s)
NN_n = CuArray{Int64}(NN_n)
NN_e = CuArray{Int64}(NN_e)
NN_w = CuArray{Int64}(NN_w)

J_NN = CuArray(J_NN)

spin_rep_ref = CuArray{Int64}(spin_rep_ref)
rand_rep_ref = CuArray{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO MAGNETIC BLOCKS
global energy_dumbbell = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATION OF DUMMBELL ENERGY
function compute_dummbell_energy()

    q_sg_plus = 1
    q_sg_minus = -1
    q_fm_plus = field_intensity                                                       #zero for no magnetic field applied
    q_fm_minus = -field_intensity

    r_fm_plus_d_x = x_pos_fm' .+ (x_dir_fm' ./ 2)
    r_fm_plus_d_y = y_pos_fm'                                           #need to add - (y_dir_fm' ./ 2) - for heisenberg spin
    r_fm_plus_d_z = z_pos_fm'                                           #need to add - (z_dir_fm' ./ 2) - for heisenberg spins

    r_fm_minus_d_x = x_pos_fm' .- (x_dir_fm' ./ 2)
    r_fm_minus_d_y = y_pos_fm'                                          #need to sustract - (y_dir_fm' ./ 2) - for heisenberg spin
    r_fm_minus_d_z = z_pos_fm'                                          #need to sustract - (z_dir_fm' ./ 2) - for heisenberg spin
    
    r_sg_plus_d_x = (x_pos_sg .+ (x_dir_sg ./ 2))
    r_sg_plus_d_y = y_pos_sg                                            #need to add - (y_dir_sg' ./ 2) - for heisenberg spin
    r_sg_plus_d_z = z_pos_sg                                            #need to add - (z_dir_sg' ./ 2) - for heisenberg spin
   
    r_sg_minus_d_x = (x_pos_sg .- (x_dir_sg ./ 2))
    r_sg_minus_d_y = y_pos_sg                                           #need to substract - (y_dir_sg' ./ 2) - for heisenberg spin
    r_sg_minus_d_z = z_pos_sg                                           #need to substract - (y_dir_sg' ./ 2) - for heisenberg spin
   
    term_1_denom = sqrt.((r_fm_plus_d_x .- r_sg_plus_d_x).^2 .+ (r_fm_plus_d_y .- r_sg_plus_d_y).^2 .+ (r_fm_plus_d_z .- r_sg_plus_d_z).^2)
    term_1 = q_fm_plus*q_sg_plus ./ term_1_denom

    term_2_denom = sqrt.((r_fm_plus_d_x .- r_sg_minus_d_x).^2 .+ (r_fm_plus_d_y .- r_sg_minus_d_y).^2 .+ (r_fm_plus_d_z .- r_sg_minus_d_z).^2)
    term_2 = q_fm_plus*q_sg_minus./ term_2_denom

    term_3_denom = sqrt.((r_fm_minus_d_x .- r_sg_minus_d_x).^2 .+ (r_fm_minus_d_y .- r_sg_minus_d_y).^2 .+ (r_fm_minus_d_z .- r_sg_minus_d_z).^2)
    term_3 = q_fm_minus*q_sg_minus ./ term_3_denom

    term_4_denom = sqrt.((r_fm_minus_d_x .- r_sg_plus_d_x).^2 .+ (r_fm_minus_d_y .- r_sg_plus_d_y).^2 .+ (r_fm_minus_d_z .- r_sg_plus_d_z).^2)
    term_4 = q_fm_minus*q_sg_plus ./ term_4_denom

    energy_dumbbell = (term_1 .+ term_2 .+ term_3 .+ term_4)
    energy_dumbbell = sum(energy_dumbbell, dims=2)

    return energy_dumbbell
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO RKKY 
global energy_RKKY = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x = x_dir_sg.*((J_NN[r_s].*x_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg[NN_w .+ spin_rep_ref]))
   
    global energy_RKKY = energy_x

    return energy_RKKY
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sg*replica_num, 1) |> CuArray
#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()
    compute_dummbell_energy()

    global energy_tot = 2*energy_RKKY .+ energy_dumbbell

    global rand_pos =  CuArray(rand(rng, (1:N_sg), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref

    global del_energy = energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global trans_rate = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    global trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = CuArray(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)

    global x_dir_sg[r] = flipit.*x_dir_sg[r]
end

#------------------------------------------------------------------------------------------------------------------------------#

#creating a matrix with zero diagonal terms  to calculate the correlation terms
diag_zero = fill(1, (N_sg, N_sg)) |> CuArray
diag_zero[diagind(diag_zero)] .= 0
global  diag_zero = repeat(diag_zero, replica_num, 1)

#Put inside the MC loop, after the MC function - Calculation of spatial correlation function terms
function spatial_correlation_terms(N_sg, replica_num)
    spin_mux = reshape(x_dir_sg, (N_sg, replica_num))' |>  Array
    spin_mux = repeat(spin_mux, inner = (N_sg, 1))                                  #Scalar indexing - less time consum$
    spin_mux = spin_mux |> CuArray

    global corltn_term1 += x_dir_sg .* spin_mux                                     #sum of sigma_i*sigma_j

    global spin_sum += x_dir_sg                                                     # sum of sigma_i
end

#------------------------------------------------------------------------------------------------------------------------------#

#put outside the MC loop, just at the end of MC loop, Inside the temp loop - Calculation of spatial coorelation function
function spatial_correlation_claculation(MC_steps, N_sg, replica_num)
    global corltn_term1 = (corltn_term1/MC_steps)                                   #<sigma_i*sigma_j> -- average

    corltn_term2 = repeat(spin_sum/MC_steps, 1, N_sg)                               #<sigma_i> -- average

    corltn_term3 = reshape(spin_sum/MC_steps, (N_sg, replica_num))' |>  Array
    corltn_term3 = repeat(corltn_term3, inner = (N_sg, 1))                          #Scalar indexing - less time consum$
    corltn_term3 = corltn_term3 |> CuArray                                          #<sigma_j>

    global spatial_corltn = corltn_term1 .- (corltn_term2 .* corltn_term3)

    return spatial_corltn
end

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATION OF DISTANCE BETWEEN TWO SPINS 
function distance_calculation(N_sg, replica_num)
    x_j = x_pos_sg' |>  Array
    x_j = repeat(x_j, N_sg, 1)                                  #Scalar indexing - less time consuming

    y_j = y_pos_sg' |>  Array
    y_j = repeat(y_j, N_sg, 1)                                  #Scalar indexing - less time consuming

    distance_ij = sqrt.((x_pos_sg .- x_j).^2 .+ (y_pos_sg .- y_j).^2)
    distance_ij = distance_ij |> Array

    #matrix to keep track of the sorted distance
    sorted_index = zeros(N_sg*replica_num, N_sg) |> Array

    for rows in 1:N_sg*replica_num
        global sorted_index[rows,:] = sortperm(distance_ij[rows,:])
    end
    global sorted_index = round.(Int64, sorted_index)
    global sorted_index = repeat(sorted_index, N_sg, 1)

    global distance_ij = sort(distance_ij, dims=2)

    return distance_ij
end

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY

#MC BURN STEPS
@CUDA.allowscalar for j in 1:MC_burns

    one_MC(rng, Temp)

end

#-----------------------------------------------------------#

#Initilization inside the temp loop, before MC loop - Calculation of spatial correlation function
global corltn_term1 = zeros(N_sg*replica_num, N_sg) |> CuArray                  #<sigma_i*sigma_j>
global spin_sum = zeros(N_sg*replica_num, 1) |> CuArray                         #<sigma_i>

#-----------------------------------------------------------#

@CUDA.allowscalar for j in 1:MC_steps

    one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 
    spatial_correlation_terms(N_sg, replica_num)
end
#-----------------------------------------------------------#

spatial_correlation_claculation(MC_steps, N_sg, replica_num)
distance_calculation()

#------------------------------------------------------------------------------------------------------------------------------#

distance_ij = reshape(distance_ij, N_sg*N_sg*replica_num, 1) |> Array
spatial_corltn = reshape(spatial_corltn, N_sg*N_sg*replica_num, 1) |> Array

#SAVING THE GENERATED DATA
open("2D_EA_SpatialCorrelation_T0.4.txt", "w") do io 					#creating a file to save data
   for i in 1:(N_sg*N_sg*replica_num)
      println(io,i,"\t", distance_ij[i],"\t", spatial_corltn[i])
   end
end



scatter(distance_ij, spatial_corltn, label="Temp:0.4", legendfont=font(14))
xlabel!("Distance (r)", guidefont=font(14), xtickfont=font(12))
xlabel!("Spatial Coorelation function (Cs)", guidefont=font(14), xtickfont=font(12))
savefig("SpatialCorrelation_T0.4.png")

