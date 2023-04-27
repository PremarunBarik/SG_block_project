using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister(1234)

#NUMBER OF REPLICAS 
replica_num = 2

#NUMBER OF MC MC STEPS 
MC_steps = 1000
MC_burns = 1000

#TEMPERATURE VALUES
min_Temp = 1.5
max_Temp = 3.0
Temp_step = 20
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = CuArray(collect(min_Temp:Temp_interval:max_Temp))

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 3
n_y = 3

N_sg = n_x*n_y

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sg = zeros(N_sg,1)
y_dir_sg = zeros(N_sg,1)
z_dir_sg = zeros(N_sg,1)

for i in 1:N_sg
    theta = pi/2
    phi = 0
    x_dir_sg[i] = sin(theta)cos(phi)
    y_dir_sg[i] = sin(theta)sin(phi)
    z_dir_sg[i] = cos(theta)
end

x_dir_sg = repeat(x_dir_sg, replica_num, 1)
y_dir_sg = repeat(y_dir_sg, replica_num, 1)
z_dir_sg = repeat(z_dir_sg, replica_num, 1)


#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sg = CuArray(collect(1:N_sg*replica_num))

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)

for i in 1:N_sg
    x_pos_sg[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sg[i] = ((i-1)%n_y)+1                         #1th position
end

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
                J_NN[i,j,k] = J_NN[j,i,k] = i+j                                   #for ising: 1, for spin glas: random
            end
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#REPLICA REFERENCE MATRIX OF SPIN ELEMENTS
spin_rep_ref = zeros(N_sg*replica_num,1)

for i in 1:length(spin_rep_ref)
    spin_rep_ref[i] = trunc((i-1)/N_sg)*N_sg
end

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref = zeros(replica_num, 1)

for i in 1:length(rand_rep_ref)
    rand_rep_ref[i] = (i-1)*N_sg
end

#------------------------------------------------------------------------------------------------------------------------------#

#CHANGING ALL THE MATRICES TO CU_ARRAY 
x_dir_sg = CuArray(x_dir_sg)
y_dir_sg = CuArray(y_dir_sg)
z_dir_sg = CuArray(z_dir_sg)

NN_s = CuArray{Int64}(NN_s)
NN_n = CuArray{Int64}(NN_n)
NN_e = CuArray{Int64}(NN_e)
NN_w = CuArray{Int64}(NN_w)

J_NN = CuArray(J_NN)

spin_rep_ref = CuArray{Int64}(spin_rep_ref)
rand_rep_ref = CuArray{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
energy_tot_NN = CuArray(zeros(N_sg*replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

function compute_tot_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x_NN = x_dir_sg.*((J_NN[r_s].*x_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg[NN_w .+ spin_rep_ref]))
    energy_y_NN = y_dir_sg.*((J_NN[r_s].*y_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*y_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*y_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*y_dir_sg[NN_w .+ spin_rep_ref]))
    energy_z_NN = z_dir_sg.*((J_NN[r_s].*z_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*z_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*z_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*z_dir_sg[NN_w .+ spin_rep_ref]))

    energy_tot_NN = energy_x_NN .+ energy_y_NN .+ energy_z_NN

    return energy_tot_NN
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
del_energy = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(MC_index, )
    compute_tot_energy_spin_glass()

    r = rand_pos[:,MC_index] .+ rand_rep_ref

    del_energy = energy_tot_NN[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
trans_rate = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(MC_index, Temp_index)
    compute_del_energy_spin_glass(MC_index)

    trans_rate = exp.(-del_energy./Temp_values[Temp_index])
    flipit = sign.(rand_num_flip[:, MC_index] .- trans_rate)

        x_dir_sg[rand_pos[MC_index, replica_index],replica_index] = flipit*x_dir_sg[rand_pos[MC_index, replica_index],replica_index]
        y_dir_sg[rand_pos[MC_index, replica_index],replica_index] = flipit*y_dir_sg[rand_pos[MC_index, replica_index],replica_index]
        z_dir_sg[rand_pos[MC_index, replica_index],replica_index] = flipit*z_dir_sg[rand_pos[MC_index, replica_index],replica_index]
    end
end

#------------------------------------------------------------------------------------------------------------------------------#
