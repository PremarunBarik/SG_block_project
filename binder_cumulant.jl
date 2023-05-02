using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister(1234)

#NUMBER OF REPLICAS 
replica_num = 5

#NUMBER OF MC MC STEPS 
MC_steps = 10000
MC_burns = 10000

#TEMPERATURE VALUES
min_Temp = 1.5
max_Temp = 3.0
Temp_step = 30
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = CuArray(collect(min_Temp:Temp_interval:max_Temp))

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 8
n_y = 8

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
                J_NN[i,j,k] = J_NN[j,i,k] = 1                                   #for ising: 1, for spin glas: random
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
global y_dir_sg = CuArray(y_dir_sg)
global z_dir_sg = CuArray(z_dir_sg)

NN_s = CuArray{Int64}(NN_s)
NN_n = CuArray{Int64}(NN_n)
NN_e = CuArray{Int64}(NN_e)
NN_w = CuArray{Int64}(NN_w)

J_NN = CuArray(J_NN)

spin_rep_ref = CuArray{Int64}(spin_rep_ref)
rand_rep_ref = CuArray{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
global energy_tot_NN = CuArray(zeros(N_sg*replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x_NN = x_dir_sg.*((J_NN[r_s].*x_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg[NN_w .+ spin_rep_ref]))
    energy_y_NN = y_dir_sg.*((J_NN[r_s].*y_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*y_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*y_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*y_dir_sg[NN_w .+ spin_rep_ref]))
    energy_z_NN = z_dir_sg.*((J_NN[r_s].*z_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*z_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*z_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*z_dir_sg[NN_w .+ spin_rep_ref]))

    global energy_tot_NN = energy_x_NN .+ energy_y_NN .+ energy_z_NN

    return energy_tot_NN
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(MC_index)
    compute_tot_energy_spin_glass()

    global r = rand_pos[:,MC_index] .+ rand_rep_ref

    global del_energy = 2*energy_tot_NN[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global trans_rate = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(MC_index, Temp_index)
    compute_del_energy_spin_glass(MC_index)

    @CUDA.allowscalar global Temp = Temp_values[Temp_index]
    global trans_rate = exp.(-del_energy/Temp)
    flipit = sign.(rand_num_flip[:, MC_index] .- trans_rate)

    global x_dir_sg[r] = flipit.*x_dir_sg[r]
    global y_dir_sg[r] = flipit.*y_dir_sg[r]
    global z_dir_sg[r] = flipit.*z_dir_sg[r]
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STOTING DATA
magnetisation = zeros(length(Temp_values), 1)
energy = zeros(length(Temp_values), 1)
binder_cumulant = zeros(length(Temp_values), 1)

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY
@CUDA.allowscalar for i in eachindex(Temp_values)                              #TEMPERATURE LOOP 
    
    global Temp_index = i 

    #-----------------------------------------------------------#
    #MATRIX WITH RANDOM INTEGER
    global rand_pos = CuArray(rand(rng, (1:N_sg), (replica_num, MC_burns)))
    #MATRIX WITH RANDOM FLOAT NUMBER TO FLIP SPIN
    global rand_num_flip = CuArray(rand(rng, Float64, (replica_num, MC_burns)))
    #-----------------------------------------------------------#

    #MC BURN STEPS
    @CUDA.allowscalar for j in 1:MC_burns

        global MC_index = j

        one_MC(MC_index, Temp_index)
    end

    #-----------------------------------------------------------#
    #MATRIX WITH RANDOM INTEGER
    global rand_pos = CuArray(rand(rng, (1:N_sg), (replica_num, MC_steps)))
    #MATRIX WITH RANDOM FLOAT NUMBER TO FLIP SPIN
    global rand_num_flip = CuArray(rand(rng, Float64, (replica_num, MC_steps)))
    #-----------------------------------------------------------#

    global mag = 0.0
    global en = 0.0
    global spin_av_pow2 = CuArray(zeros(1, replica_num))
    global spin_av_pow4 = CuArray(zeros(1, replica_num))

    #-----------------------------------------------------------#
    @CUDA.allowscalar for j in 1:MC_steps

        global MC_index = j

        one_MC(MC_index, Temp_index)

        mag = mag + sum(x_dir_sg)
        spin_av_per_replica = CuArray(sum(reshape(x_dir_sg, (N_sg,replica_num)), dims=1)/N_sg)

        spin_av_pow2 .= spin_av_pow2 .+ spin_av_per_replica .^2
        spin_av_pow4 .= spin_av_pow4 .+ spin_av_per_replica .^ 4

    end
    spin_av_pow2 .= (spin_av_pow2/MC_steps) .^ 2
    spin_av_pow4 .= (spin_av_pow2/MC_steps)

    binder_cumulant[Temp_index] = 1 - (sum(spin_av_pow4)/(3*sum(spin_av_pow2)))

    magnetisation[Temp_index] = mag/(replica_num*MC_steps*N_sg)
end

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING AND PLOTTING DATA
Temp_values = Array(Temp_values)

#open("2D_ising_gpu_magVsTemp_apprch2_30_30.txt", "w") do io 					#creating a file to save data
#   for i in 1:length(Temp_values)
#      println(io,i,"\t",Temp_values[i],"\t",magnetisation[i])
#   end
#end

display(plot(Temp_values, binder_cumulant))
