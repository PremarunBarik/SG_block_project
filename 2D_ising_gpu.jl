using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister(1234)

#NUMBER OF REPLICAS 
replica_num = 3

#NUMBER OF MC MC STEPS 
MC_steps = 1000
MC_burns = 1000

#TEMPERATURE VALUES
min_Temp = 0.1
max_Temp = 3.6
Temp_step = 20
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = CuArray(collect(min_Temp:Temp_interval:max_Temp))

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 10
n_y = 10

N_sg = n_x*n_y

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sg = CuArray(zeros(N_sg,1))
y_dir_sg = CuArray(zeros(N_sg,1))
z_dir_sg = CuArray(zeros(N_sg,1))

for i in 1:N_sg
    theta = pi/2
    phi = 0
    x_dir_sg[i] = sin(theta)cos(phi)
    y_dir_sg[i] = sin(theta)sin(phi)
    z_dir_sg[i] = cos(theta)
end

x_dir_sg = repeat(x_dir_sg, 1, replica_num)
y_dir_sg = repeat(y_dir_sg, 1, replica_num)
z_dir_sg = repeat(z_dir_sg, 1, replica_num)


#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS
x_pos_sg = CuArray(zeros(N_sg, 1))
y_pos_sg = CuArray(zeros(N_sg, 1))

for i in 1:N_sg
    x_pos_sg[i] = trunc((i-1)/n_x)+1             #10th position
    y_pos_sg[i] = ((i-1)%n_y)+1                     #1th position
end

x_pos_sg = repeat(x_pos_sg, 1, replica_num)
y_pos_sg = repeat(y_pos_sg, 1, replica_num)

#------------------------------------------------------------------------------------------------------------------------------#

#ISING INTERACTION MATRIX WITH NN INTERACTION
J_NN = CuArray(zeros(N_sg,N_sg, replica_num))

for i in 1:N_sg                             #loop over all the spin ELEMENTS
        for k in 1:replica_num
            if x_pos_sg[i,k]%n_x == 0
                r_s =  (x_pos_sg[i,k]-n_x)*n_x + y_pos_sg[i,k]
            else
                r_s =  x_pos_sg[i,k]*n_x + y_pos_sg[i,k]
            end
            r_s = convert(Int64, trunc(r_s)) 
            J_NN[i,r_s,k] = J_NN[r_s,i,k] = 1
            #-----------------------------------------------------------#
            if x_pos_sg[i,k]%n_x == 1
                r_n = (x_pos_sg[i,k]+n_x-2)*n_x + y_pos_sg[i,k]
            else
                r_n = (x_pos_sg[i,k]-2)*n_x + y_pos_sg[i,k]
            end
            r_n = convert(Int64, trunc(r_n)) 
            J_NN[i,r_n,k] = J_NN[r_n,i,k] = 1
            #-----------------------------------------------------------#
            if y_pos_sg[i,k]%n_y == 0
                r_e =  (x_pos_sg[i,k]-1)*n_x + (y_pos_sg[i,k]-n_y+1)
            else
                r_e = (x_pos_sg[i,k]-1)*n_x + y_pos_sg[i,k]+1
            end
            r_e = convert(Int64, trunc(r_e)) 
            J_NN[i,r_e,k] = J_NN[r_e,i,k] = 1
            #-----------------------------------------------------------#
            if y_pos_sg[i,k]%n_y == 1
                r_w = (x_pos_sg[i,k]-1)*n_x + (y_pos_sg[i,k]+n_y-1)
            else
                r_w = (x_pos_sg[i,k]-1)*n_x + y_pos_sg[i,k]-1
            end
            r_w = convert(Int64, trunc(r_w)) 
            J_NN[i,r_w,k] = J_NN[r_w,i,k] = 1
        end
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
energy_tot_NN = CuArray(zeros(N_sg, replica_num))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass(replica_num)
    for i in 1:replica_num
        replica_index = i
        
        energy_x_NN = x_dir_sg[:,replica_index]'.*J_NN[:,:,replica_index]
        energy_x_NN = sum(energy_x_NN, dims=2)
        energy_x_NN = energy_x_NN.*x_dir_sg[:,replica_index]

        energy_y_NN = y_dir_sg[:,replica_index]'.*J_NN[:,:,replica_index]
        energy_y_NN = sum(energy_y_NN, dims=2)
        energy_y_NN = energy_y_NN.*y_dir_sg[:,replica_index]

        energy_z_NN = z_dir_sg[:,replica_index]'.*J_NN[:,:,replica_index]
        energy_z_NN = sum(energy_z_NN, dims=2)
        energy_z_NN = energy_z_NN.*z_dir_sg[:,replica_index]
        
        energy_tot_NN[:,replica_index] = energy_x_NN.+energy_y_NN.+energy_z_NN 
    end

    return energy_tot_NN
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX WITH RANDOM INTEGER
rand_pos = CuArray(rand(rng, (1:N_sg), (MC_steps,replica_num)))
#MATRIX WITH RANDOM FLOAT NUMBER TO FLIP SPIN
rand_num_flip = CuArray(rand(rng, Float64, (MC_steps,replica_num)))

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
del_energy = CuArray(zeros(1, replica_num))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(MC_index, replica_num)
    compute_tot_energy_spin_glass(replica_num)

    for i in 1:replica_num
        replica_index = i 
        del_energy[1,replica_index] = 2*energy_tot_NN[rand_pos[MC_index,replica_index],replica_index]
    end
    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
trans_rate = CuArray(zeros(1, replica_num))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(MC_index, Temp_index, replica_num)
    compute_del_energy_spin_glass(MC_index, replica_num)

    for i in 1:replica_num
        replica_index = i 
        trans_rate[1,replica_index] = exp(-del_energy[1,replica_index]/Temp_values[Temp_index])
        flipit = sign(rand_num_flip[MC_index,replica_index]-trans_rate[1,replica_index])

        x_dir_sg[rand_pos[MC_index, replica_index]] = flipit*x_dir_sg[rand_pos[MC_index, replica_index]]
        y_dir_sg[rand_pos[MC_index, replica_index]] = flipit*y_dir_sg[rand_pos[MC_index, replica_index]]
        z_dir_sg[rand_pos[MC_index, replica_index]] = flipit*z_dir_sg[rand_pos[MC_index, replica_index]]
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STOTING DATA
magnetisation = CuArray(zeros(length(Temp_values), 1))
energy = CuArray(zeros(length(Temp_values), 1))

#------------------------------------------------------------------------------------------------------------------------------#

#MC BURN STEPS
for i in 1:MC_burns
  global MC_index = i
  global Temp_index = 1
  one_MC(MC_index, Temp_index, replica_num)
end

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY
for i in 1:length(Temp_values)                              #TEMPERATURE LOOP 
    
    global Temp_index = i 
    #-----------------------------------------------------------#
    #MATRIX WITH RANDOM INTEGER
    rand_pos = CuArray(rand(rng, (1:N_sg), (MC_steps,replica_num)))
    #MATRIX WITH RANDOM FLOAT NUMBER TO FLIP SPIN
    rand_num_flip = CuArray(rand(rng, Float64, (MC_steps,replica_num)))
    #-----------------------------------------------------------#
    global mag = 0.0
    global en = 0.0
    #-----------------------------------------------------------#
    for j in 1:MC_steps
        global MC_index = j
        one_MC(MC_index, Temp_index, replica_num)

        mag = mag + sum(x_dir_sg)
    end
    #-----------------------------------------------------------#
    magnetisation[Temp_index] = mag/(replica_num*MC_steps*N_sg)
end

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING AND PLOTTING DATA
magnetisation = Array(magnetisation)
Temp_values = Array(Temp_values)

open("2D_Ising_gpu_magVsTemp_10_10.txt", "w") do io 					#creating a file to save data
  for i in 1:length(Temp_values)
      println(io,i,"\t",Temp_values[i],"\t",magnetisation[i])
  end
end
println("--COMPLETE--")

display(plot(Temp_values, magnetisation))
savefig("2D_Ising_gpu_magVsTemp.png")
