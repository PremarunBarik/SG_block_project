using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2



rng = MersenneTwister()

#NUMBER OF REPLICAS 
replica_num = 2
MC_runs = 100                                               #averaging Monte Carlo runs 

#NUMBER OF MC MC STEPS 
MC_steps = 625000
MC_burns = 625000

#TEMPERATURE VALUES
min_Temp = 0.1
max_Temp = 2.0
Temp_step = 100
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = collect(min_Temp:Temp_interval:max_Temp)
Temp_values = reverse(Temp_values)

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 4
n_y = 4

N_sg = n_x*n_y

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sg = [(-1)^rand(rng, Int64) for i in 1:N_sg*MC_runs, j in 1:replica_num]
#y_dir_sg = zeros(N_sg, 1)
#z_dir_sg = zeros(N_sg, 1)

#spin initialization -- for heisenberg spins -- need to chnage in the Array section -- need to change in the dummbell energy function
#for i in 1:N_sg
#    theta = pi/2
#    phi = 0
#    x_dir_sg[i] = sin(theta)cos(phi)
#    y_dir_sg[i] = sin(theta)sin(phi)
#    z_dir_sg[i] = cos(theta)
#end
#spin initialization -- for ising spins

#x_dir_sg = repeat(x_dir_sg, MC_runs, 1)
#y_dir_sg = repeat(y_dir_sg, replica_num, 1)
#z_dir_sg = repeat(z_dir_sg, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sg = Array(collect(1:N_sg*MC_runs))

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

NN_s = repeat(NN_s, MC_runs, 1)
NN_n = repeat(NN_n, MC_runs, 1)
NN_e = repeat(NN_e, MC_runs, 1)
NN_w = repeat(NN_w, MC_runs, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#INTERACTION COEFFICIENT MATRIX
J_NN = zeros(N_sg,N_sg,MC_runs)

for i in 1:N_sg
    for j in i:N_sg
        for k in 1:MC_runs
            if i==j
                continue
            else
                J_NN[i,j,k] = J_NN[j,i,k] = (-1)^rand(rng, Int64)                                   #for ising: 1, for spin glas: random
            end
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#REPLICA REFERENCE MATRIX OF SPIN ELEMENTS
spin_rep_ref = zeros(N_sg*MC_runs,1)

for i in eachindex(spin_rep_ref)
    spin_rep_ref[i] = trunc((i-1)/N_sg)*N_sg
end

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref = zeros(MC_runs, 1)

for i in eachindex(rand_rep_ref)
    rand_rep_ref[i] = (i-1)*N_sg
end

#------------------------------------------------------------------------------------------------------------------------------#

#CHANGING ALL THE MATRICES TO CuArray 
global x_dir_sg = CuArray(x_dir_sg)
#global y_dir_sg = CuArray(y_dir_sg)
#global z_dir_sg = CuArray(z_dir_sg)

global x_pos_sg = CuArray(x_pos_sg)
global y_pos_sg = CuArray(y_pos_sg)

NN_s = CuArray{Int64}(NN_s)
NN_n = CuArray{Int64}(NN_n)
NN_e = CuArray{Int64}(NN_e)
NN_w = CuArray{Int64}(NN_w)

J_NN = CuArray(J_NN)

spin_rep_ref = Array{Int64}(spin_rep_ref)
rand_rep_ref = Array{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO rkky AND MAGNETIC BLOCKS
global energy_tot = zeros(N_sg*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x_NN = x_dir_sg.*((J_NN[r_s].*x_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg[NN_w .+ spin_rep_ref]))
   
    global energy_tot = energy_x_NN

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = Array(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos =  Array(rand(rng, (1:N_sg), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref

    global del_energy = 2*energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global trans_rate = Array(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp_index)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    global Temp = Temp_values[Temp_index]
    global trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = Array(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)

    global x_dir_sg[r] = flipit.*x_dir_sg[r]
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
susceptibility = zeros(length(Temp_values), 1)
#energy = zeros(length(Temp_values), 1)
magnetization = zeros(length(Temp_values), 1)
#spatial_correlation = zeros(length(Temp_values), 1)

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY
@CUDA.allowscalar for i in eachindex(Temp_values)                              #TEMPERATURE LOOP 
    
    global Temp_index = i 

    #MC BURN STEPS
    for j in 1:MC_burns

        one_MC(rng, Temp_index)

    end

    #-----------------------------------------------------------#

    #Initilization inside the temp loop, before MC loop - Calculation of spatial correlation function
#   global corltn_term1 = zeros(N_sg*replica_num, N_sg) |> Array                  #<sigma_i*sigma_j>
    global spin_sum = zeros(N_sg*replica_num, 1) |> Array                         #<sigma_i>
    global spin_sqr_sum = zeros(N_sg*replica_num, 1)|> Array

#   global mag = 0.0                                                                #storing magnetization data over monte carlo steps for a time step
#   global en = 0.0                                                                 #storing energy data over monte carlo steps for a time step

    #-----------------------------------------------------------#

    for j in 1:MC_steps

        global MC_index = j

        one_MC(rng, Temp_index)                                                     #MONTE CARLO FUNCTION 
        spin_sum += x_dir_sg
        spin_sqr_sum += x_dir_sg.^2

    end
    #-----------------------------------------------------------#

    spin_sum_sqr = (spin_sum/MC_steps).^2
    spin_sqr_sum = spin_sqr_sum/MC_steps

    suscep_calculation = (spin_sqr_sum .- spin_sum_sqr)
    magnetisation[Temp_index] = sum(spin_sum)/(MC_steps*N_sg*replica_num)
    susceptibility[Temp_index] = sum(suscep_calculation)/(N_sg*replica_num*Temp_values[Temp_index])

end

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING AND PLOTTING DATA
Temp_values = Array(Temp_values)

open("2D_SG_EA_eaOrdr_25_25_reproduce.txt", "w") do io 					#creating a file to save data
   for i in 1:length(Temp_values)
      println(io,i,"\t",Temp_values[i],"\t",magnetisation[i],"\t",susceptibility[i])
   end
end

#plot(Temp_values, spatial_correlation, xlabel="Temperature(T)", ylabel="Spatial Correlation (spin glass susceptibility)")
#savefig("SpatialCorrlnVsTemp_SG_EA.png")

#plot(Temp_values, EA_order_para, xlabel="Temperature(T)", ylabel="EA order parameter")
#savefig("EA_OrderparameterVsTemp_SG_EA.png")
