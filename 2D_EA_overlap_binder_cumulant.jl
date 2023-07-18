using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2



rng = MersenneTwister()

#NUMBER OF REPLICAS 
replica_num = 400

#NUMBER OF MC MC STEPS 
MC_steps = 100000
MC_burns = 100000

#TEMPERATURE VALUES
min_Temp = 0.1
max_Temp = 2.0
Temp_step = 100
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = collect(min_Temp:Temp_interval:max_Temp)
Temp_values = reverse(Temp_values)

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 10
n_y = 10

N_sg = n_x*n_y

#SPIN ELEMENT DIRECTION IN REPLICAS
x_dir_sg_1 = [-1 for i in 1:N_sg]
x_dir_sg_2 = [-1 for i in 1:N_sg]

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

x_dir_sg_1 = repeat(x_dir_sg_1, replica_num, 1)
x_dir_sg_2 = repeat(x_dir_sg_2, replica_num, 1)
#y_dir_sg = repeat(y_dir_sg, replica_num, 1)
#z_dir_sg = repeat(z_dir_sg, replica_num, 1)

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
                J_NN[i,j,k] = J_NN[j,i,k] = (-1)^rand(rng, Int64)                                   #for ising: 1, for spin glas: random
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
global x_dir_sg_1 = CuArray(x_dir_sg_1)
global x_dir_sg_2 = CuArray(x_dir_sg_2)
#global y_dir_sg = CuArray(y_dir_sg)
#global z_dir_sg = CuArray(z_dir_sg)

global x_pos_sg = CuArray(x_pos_sg)
global y_pos_sg = CuArray(y_pos_sg)

NN_s = CuArray{Int64}(NN_s)
NN_n = CuArray{Int64}(NN_n)
NN_e = CuArray{Int64}(NN_e)
NN_w = CuArray{Int64}(NN_w)

J_NN = CuArray(J_NN)

spin_rep_ref = CuArray{Int64}(spin_rep_ref)
rand_rep_ref = CuArray{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO rkky AND MAGNETIC BLOCKS
global energy_tot_1 = zeros(N_sg*replica_num, 1) |> CuArray
global energy_tot_2 = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x_1 = x_dir_sg_1.*((J_NN[r_s].*x_dir_sg_1[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg_1[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg_1[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg_1[NN_w .+ spin_rep_ref]))
    energy_x_2 = x_dir_sg_2.*((J_NN[r_s].*x_dir_sg_2[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg_2[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg_2[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg_2[NN_w .+ spin_rep_ref]))
   
    global energy_tot_1 = energy_x_1
    global energy_tot_2 = energy_x_2

    return energy_tot_1, energy_tot_2
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy_1 = CuArray(zeros(replica_num, 1))
global del_energy_2 = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos_1 =  CuArray(rand(rng, (1:N_sg), (replica_num, 1)))
    global rand_pos_2 =  CuArray(rand(rng, (1:N_sg), (replica_num, 1)))
    global r_1 = rand_pos_1 .+ rand_rep_ref
    global r_2 = rand_pos_2 .+ rand_rep_ref

    global del_energy_1 = 2*energy_tot_1[r_1]
    global del_energy_2 = 2*energy_tot_2[r_2]

    return del_energy_1, del_energy_2
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global trans_rate_1 = CuArray(zeros(replica_num, 1))
global trans_rate_2 = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    global trans_rate_1 = exp.(-del_energy_1/Temp)
    global trans_rate_2 = exp.(-del_energy_2/Temp)
    global rand_num_flip_1 = CuArray(rand(rng, Float64, (replica_num, 1)))
    global rand_num_flip_2 = CuArray(rand(rng, Float64, (replica_num, 1)))
    flipit_1 = sign.(rand_num_flip_1 .- trans_rate_1)
    flipit_2 = sign.(rand_num_flip_2 .- trans_rate_2)

    global x_dir_sg_1[r_1] = flipit_1.*x_dir_sg_1[r_1]
    global x_dir_sg_2[r_2] = flipit_2.*x_dir_sg_2[r_2]
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
overlap_binder = zeros(length(Temp_values), 1)

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY
@CUDA.allowscalar for i in eachindex(Temp_values)                              #TEMPERATURE LOOP 
    
    global Temp_index = i
    global Temp = Temp_values[Temp_index] 

    #MC BURN STEPS
    @CUDA.allowscalar for j in 1:MC_burns

        one_MC(rng, Temp)

    end

    #-----------------------------------------------------------#

    #Initilization inside the temp loop, before MC loop - Calculation of spatial correlation function
    global overlap_av_pow2 = zeros(1, replica_num)
    global overlap_av_pow4 = zeros(1, replica_num)
    global overlap_binder_replica = zeros(1, replica_num)

    #-----------------------------------------------------------#

    @CUDA.allowscalar for j in 1:MC_steps

        one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 
        overlap = x_dir_sg_1 .* x_dir_sg_2
        overlap_av_per_replica = Array(sum(reshape(overlap, (N_sg,replica_num)), dims=1)/N_sg)

        overlap_av_pow2 .= overlap_av_pow2 .+ overlap_av_per_replica .^2
        overlap_av_pow4 .= overlap_av_pow4 .+ overlap_av_per_replica .^ 4
        
    end
    #-----------------------------------------------------------#

    overlap_av_pow2 .= (overlap_av_pow2/MC_steps) .^ 2
    overlap_av_pow4 .= (overlap_av_pow4/MC_steps)

    overlap_binder_replica .= 0.5 .* (3 .- (overlap_av_pow4 ./ overlap_av_pow2))
    overlap_binder[Temp_index] = sum(overlap_binder_replica)/replica_num

end

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING THE GENERATED DATA
open("2D_EA_OverlapBinder_10x10_100K.txt", "w") do io 					#creating a file to save data
   for i in 1:length(Temp_values)
      println(io,i,"\t", Temp_values[i],"\t",overlap_binder[i])
   end
end

#plot(Temp_values, spatial_correlation, xlabel="Temperature(T)", ylabel="Spatial Correlation (spin glass susceptibility)")
#savefig("SpatialCorrlnVsTemp_SG_EA.png")

#plot(Temp_values, EA_order_para, xlabel="Temperature(T)", ylabel="EA order parameter")
#savefig("EA_OrderparameterVsTemp_SG_EA.png")
