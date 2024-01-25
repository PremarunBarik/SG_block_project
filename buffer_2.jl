using Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister()

#considered lattice size 
n_x = 40
n_y = 40

#define the number of replicas
replica_num = 2

N_lattice = (n_x * n_y)

#consideres percentage of magnetic spin
percent = 10

#number of spins
N_sg = (N_lattice * percent)/100  |> Int64
#N_sg = n_x * n_y

#spin element directions in magnetic elements list
x_dir_sg = [(-1)^rand(rng, Int64) for i in 1:N_sg]
x_dir_sg = repeat(x_dir_sg, 1, replica_num)
#x_dir_sg = collect(1:N_sg)

#matrix to keep the spin-reference number 
mx_sg = zeros(N_sg, replica_num)

#selecting spin postion among replicas
for k in 1:replica_num
    #selecting spins in random positionsin one single replica
    random_position = randperm(N_lattice)
    mx_sg[:,k] = random_position[1:N_sg]
end

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, replica_num)
y_pos_sg = zeros(N_sg, replica_num)
z_pos_sg = fill(1, (N_sg, replica_num))

for k in 1:replica_num
    for i in 1:N_sg
        x_pos_sg[i,k] = trunc((mx_sg[i,k]-1)/n_x)+1                    #10th position
        y_pos_sg[i,k] = ((mx_sg[i,k]-1)%n_y)+1                         #1th position
    end
end

#display(scatter(x_pos_sg, y_pos_sg))

#------------------------------------------------------------------------------------------------------------------------------#

#NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sg,replica_num)
NN_n = zeros(N_sg,replica_num)
NN_e = zeros(N_sg,replica_num)
NN_w = zeros(N_sg,replica_num)

for k in 1:replica_num
for i in 1:N_sg                             #loop over all the spin ELEMENTS
        if x_pos_sg[i,k]%n_x == 0
            r_e =  (x_pos_sg[i,k]-n_x)*n_x + y_pos_sg[i,k]
        else
            r_e =  x_pos_sg[i,k]*n_x + y_pos_sg[i,k]
        end
        NN_e[i,k] = r_e + (k-1)*N_lattice
        
        #-----------------------------------------------------------#

        if x_pos_sg[i,k]%n_x == 1
            r_w = (x_pos_sg[i,k]+n_x-2)*n_x + y_pos_sg[i,k]
        else
            r_w = (x_pos_sg[i,k]-2)*n_x + y_pos_sg[i,k]
        end
        NN_w[i,k] = r_w + (k-1)*N_lattice

        #-----------------------------------------------------------#

        if y_pos_sg[i,k]%n_y == 0
            r_n =  (x_pos_sg[i,k]-1)*n_x + (y_pos_sg[i,k]-n_y+1)
        else
            r_n = (x_pos_sg[i,k]-1)*n_x + y_pos_sg[i,k]+1
        end
        NN_n[i,k] = r_n + (k-1)*N_lattice

        #-----------------------------------------------------------#

        if y_pos_sg[i,k]%n_y == 1
            r_s = (x_pos_sg[i,k]-1)*n_x + (y_pos_sg[i,k]+n_y-1)
        else
            r_s = (x_pos_sg[i,k]-1)*n_x + y_pos_sg[i,k]-1
        end
        NN_s[i,k] = r_s + (k-1)*N_lattice

end
end

#------------------------------------------------------------------------------------------------------------------------------#

#NEXT NEAREST NEIGHBOUR CALCULATION
NNN_se = zeros(N_sg,replica_num)
NNN_ne = zeros(N_sg,replica_num)
NNN_sw = zeros(N_sg,replica_num)
NNN_nw = zeros(N_sg,replica_num)

for k in 1:replica_num
for i in 1:N_sg
    if y_pos_sg[i,k]%n_y == 0
        if x_pos_sg[i,k]%n_x == 1
            r_nw = (x_pos_sg[i,k]+n_x-2)*n_x + (y_pos_sg[i,k]-n_y+1)
        else
            r_nw = (x_pos_sg[i,k]-2)*n_x + (y_pos_sg[i,k]-n_y+1)
        end
    else
        if x_pos_sg[i,k]%n_x == 1
            r_nw = (x_pos_sg[i,k]+n_x-2)*n_x + (y_pos_sg[i,k]+1)
        else
            r_nw = (x_pos_sg[i,k]-2)*n_x + (y_pos_sg[i,k]+1)
        end
    end
    NNN_nw[i,k]= r_nw + (k-1)*N_lattice

    #-----------------------------------------------------------#

    if y_pos_sg[i,k]%n_y == 0
        if x_pos_sg[i]%n_x == 0
            r_ne =  (x_pos_sg[i,k]-n_x)*n_x + (y_pos_sg[i,k]-n_y+1)
        else
            r_ne =  x_pos_sg[i,k]*n_x + (y_pos_sg[i,k]-n_y+1)
        end
    else
        if x_pos_sg[i,k]%n_x == 0
            r_ne =  (x_pos_sg[i,k]-n_x)*n_x + (y_pos_sg[i,k]+1)
        else
            r_ne =  x_pos_sg[i,k]*n_x + (y_pos_sg[i,k]+1)
        end
    end
    NNN_ne[i,k]= r_ne + (k-1)*N_lattice

    #-----------------------------------------------------------#

    if y_pos_sg[i,k]%n_y == 1
        if x_pos_sg[i,k]%n_x == 1
            r_sw = (x_pos_sg[i,k]+n_x-2)*n_x + (y_pos_sg[i,k]+n_y-1)
        else
            r_sw = (x_pos_sg[i,k]-2)*n_x + (y_pos_sg[i,k]+n_y-1)
        end
    else
        if x_pos_sg[i,k]%n_x == 1
            r_sw = (x_pos_sg[i,k]+n_x-2)*n_x + (y_pos_sg[i,k]-1)
        else
            r_sw = (x_pos_sg[i,k]-2)*n_x + (y_pos_sg[i,k]-1)
        end
    end
    NNN_sw[i,k] = r_sw + (k-1)*N_lattice

    #-----------------------------------------------------------#

    if y_pos_sg[i,k]%n_y == 1
        if x_pos_sg[i,k]%n_x == 0
            r_se = (x_pos_sg[i,k]-n_x)*n_x + (y_pos_sg[i,k]+n_y-1)
        else
            r_se = x_pos_sg[i,k]*n_x + (y_pos_sg[i,k]+n_y-1)
        end
    else
        if x_pos_sg[i,k]%n_x == 0
            r_se = (x_pos_sg[i,k]-n_x)*n_x + (y_pos_sg[i,k]-1)
        else
            r_se = x_pos_sg[i,k]*n_x + (y_pos_sg[i,k]-1)
        end
    end
    NNN_se[i,k] = r_se + (k-1)*N_lattice

end
end

#------------------------------------------------------------------------------------------------------------------------------#
#reference matrix for updating the sg spin directions to lattice spin direction among replicas
lattice_replica_ref = collect(0:(replica_num-1))' .* ones(N_sg, 1) .* N_lattice
lattice_replica_ref = reshape(Array{Int64}(lattice_replica_ref), N_sg*replica_num, 1)

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref_sg = zeros(replica_num, 1)

for i in eachindex(rand_rep_ref_sg)
    rand_rep_ref_sg[i] = (i-1)*N_sg
end

#REPLICA REFERENCE MATRIX OF RANDOMLY SELECTED SPIN IN MC_STEP
rand_rep_ref_lattice = zeros(replica_num, 1)

for i in eachindex(rand_rep_ref_lattice)
    rand_rep_ref_lattice[i] = (i-1)*N_lattice
end

#------------------------------------------------------------------------------------------------------------------------------#
#In this section we change all the 2D matrices to 1D matrices.

x_dir_sg = reshape(x_dir_sg, N_sg*replica_num, 1)

mx_sg = reshape(Array{Int64}(mx_sg), N_sg*replica_num, 1)

x_pos_sg = reshape(Array{Int64}(x_pos_sg), N_sg*replica_num, 1)
y_pos_sg = reshape(Array{Int64}(y_pos_sg), N_sg*replica_num, 1)
z_pos_sg = reshape(Array{Int64}(z_pos_sg), N_sg*replica_num, 1)

NN_e = reshape(Array{Int64}(NN_e), N_sg*replica_num, 1)
NN_n = reshape(Array{Int64}(NN_n), N_sg*replica_num, 1)
NN_s = reshape(Array{Int64}(NN_s), N_sg*replica_num, 1)
NN_w = reshape(Array{Int64}(NN_w), N_sg*replica_num, 1)

NNN_ne = reshape(Array{Int64}(NNN_ne), N_sg*replica_num, 1)
NNN_nw = reshape(Array{Int64}(NNN_nw), N_sg*replica_num, 1)
NNN_se = reshape(Array{Int64}(NNN_se), N_sg*replica_num, 1)
NNN_sw = reshape(Array{Int64}(NNN_sw), N_sg*replica_num, 1)

rand_rep_ref_sg = Array{Int64}(rand_rep_ref_sg)
rand_rep_ref_lattice = Array{Int64}(rand_rep_ref_lattice)

#------------------------------------------------------------------------------------------------------------------------------#

#reference of spin directions in lattice
x_dir_lattice = zeros(N_lattice*replica_num, 1)
x_dir_lattice[mx_sg .+ lattice_replica_ref] .= x_dir_sg

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO RKKY 
global energy_RKKY = zeros(N_sg*replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#
#calculate RKKY energy of spin_glass

function compute_RKKY_energy_spin_glass()
    
    energy_x = x_dir_sg.*((J_NN .* (x_dir_lattice[NN_n] .+ x_dir_lattice[NN_s] .+ x_dir_lattice[NN_e] .+ x_dir_lattice[NN_w]))
                        .+ (J_NNN .* (x_dir_lattice[NNN_ne] .+ x_dir_lattice[NNN_nw] .+ x_dir_lattice[NNN_se] .+ x_dir_lattice[NNN_sw])))

    global energy_RKKY = energy_x

end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sg*replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_RKKY_energy_spin_glass()

    global energy_tot = 2*(energy_RKKY .+ (B_global*x_dir_sg))

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos = rand(rng, (1:N_sg), (replica_num, 1))
    global r = rand_pos .+ rand_rep_ref_sg

    global del_energy = energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = CuArray(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)
    global x_dir_sg[r] = flipit.*x_dir_sg[r]
    global x_dir_lattice[mx[r] .+ rand_rep_ref_lattice] = flipit .* x_dir_lattice[mx[r] .+ rand_rep_ref_lattice]

#    flipit = (abs.(flipit .- 1))/2
#    global flip_count[r] = flip_count[r] .+ flipit
end

#------------------------------------------------------------------------------------------------------------------------------#
