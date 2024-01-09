using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister()

#considered system size
n_x = 3
n_y = 3

#N_lattice = (Nx * Ny)

#consideres percentage of msgnetic spin
percent = 10

#number of spins
#N_sg = (N_lattice * percent)/100  |> Int64
N_sg = n_x * n_y

#spin element directions
x_dir_sg = [(-1)^rand(rng, Int64) for i in 1:N_sg]

#selecting spins in random positions
random_position = randperm(Nx*Ny)

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sg = collect(1:N_sg)
mx_sg = shuffle!(mx_sg)

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = fill(1, N_sg)

for i in 1:N_sg
    x_pos_sg[i] = trunc((mx_sg[i]-1)/n_x)+1                    #10th position
    y_pos_sg[i] = ((mx_sg[i]-1)%n_y)+1                         #1th position
end

display(scatter(x_pos_sg, y_pos_sg))

#matrix to keep track of interaction coefficients.
J_NN = zeros(N_sg, N_sg, replica_num)

#matrix to keep track of interaction coefficients
J_NN_counter = zeros(N_sg, N_sg, replica_num)

#------------------------------------------------------------------------------------------------------------------------------#

#ISING NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sg,replica_num)
NN_n = zeros(N_sg,replica_num)
NN_e = zeros(N_sg,replica_num)
NN_w = zeros(N_sg,replica_num)

for k in 1: replica_num
for i in 1:N_sg                             #loop over all the spin ELEMENTS
        if x_pos_sg[i,k]%n_x == 0
            r_e =  (x_pos_sg[i,k]-n_x)*n_x + y_pos_sg[i,k]
        else
            r_e =  x_pos_sg[i,k]*n_x + y_pos_sg[i,k]
        end

        for j in 1:N_sg
            index_1 = (j-1)*N_sg + i
            index_2 = (i-1)*N_sg + j
            if mx_sg[j]==r_e && J_NN_counter[index_1]==0 && J_NN_counter[index_2]==0
                J_NN[index_1] = J_NN[index_2] = rand(rng, Int64)
                J_NN_counter[index_1] += 1
                J_NN_counter[index_2] += 1
                NN_e[i,k] = j
            end
        end
        #-----------------------------------------------------------#
        if x_pos_sg[i]%n_x == 1
            r_w = (x_pos_sg[i]+n_x-2)*n_x + y_pos_sg[i]
        else
            r_w = (x_pos_sg[i]-2)*n_x + y_pos_sg[i]
        end
        NN_w[i] = r_w
        for j in 1:N_sg
            index_1 = (j-1)*N_sg + i
            index_2 = (i-1)*N_sg + j
            if mx_sg[j]==r_w && J_NN_counter[index_1]==0 && J_NN_counter[index_2]==0
                J_NN[index_1] = J_NN[index_2] = rand(rng, Int64)
                J_NN_counter[index_1] += 1
                J_NN_counter[index_2] += 1
            end
        end
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 0
            r_n =  (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-n_y+1)
        else
            r_n = (x_pos_sg[i]-1)*n_x + y_pos_sg[i]+1
        end
        NN_n[i] = r_n
        for j in 1:N_sg
            index_1 = (j-1)*N_sg + i
            index_2 = (i-1)*N_sg + j
            if mx_sg[j]==r_n && J_NN_counter[index_1]==0 && J_NN_counter[index_2]==0
                J_NN[index_1] = J_NN[index_2] = rand(rng, Int64)
                J_NN_counter[index_1] += 1
                J_NN_counter[index_2] += 1
            end
        end
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 1
            r_s = (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]+n_y-1)
        else
            r_s = (x_pos_sg[i]-1)*n_x + y_pos_sg[i]-1
        end
        NN_s[i] = r_s
        for j in 1:N_sg
            index_1 = (j-1)*N_sg + i
            index_2 = (i-1)*N_sg + j
            if mx_sg[j]==r_s && J_NN_counter[index_1]==0 && J_NN_counter[index_2]==0
                J_NN[index_1] = J_NN[index_2] = rand(rng, Int64)
                J_NN_counter[index_1] += 1
                J_NN_counter[index_2] += 1
            end
        end
end
end
