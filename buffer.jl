using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister()

#considered lattice size 
n_x = 10
n_y = 10

N_lattice = (n_x * n_y)

#consideres percentage of magnetic spin
percent = 20

#number of spins
N_sg = (N_lattice * percent)/100  |> Int64
#N_sg = n_x * n_y

#spin element directions in magnetic elements list
#x_dir_sg = [(-1)^rand(rng, Int64) for i in 1:N_sg]
x_dir_sg = collect(1:N_sg)

#selecting spins in random positions
random_position = randperm(N_lattice)

#REFERENCE POSITION OF THE SPIN ELEMENTS IN MATRIX
mx_sg = random_position[1:N_sg]

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = ones(N_sg, 1)

for i in 1:N_sg
    x_pos_sg[i] = trunc((mx_sg[i]-1)/n_x)+1                    #10th position
    y_pos_sg[i] = ((mx_sg[i]-1)%n_y)+1                         #1th position
end

#display(scatter(x_pos_sg, y_pos_sg))

#reference of spin directions in lattice
x_dir_lattice = zeros(N_lattice,1)
x_dir_lattice[mx_sg] += x_dir_sg

#matrix to keep track of interaction coefficients.
J_NN = zeros(N_sg, N_sg)

#matrix to keep track of interaction coefficients
J_NN_counter = zeros(N_sg, N_sg)

#------------------------------------------------------------------------------------------------------------------------------#

#ISING NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sg, 1)
NN_n = zeros(N_sg, 1)
NN_e = zeros(N_sg, 1)
NN_w = zeros(N_sg, 1)

J_NN_s = zeros(N_sg, 1)
J_NN_n = zeros(N_sg, 1)
J_NN_e = zeros(N_sg, 1)
J_NN_w = zeros(N_sg, 1)

J_NN_s_counter = zeros(N_sg, 1)
J_NN_n_counter = zeros(N_sg, 1)
J_NN_e_counter = zeros(N_sg, 1)
J_NN_w_counter = zeros(N_sg, 1)

for i in 1:N_sg                             #loop over all the spin ELEMENTS
        if x_pos_sg[i]%n_x == 0
            r_e =  (x_pos_sg[i]-n_x)*n_x + y_pos_sg[i]
        else
            r_e =  x_pos_sg[i]*n_x + y_pos_sg[i]
        end
        NN_e[i] = r_e
        for j in 1:N_sg
            if mx_sg[j]==r_e && J_NN_e_counter[i]==0 && J_NN_w_counter[j]==0
                J_NN_e[i] = J_NN_w[j] = rand(rng, Int64)
                J_NN_e_counter[i] = J_NN_w_counter[j] = 1
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
            if mx_sg[j]==r_w && J_NN_w_counter[i]==0 && J_NN_e_counter[j]==0
                J_NN_w[i] = J_NN_e[j] = rand(rng, Int64)
                J_NN_w_counter[i] = J_NN_e_counter[j] = 1
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
            if mx_sg[j]==r_n && J_NN_n_counter[i]==0 && J_NN_s_counter[j]==0
                J_NN_n[i] = J_NN_s[j] = rand(rng, Int64)
                J_NN_n_counter[i] = J_NN_s_counter[j] = 1
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
            if mx_sg[j]==r_s && J_NN_s_counter[i]==0 && J_NN_n_counter[j]==0
                J_NN_s[i] = J_NN_n[j] = rand(rng, Int64)
                J_NN_s_counter[i] = J_NN_n_counter[j] = 1
            end
        end
end
