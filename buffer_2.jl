using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

rng = MersenneTwister()

#considered lattice size 
n_x = 30
n_y = 30

#define the number of replicas
replica_num = 1

N_lattice = (n_x * n_y)

#consideres percentage of magnetic spin
percent = 10

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
z_pos_sg = fill(1, N_sg)

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
        NN_e[i,k] = r_e
        
        #-----------------------------------------------------------#

        if x_pos_sg[i,k]%n_x == 1
            r_w = (x_pos_sg[i,k]+n_x-2)*n_x + y_pos_sg[i,k]
        else
            r_w = (x_pos_sg[i,k]-2)*n_x + y_pos_sg[i,k]
        end
        NN_w[i,k] = r_w

        #-----------------------------------------------------------#

        if y_pos_sg[i,k]%n_y == 0
            r_n =  (x_pos_sg[i,k]-1)*n_x + (y_pos_sg[i,k]-n_y+1)
        else
            r_n = (x_pos_sg[i,k]-1)*n_x + y_pos_sg[i,k]+1
        end
        NN_n[i,k] = r_n

        #-----------------------------------------------------------#

        if y_pos_sg[i,k]%n_y == 1
            r_s = (x_pos_sg[i,k]-1)*n_x + (y_pos_sg[i,k]+n_y-1)
        else
            r_s = (x_pos_sg[i,k]-1)*n_x + y_pos_sg[i,k]-1
        end
        NN_s[i,k] = r_s

end
end

#------------------------------------------------------------------------------------------------------------------------------#

#NEXT NEAREST NEIGHBOUR CALCULATION
NNN_se = zeros(N_sg,1)
NNN_ne = zeros(N_sg,1)
NNN_sw = zeros(N_sg,1)
NNN_nw = zeros(N_sg,1)

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
    NNN_nw[i,k]= r_nw

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
    NNN_ne[i,k]= r_ne

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
    NNN_sw[i,k] = r_sw

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
    NNN_se[i,k] = r_se

end
end

#------------------------------------------------------------------------------------------------------------------------------#
