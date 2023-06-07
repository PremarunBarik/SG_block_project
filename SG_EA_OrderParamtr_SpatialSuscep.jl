using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools

#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2



rng = MersenneTwister(1234)

#NUMBER OF REPLICAS 
replica_num = 50

#NUMBER OF MC MC STEPS 
MC_steps = 10000
MC_burns = 10000

#TEMPERATURE VALUES
min_Temp = 0.3
max_Temp = 1.9
Temp_step = 50
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = CuArray(collect(min_Temp:Temp_interval:max_Temp))
Temp_values = reverse(Temp_values)

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 10
n_y = 10
n_z = 3

N_sg = n_x*n_y*n_z

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

#REFERENCE POSITION OF THE SPIN ELEMENTS
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = zeros(N_sg, 1)

for i in 1:N_sg
    x_pos_sg[i] = trunc((((i-1)% (n_x*n_y)))/n_x)+1             #10th position
    y_pos_sg[i] = (((i-1)%(n_x*n_y))%n_y)+1                     #1th position
    z_pos_sg[i] = trunc((i-1)/(n_x*n_y)) +1                     #100th position
end

x_pos_sg = repeat(x_pos_sg, replica_num, 1)
y_pos_sg = repeat(y_pos_sg, replica_num, 1)
z_pos_sg = repeat(z_pos_sg, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#INITIALIZATION OF FERROMAGNETIC BLOCKS
x_num = 3                                                       #number of blocks along X axis 
y_num = 3                                                       #number of blocks along Y axis
N_fm = x_num*y_num

x_dist = n_x/x_num                                              #distance between two blocks along x axis 
y_dist = n_y/y_num                                              #distance between two blocks along y axis 

#REFERENCE POSITION OF THE FERROMAGNETIC BLOCKS IN MATRIX
mx_fm = CuArray(collect(1:N_fm))

#REFERENCE POSITION OF THE BLOCKS
x_pos_fm = zeros(N_fm, 1)
y_pos_fm = zeros(N_fm, 1)
z_pos_fm = fill(N_sg + 1, N_fm) 

for i in 1:N_fm
    x_pos_fm[i] = trunc((i-1)/x_num)*(x_dist) + (x_dist/2)                  #10th position
    y_pos_fm[i] = ((i-1)%y_num)*(y_dist) + (y_dist/2)                       #1th position
end

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
NN_u = zeros(N_sg,1)
NN_d = zeros(N_sg,1)

for i in 1:N_sg                             #loop over all the spin ELEMENTS
    if x_pos_sg[i]%n_x == 0
        r_s = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-n_x)*n_x + y_pos_sg[i]
    else
        r_s = (z_pos_sg[i]-1)*(n_x*n_y) + x_pos_sg[i]*n_x + y_pos_sg[i]
    end
    r_s = convert(Int64, trunc(r_s)) 
    NN_s[i] = r_s
    #-----------------------------------------------------------#
    if x_pos_sg[i]%n_x == 1
        r_n = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]+n_x-2)*n_x + y_pos_sg[i]
    else
        r_n = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-2)*n_x + y_pos_sg[i]
    end
    r_n = convert(Int64, trunc(r_n)) 
    NN_n[i] = r_n
    #-----------------------------------------------------------#
    if y_pos_sg[i]%n_y == 0
        r_e = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-n_y+1)
    else
        r_e = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]+1
    end
    r_e = convert(Int64, trunc(r_e)) 
    NN_e[i] = r_e
    #-----------------------------------------------------------#
    if y_pos_sg[i]%n_y == 1
        r_w = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]+n_y-1)
    else
        r_w = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]-1
    end
    r_w = convert(Int64, trunc(r_w)) 
    NN_w[i] = r_w
    #-----------------------------------------------------------#
    if z_pos_sg[i]%n_z == 0
        r_u = ((z_pos_sg[i]-n_z)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
    else
        r_u = ((z_pos_sg[i])*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
    end
    r_u = convert(Int64, trunc(r_u)) 
    NN_u[i] = r_u
    #-----------------------------------------------------------#
    if z_pos_sg[i]%n_z == 1
        r_d = ((z_pos_sg[i]+n_z-2)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
    else
        r_d = ((z_pos_sg[i]-2)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
    end
    r_d = convert(Int64, trunc(r_d)) 
    NN_d[i] = r_d
end

NN_s = repeat(NN_s, replica_num, 1)
NN_n = repeat(NN_n, replica_num, 1)
NN_e = repeat(NN_e, replica_num, 1)
NN_w = repeat(NN_w, replica_num, 1)
NN_u = repeat(NN_u, replica_num, 1)
NN_d = repeat(NN_d, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#ISING NEAR NEIGHBOUR CALCULATION
NNN_s = zeros(N_sg,1)
NNN_n = zeros(N_sg,1)
NNN_e = zeros(N_sg,1)
NNN_w = zeros(N_sg,1)
NNN_u = zeros(N_sg,1)
NNN_d = zeros(N_sg,1)

for i in 1:N_sg                             #loop over all the spin ELEMENTS
        if x_pos_sg[i]%n_x == (n_x-1)
            r_s = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-n_x+1)*n_x + y_pos_sg[i]
        elseif x_pos_sg[i]%n_x == 0
            r_s = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-n_x+1)*n_x + y_pos_sg[i]
        else 
            r_s = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]+1)*n_x + y_pos_sg[i]
        end
        r_s = convert(Int64, trunc(r_s)) 
        NNN_s[i] = r_s
        #-----------------------------------------------------------#
        if x_pos_sg[i]%n_x == 1
            r_n = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]+n_x-3)*n_x + y_pos_sg[i]
        elseif x_pos_sg[i]%n_x == 2
            r_n = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]+n_x-3)*n_x + y_pos_sg[i]
        else
            r_n = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-3)*n_x + y_pos_sg[i]
        end
        r_n = convert(Int64, trunc(r_n)) 
        NNN_n[i] = r_n
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 0
            r_e = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-n_y+2)
        elseif y_pos_sg[i]%n_y == (n_y-1)
            r_e = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-n_y+2)
        else
            r_e = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]+2
        end
        NNN_e[i] = r_e
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 1
            r_w = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]+n_y-2)
        elseif y_pos_sg[i]%n_y == 2
            r_w = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]+n_y-2)
        else
            r_w = (z_pos_sg[i]-1)*(n_x*n_y) + (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-2)
        end
        r_w = convert(Int64, trunc(r_w)) 
        NNN_w[i] = r_w
        #-----------------------------------------------------------#
        if z_pos_sg[i]%n_z == 0
            r_u = ((z_pos_sg[i]-n_z+1)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
        elseif z_pos_sg[i]%n_z == (n_z-1)
            r_u = ((z_pos_sg[i]-n_z+1)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
        else
            r_u = ((z_pos_sg[i]+1)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
        end
        r_u = convert(Int64, trunc(r_u)) 
        NNN_u[i] = r_u
        #-----------------------------------------------------------#
        if z_pos_sg[i]%n_z == 1
            r_d = ((z_pos_sg[i]+n_z-3)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
        elseif z_pos_sg[i]%n_z == 2
            r_d = ((z_pos_sg[i]+n_z-3)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
        else
            r_d = ((z_pos_sg[i]-3)*(n_x*n_y)) + (x_pos_sg[i]-1)*n_x + y_pos_sg[i]
        end
        r_d = convert(Int64, trunc(r_d)) 
        NNN_d[i] = r_d 
end

NNN_s = repeat(NNN_s, replica_num, 1)
NNN_n = repeat(NNN_n, replica_num, 1)
NNN_e = repeat(NNN_e, replica_num, 1)
NNN_w = repeat(NNN_w, replica_num, 1)
NNN_u = repeat(NNN_u, replica_num, 1)
NNN_d = repeat(NNN_d, replica_num, 1)

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

#INTERACTION COEFFICIENT MATRIX
J_NNN = zeros(N_sg,N_sg,replica_num)

for i in 1:N_sg
    for j in i:N_sg
        for k in 1:replica_num
            if i==j
                continue
            else
                J_NNN[i,j,k] = J_NNN[j,i,k] = (-1)^rand(rng, Int64)                                   #for ising: 1, for spin glas: random
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

global x_pos_sg = CuArray(x_pos_sg)
global y_pos_sg = CuArray(y_pos_sg)
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
NN_u = CuArray{Int64}(NN_u)
NN_d = CuArray{Int64}(NN_d)

NNN_s = CuArray{Int64}(NNN_s)
NNN_n = CuArray{Int64}(NNN_n)
NNN_e = CuArray{Int64}(NNN_e)
NNN_w = CuArray{Int64}(NNN_w)
NNN_d = CuArray{Int64}(NNN_d)
NNN_u = CuArray{Int64}(NNN_u)

J_NN = CuArray(J_NN)
J_NNN = CuArray(J_NNN)

spin_rep_ref = CuArray{Int64}(spin_rep_ref)
rand_rep_ref = CuArray{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATION OF DUMMBELL ENERGY
function compute_dummbell_energy()
    E_0 = 1/10
    q_sg_plus = 1
    q_sg_minus = -1
    q_fm_plus = 0                                                       #zero for no magnetic field applied
    q_fm_minus = 0

    r_fm_plus_d_x = x_pos_fm' .+ (x_dir_fm' ./ 2)
    r_fm_plus_d_y = y_pos_fm'                                           #need to add - (y_dir_fm' ./ 2) - for heisenberg spin
    r_fm_plus_d_z = z_pos_fm'                                           #need to add - (z_dir_fm' ./ 2) - for heisenberg spins

    r_fm_minus_d_x = x_pos_fm' .- (x_dir_fm' ./ 2)
    r_fm_minus_d_y = y_pos_fm'                                          #need to sustract - (y_dir_fm' ./ 2) - for heisenberg spin
    r_fm_minus_d_z = z_pos_fm'                                          #need to sustract - (z_dir_fm' ./ 2) - for heisenberg spin
    
    r_sg_plus_d_x = (x_pos_sg .+ (x_dir_sg ./ 2))
    r_sg_plus_d_y = y_pos_sg                                           #need to add - (y_dir_sg' ./ 2) - for heisenberg spin
    r_sg_plus_d_z = z_pos_sg                                           #need to add - (z_dir_sg' ./ 2) - for heisenberg spin
   
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

    E_dumbbell = E_0*(term_1 .+ term_2 .+ term_3 .+ term_4)
    E_dumbbell = sum(E_dumbbell, dims=2)

    return E_dumbbell
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO rkky AND MAGNETIC BLOCKS
global energy_tot = zeros(N_sg*replica_num, 1) |> CuArray
global energy_dumbbell = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass()
    r_NN_s = (mx_sg.-1).*N_sg .+ NN_s
    r_NN_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_NN_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_NN_w = (mx_sg.-1).*N_sg .+ NN_w 
    r_NN_u = (mx_sg.-1).*N_sg .+ NN_u 
    r_NN_d = (mx_sg.-1).*N_sg .+ NN_d 

    r_NNN_s = (mx_sg.-1).*N_sg .+ NNN_s
    r_NNN_n = (mx_sg.-1).*N_sg .+ NNN_n 
    r_NNN_e = (mx_sg.-1).*N_sg .+ NNN_e 
    r_NNN_w = (mx_sg.-1).*N_sg .+ NNN_w 
    r_NNN_u = (mx_sg.-1).*N_sg .+ NNN_u 
    r_NNN_d = (mx_sg.-1).*N_sg .+ NNN_d 

    energy_x_NN = x_dir_sg.*((J_NN[r_NN_s].*x_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_NN_n].*x_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_NN_e].*x_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_NN_w].*x_dir_sg[NN_w .+ spin_rep_ref]) .+ (J_NN[r_NN_u].*x_dir_sg[NN_u .+ spin_rep_ref]) .+ (J_NN[r_NN_d].*x_dir_sg[NN_d .+ spin_rep_ref]))
    #energy_y_NN = y_dir_sg.*((J_NN[r_NN_s].*y_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_NN_n].*y_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_NN_e].*y_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_NN_w].*y_dir_sg[NN_w .+ spin_rep_ref]) .+ (J_NN[r_NN_u].*y_dir_sg[NN_u .+ spin_rep_ref]) .+ (J_NN[r_NN_d].*y_dir_sg[NN_d .+ spin_rep_ref]))
    #energy_z_NN = z_dir_sg.*((J_NN[r_NN_s].*z_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_NN_n].*z_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_NN_e].*z_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_NN_w].*z_dir_sg[NN_w .+ spin_rep_ref]) .+ (J_NN[r_NN_u].*z_dir_sg[NN_u .+ spin_rep_ref]) .+ (J_NN[r_NN_d].*z_dir_sg[NN_d .+ spin_rep_ref]))

    energy_x_NNN = x_dir_sg.*((J_NNN[r_NNN_s].*x_dir_sg[NNN_s .+ spin_rep_ref]) .+ (J_NNN[r_NNN_n].*x_dir_sg[NNN_n .+ spin_rep_ref]) .+ (J_NNN[r_NNN_e].*x_dir_sg[NNN_e .+ spin_rep_ref]) .+ (J_NNN[r_NNN_w].*x_dir_sg[NNN_w .+ spin_rep_ref]) .+ (J_NNN[r_NNN_d].*x_dir_sg[NNN_d .+ spin_rep_ref]) .+ (J_NNN[r_NNN_u].*x_dir_sg[NNN_u .+ spin_rep_ref]))
    #energy_y_NNN = y_dir_sg.*((J_NNN[r_NNN_s].*y_dir_sg[NNN_s .+ spin_rep_ref]) .+ (J_NNN[r_NNN_n].*y_dir_sg[NNN_n .+ spin_rep_ref]) .+ (J_NNN[r_NNN_e].*y_dir_sg[NNN_e .+ spin_rep_ref]) .+ (J_NNN[r_NNN_w].*y_dir_sg[NNN_w .+ spin_rep_ref]) .+ (J_NNN[r_NNN_d].*y_dir_sg[NNN_d .+ spin_rep_ref]) .+ (J_NNN[r_NNN_u].*y_dir_sg[NNN_u .+ spin_rep_ref]))
    #energy_z_NNN = z_dir_sg.*((J_NNN[r_NNN_s].*z_dir_sg[NNN_s .+ spin_rep_ref]) .+ (J_NNN[r_NNN_n].*z_dir_sg[NNN_n .+ spin_rep_ref]) .+ (J_NNN[r_NNN_e].*z_dir_sg[NNN_e .+ spin_rep_ref]) .+ (J_NNN[r_NNN_w].*z_dir_sg[NNN_w .+ spin_rep_ref]) .+ (J_NNN[r_NNN_d].*z_dir_sg[NNN_d .+ spin_rep_ref]) .+ (J_NNN[r_NNN_u].*z_dir_sg[NNN_u .+ spin_rep_ref]))

    global energy_RKKY = energy_x_NN .+ energy_x_NNN 

    global energy_dumbbell = compute_dummbell_energy()
    global energy_tot = energy_RKKY .+ energy_dumbbell

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(MC_index)
    compute_tot_energy_spin_glass()

    global r = rand_pos[:,MC_index] .+ rand_rep_ref

    global del_energy = 2*energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global trans_rate = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(MC_index, Temp_index)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(MC_index)

    @CUDA.allowscalar global Temp = Temp_values[Temp_index]
    global trans_rate = exp.(-del_energy/Temp)
    flipit = sign.(rand_num_flip[:, MC_index] .- trans_rate)

    global x_dir_sg[r] = flipit.*x_dir_sg[r]
    #global y_dir_sg[r] = flipit.*y_dir_sg[r]
    #global z_dir_sg[r] = flipit.*z_dir_sg[r]
end

#------------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------------------#

#creating a matrix with zero diagonal terms  to calculate the correlation terms
diag_zero = fill(1, (N_sg, N_sg)) |> CuArray 
diag_zero[diagind(diag_zero)] .= 0
global  diag_zero = repeat(diag_zero, replica_num, 1)

#Put inside the MC loop, after the MC function - Calculation of spatial correlation function terms
function spatial_correlation_terms(N_sg, replica_num)
    spin_mux = reshape(x_dir_sg, (N_sg, replica_num))' |>  Array
    spin_mux = repeat(spin_mux, inner = (N_sg, 1))                                  #Scalar indexing - less time consuming in CPU
    spin_mux = spin_mux |> CuArray

    global corltn_term1 += x_dir_sg .* spin_mux                                     #<sigma_i*sigma_j>
   
    global spin_sum += x_dir_sg                                                     #<sigma_i>
end

#------------------------------------------------------------------------------------------------------------------------------#

#put outside the MC loop, just at the end of MC loop, Inside the temp loop - Calculation of spatial coorelation function
function spatial_correlation_claculation(MC_steps, N_sg, replica_num)
    global corltn_term1 = (corltn_term1/MC_steps) .* diag_zero

    corltn_term2 = repeat(spin_sum/MC_steps, 1, N_sg)
    corltn_term2 = corltn_term2 .*diag_zero                                         #<sigma_i>

    corltn_term3 = reshape(spin_sum/MC_steps, (N_sg, replica_num))' |>  Array
    corltn_term3 = repeat(corltn_term3, inner = (N_sg, 1))                          #Scalar indexing - less time consuming in CPU
    corltn_term3 = corltn_term3 |> CuArray
    corltn_term3 = corltn_term3 .* diag_zero                                        #<sigma_j>

    sp_corltn = corltn_term1 .- (corltn_term2 .* corltn_term3)

    return sum(sp_corltn)/(N_sg*replica_num*(N_sg-1))
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX FOR STORING DATA
#magnetisation = zeros(length(Temp_values), 1)
#energy = zeros(length(Temp_values), 1)
EA_order_para = zeros(length(Temp_values), 1)
spatial_correlation = zeros(length(Temp_values), 1)

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

    #Initilization inside the temp loop, before MC loop - Calculation of spatial correlation function
    global corltn_term1 = zeros(N_sg*replica_num, N_sg) |> CuArray                  #<sigma_i*sigma_j>
    global spin_sum = zeros(N_sg*replica_num, 1) |> CuArray                         #<sigma_i>

 #  global mag = 0.0                                                                #storing magnetization data over monte carlo steps for a time step
 #  global en = 0.0                                                                 #storing energy data over monte carlo steps for a time step

    #-----------------------------------------------------------#

    @CUDA.allowscalar for j in 1:MC_steps

        global MC_index = j

        one_MC(MC_index, Temp_index)                                                #MONTE CARLO FUNCTION 
        spatial_correlation_terms(N_sg, replica_num)                                #CALCULATION OF TERMS TO CALCULATE SPATIAL CORRELATION

    end
    #-----------------------------------------------------------#

    spin_sqr = (spin_sum/MC_steps).^2
    EA_order_para[Temp_index] = sum(spin_sqr)/(N_sg*replica_num)

    spatial_correlation[Temp_index] = spatial_correlation_claculation(MC_steps, N_sg, replica_num)
end

#------------------------------------------------------------------------------------------------------------------------------#

#SAVING AND PLOTTING DATA
Temp_values = Array(Temp_values)

open("SG_EA_eaOrdr_SpCorrln_10_10_3.txt", "w") do io 					#creating a file to save data
   for i in 1:length(Temp_values)
      println(io,i,"\t",Temp_values[i],"\t",spatial_correlation[i],"\t",EA_order_para[i])
   end
end

#plot(Temp_values, spatial_correlation, xlabel="Temperature(T)", ylabel="Spatial Correlation (spin glass susceptibility)")
#savefig("SpatialCorrlnVsTemp_SG_EA.png")

#plot(Temp_values, EA_order_para, xlabel="Temperature(T)", ylabel="EA order parameter")
#savefig("EA_OrderparameterVsTemp_SG_EA.png")
