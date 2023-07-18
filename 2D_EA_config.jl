using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools


                                    # *******  CPU Code  ****** #

#This script plots the spin configuration and J-configuration for one replica only. Which does not take a lot of time to execute in CPU.


#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2

#FERROMAGNETIC BLOCK FIELD INTENSITY
global field_intensity = 1.0

rng = MersenneTwister()

#NUMBER OF REPLICAS 
replica_num = 2

#NUMBER OF MC MC STEPS 
MC_steps = 1000
MC_burns = 1000

#TEMPERATURE VALUES
Temp = 0.3

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 10
n_y = 10
n_z = 1

N_sg = n_x*n_y

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
mx_sg = collect(1:N_sg*replica_num) |> Array

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = fill(n_z, N_sg)

for i in 1:N_sg
    x_pos_sg[i] = trunc((i-1)/n_x)+1                    #10th position
    y_pos_sg[i] = ((i-1)%n_y)+1                         #1th position
end

#------------------------------------------------------------------------------------------------------------------------------#

#INITIALIZATION OF FERROMAGNETIC BLOCKS
x_num = 2                                                       #number of blocks along X axis 
y_num = 2                                                       #number of blocks along Y axis
N_fm = x_num*y_num

x_dist = n_x/x_num                                              #distance between two blocks along x axis 
y_dist = n_y/y_num                                              #distance between two blocks along y axis 

#REFERENCE POSITION OF THE FERROMAGNETIC BLOCKS IN MATRIX
mx_fm = collect(1:N_fm) |> Array

#LENGTH OF FERROMAGNETIC DIPOLE 
fm_length = 3

#REFERENCE POSITION OF THE BLOCKS
x_pos_fm = zeros(N_fm, 1)
y_pos_fm = zeros(N_fm, 1)
z_pos_fm = fill(n_z + 1, N_fm) 

for i in 1:N_fm
    x_pos_fm[i] = trunc((i-1)/x_num)*(x_dist) + (x_dist/2)                  #10th position
    y_pos_fm[i] = ((i-1)%y_num)*(y_dist) + (y_dist/2)                       #1th position
end

global positive_x_pos_fm = x_pos_fm .+ (fm_length/2)
global negative_x_pos_fm = x_pos_fm .- (fm_length/2)

global positive_y_pos_fm = y_pos_fm
global negative_y_pos_fm = y_pos_fm

global positive_z_pos_fm = z_pos_fm
global negative_z_pos_fm = z_pos_fm

#------------------------------------------------------------------------------------------------------------------------------#

#ISING NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sg,1)
NN_n = zeros(N_sg,1)
NN_e = zeros(N_sg,1)
NN_w = zeros(N_sg,1)

#ISING NEAR NEIGHBOUR CALCULATION
NN_s = zeros(N_sg,1)
NN_n = zeros(N_sg,1)
NN_e = zeros(N_sg,1)
NN_w = zeros(N_sg,1)

for i in 1:N_sg                             #loop over all the spin ELEMENTS
        if x_pos_sg[i]%n_x == 0
            r_e =  (x_pos_sg[i]-n_x)*n_x + y_pos_sg[i]
        else
            r_e =  x_pos_sg[i]*n_x + y_pos_sg[i]
        end
        NN_e[i] = r_e
        #-----------------------------------------------------------#
        if x_pos_sg[i]%n_x == 1
            r_w = (x_pos_sg[i]+n_x-2)*n_x + y_pos_sg[i]
        else
            r_w = (x_pos_sg[i]-2)*n_x + y_pos_sg[i]
        end
        NN_w[i] = r_w
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 0
            r_n =  (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]-n_y+1)
        else
            r_n = (x_pos_sg[i]-1)*n_x + y_pos_sg[i]+1
        end
        NN_n[i] = r_n
        #-----------------------------------------------------------#
        if y_pos_sg[i]%n_y == 1
            r_s = (x_pos_sg[i]-1)*n_x + (y_pos_sg[i]+n_y-1)
        else
            r_s = (x_pos_sg[i]-1)*n_x + y_pos_sg[i]-1
        end
        NN_s[i] = r_s
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
global x_dir_sg = Array(x_dir_sg)
#global y_dir_sg = CuArray(y_dir_sg)
#global z_dir_sg = CuArray(z_dir_sg)

global x_pos_sg = Array(x_pos_sg .- 0.5)                                  #fixing the exact position of 
global y_pos_sg = Array(y_pos_sg .- 0.5)
global z_pos_sg = Array(z_pos_sg)

global x_pos_fm = Array(x_pos_fm)
global y_pos_fm = Array(y_pos_fm)
global z_pos_fm = Array(z_pos_fm)

NN_s = Array{Int64}(NN_s)
NN_n = Array{Int64}(NN_n)
NN_e = Array{Int64}(NN_e)
NN_w = Array{Int64}(NN_w)

J_NN = Array(J_NN)

spin_rep_ref = Array{Int64}(spin_rep_ref)
rand_rep_ref = Array{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO MAGNETIC BLOCKS
global dipole_field = zeros(N_sg*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATION OF ENERGY DUE TO FERROMAGNETIC BLOCKS AS DIPOLES
function calculate_dipole_field()
    positive_distance = sqrt.( ((x_pos_sg .- positive_x_pos_fm').^2) .+ ((y_pos_sg .- positive_y_pos_fm').^2) .+ ((z_pos_sg .- positive_z_pos_fm').^2))
    negative_distance = sqrt.( ((x_pos_sg .- negative_x_pos_fm').^2) .+ ((y_pos_sg .- negative_y_pos_fm').^2) .+ ((z_pos_sg .- negative_z_pos_fm').^2))

    q_positive = field_intensity
    q_negative = -field_intensity

    B_x_positive = q_positive * (x_pos_sg .- positive_x_pos_fm')./(positive_distance.^3)
    B_y_positive = q_positive * (y_pos_sg .- positive_y_pos_fm')./positive_distance.^3

    B_x_negative = q_negative * (x_pos_sg .- negative_x_pos_fm')./(negative_distance.^3)
    B_y_negative = q_negative * (y_pos_sg .- negative_y_pos_fm')./negative_distance.^3

    B_x_tot = B_x_positive .+ B_x_negative
    B_y_tot = B_y_positive .+ B_y_negative

    global dipole_field_x = sum(B_x_tot, dims=2) 
    global dipole_field_y = sum(B_y_tot, dims=2) 
    
    global dipole_field = repeat(dipole_field_x, replica_num, 1) |> Array

    return dipole_field
end

#------------------------------------------------------------------------------------------------------------------------------#

#J-DISTRIBUTION AND SPIN CONFIGURATION PLOTTING
function plot_spin_config(Temp)
    #COLOR PALETTE TO PLOT THE J DISTRIBUTION
    global color_palette = [:yellow, :cyan, :green]
    global J_NN_plot = Array{Int64}(J_NN[:,:,1] .+ 2) 

    global x_pos_sg_plot = x_pos_sg[1:N_sg] |> Array
    global y_pos_sg_plot = y_pos_sg[1:N_sg] |> Array

    global J_x_pos_s = x_pos_sg_plot |> Array
    global J_y_pos_s = (y_pos_sg_plot .- 0.5) |> Array

    global J_x_pos_n = x_pos_sg_plot |> Array
    global J_y_pos_n = (y_pos_sg_plot .+ 0.5) |> Array

    global J_x_pos_e = (x_pos_sg_plot .+ 0.5) |> Array
    global J_y_pos_e = y_pos_sg_plot |> Array

    global J_x_pos_w = (x_pos_sg_plot .- 0.5) |> Array
    global J_y_pos_w = y_pos_sg_plot |> Array

    global mx_sg_plot = mx_sg[1:N_sg] |> Array

    global NN_s_plot = NN_s[1:N_sg] |> Array
    global NN_n_plot = NN_n[1:N_sg] |> Array
    global NN_e_plot = NN_e[1:N_sg] |> Array
    global NN_w_plot = NN_w[1:N_sg] |> Array

    global r_s_plot = ((mx_sg_plot.-1).*N_sg .+ NN_s_plot) |> Array
    global r_n_plot = ((mx_sg_plot.-1).*N_sg .+ NN_n_plot) |> Array
    global r_e_plot = ((mx_sg_plot.-1).*N_sg .+ NN_e_plot) |> Array
    global r_w_plot = ((mx_sg_plot.-1).*N_sg .+ NN_w_plot) |> Array

    global alpha = 0.2*ones(N_sg, 1)

    scatter(J_x_pos_s, J_y_pos_s, color=color_palette[J_NN_plot[r_s_plot]], markerstrokewidth=0, markersize=7, alpha=alpha, label="", legend=false)
    scatter!(J_x_pos_n, J_y_pos_n, color=color_palette[J_NN_plot[r_n_plot]], markerstrokewidth=0, markersize=7, alpha=alpha, label="")
    scatter!(J_x_pos_e, J_y_pos_e, color=color_palette[J_NN_plot[r_e_plot]], markerstrokewidth=0, markersize=7, alpha=alpha, label="")
    scatter!(J_x_pos_w, J_y_pos_w, color=color_palette[J_NN_plot[r_w_plot]], markerstrokewidth=0, markersize=7, alpha=alpha, label="")

    global x_dir_sg_plot = (x_dir_sg[1:N_sg] .* 0.5) |> Array
    global y_dir_sg_plot = zeros(N_sg,1) |> Array

    global x_pos_sg_start = (x_pos_sg_plot .- (x_dir_sg_plot./2)) |> Array
    global y_pos_sg_start = (y_pos_sg_plot .- (y_dir_sg_plot./2)) |> Array

    global x_dir_sg_plot = x_dir_sg_plot |> Array
    global y_dir_sg_plot = y_dir_sg_plot |> Array

    quiver!(x_pos_sg_start, y_pos_sg_start, quiver=(x_dir_sg_plot, y_dir_sg_plot), color=:blue, linewidth=1)
    scatter!(positive_x_pos_fm, positive_y_pos_fm, label="+Qm(+1)", legendfont=font(14), xtickfont=font(12), ytickfont=font(12), markerstrokewidth=0, markersize=5, color=:red)
    scatter!(negative_x_pos_fm, negative_y_pos_fm, label="-Qm(-1)", legendfont=font(14), xtickfont=font(12), ytickfont=font(12), markerstrokewidth=0, markersize=5, color=:purple)

    global B_start_x = x_pos_sg_plot .- (dipole_field_x/2)
    global B_start_y = y_pos_sg_plot .- (dipole_field_y/2)

    quiver!(B_start_x, B_start_y, quiver=(dipole_field_x, dipole_field_y), color=:gray)
    
    title!("Spin configuration at T:$Temp")

    #savefig("config_T$Temp.png")
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO RKKY 
global energy_RKKY = zeros(N_sg*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_tot_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x = x_dir_sg.*((J_NN[r_s].*x_dir_sg[NN_s .+ spin_rep_ref]) .+ (J_NN[r_n].*x_dir_sg[NN_n .+ spin_rep_ref]) .+ (J_NN[r_e].*x_dir_sg[NN_e .+ spin_rep_ref]) .+ (J_NN[r_w].*x_dir_sg[NN_w .+ spin_rep_ref]))
   
    global energy_RKKY = energy_x

    return energy_RKKY
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sg*replica_num, 1) |> Array
#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global energy_tot = 2*(energy_RKKY .+ (dipole_field .* x_dir_sg))

    global rand_pos =  Array(rand(rng, (1:N_sg), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref

    global del_energy = energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global trans_rate = Array(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    global trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = Array(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)

    global x_dir_sg[r] = flipit.*x_dir_sg[r]
end

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY
calculate_dipole_field()                                                       #CALCULATION OF MAGNETIC FIELD LINES ONE TIME AND IT WILL NOT CHANGE OVER TIME

anim = @animate for j in 1:10
    plot_spin_config(Temp)
    for s in 1:10000
        one_MC(rng, Temp)                                                     #MONTE CARLO FUNCTION 
#       spin_sum += x_dir_sg
#       spin_sqr_sum += x_dir_sg.^2
    end
end

gif(anim, "Config_T$(Temp)_B$(field_intensity)_$(x_num)x$(y_num).gif", fps=1)

