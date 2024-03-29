using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools, NaNStatistics


                    #*************************CPU CODE*****************************#

#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2

#time = @elapsed begin                           #calculation of execution time -- needs a 'end' at the bottom of the code 

rng = MersenneTwister()

#NUMBER OF REPLICAS 
replica_num = 50

#NUMBER OF MC MC STEPS 
MC_steps = 1000
MC_burns = 1000

#TEMPERATURE VALUES
min_Temp = 1.7
max_Temp = 2.7 
Temp_step = 50
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = collect(min_Temp:Temp_interval:max_Temp)
Temp_values = reverse(Temp_values)

Temp = 1.2                                      #for fixed temperature calculation. meaning no temp loop

#GLOBALLY APPLIED FIELD -- field intensity of globally applied field
global B_global = 0.0                         
#FERROMAGNETIC BLOCK FIELD INTENSITY -- field intensity of locally appplied field
global field_intensity = 0.0                    

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 4
n_y = 4
n_z =1
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
mx_sg = Array(collect(1:N_sg*replica_num))

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
x_num = 1                                                       #number of blocks along X axis 
y_num = 1                                                       #number of blocks along Y axis
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

#initialization of Ewald-sum
function Ewald_sum_initialize()
    global period_num = 100                         #number of periods to consider for ewald sum
    
    global simulation_box_num = (2*period_num + 1)^2
    
    x_pos_ES = n_x*collect(-period_num:1:period_num)
    x_pos_ES = repeat(x_pos_ES, inner=(2*period_num + 1))
    x_pos_ES = repeat(x_pos_ES, outer=N_fm)
    
    y_pos_ES = n_y*collect(-period_num:1:period_num)
    y_pos_ES = repeat(y_pos_ES, outer=(2*period_num +1))
    y_pos_ES = repeat(y_pos_ES, outer=N_fm)
    
                    #-----------------------------------------------------------#
    
    #initialization of image simulation boxes:
    positive_x_pos_ES = repeat(positive_x_pos_fm, outer = simulation_box_num)
    negative_x_pos_ES = repeat(negative_x_pos_fm, outer = simulation_box_num)
    
    positive_y_pos_ES = repeat(positive_y_pos_fm, outer = simulation_box_num)
    negative_y_pos_ES = repeat(negative_y_pos_fm, outer = simulation_box_num)
    
    global positive_z_pos_ES = repeat(positive_z_pos_fm, outer = simulation_box_num)
    global negative_z_pos_ES = repeat(negative_z_pos_fm, outer = simulation_box_num)
    
    global positive_x_pos_ES = positive_x_pos_ES - x_pos_ES
    global negative_x_pos_ES = negative_x_pos_ES - x_pos_ES
    
    global positive_y_pos_ES = positive_y_pos_ES - y_pos_ES
    global negative_y_pos_ES = negative_y_pos_ES - y_pos_ES
    
end
    
                #-----------------------------------------------------------#
    
#scatter(positive_x_pos_ES, positive_y_pos_ES, aspect_ratio=:equal)
#scatter!(negative_x_pos_ES, negative_y_pos_ES)
    
#plotting the central block
#display(plot!([0, n_x, n_x, 0, 0],[0, 0, n_y, n_y, 0], color=:red, legend=:none))
    
#------------------------------------------------------------------------------------------------------------------------------#
    
#CALCULATION OF MAGNETIC FIELD DUE TO DIPOLE
function dipole_magnetic_field()
    Ewald_sum_initialize()
    
    positive_distance = sqrt.( ((x_pos_sg .- positive_x_pos_ES').^2) .+ ((y_pos_sg .- positive_y_pos_ES').^2) .+ ((z_pos_sg .- positive_z_pos_ES').^2) )
    negative_distance = sqrt.( ((x_pos_sg .- negative_x_pos_ES').^2) .+ ((y_pos_sg .- negative_y_pos_ES').^2) .+ ((z_pos_sg .- negative_z_pos_ES').^2) )
    
    q_positive = field_intensity
    q_negative = -field_intensity
    
    B_x_positive = q_positive * (x_pos_sg .- positive_x_pos_ES')./positive_distance.^3
    B_y_positive = q_positive * (y_pos_sg .- positive_y_pos_ES')./positive_distance.^3
    
    B_x_negative = q_negative * (x_pos_sg .- negative_x_pos_ES')./negative_distance.^3
    B_y_negative = q_negative * (y_pos_sg .- negative_y_pos_ES')./negative_distance.^3
    
    B_x_tot = B_x_positive .+ B_x_negative
    B_y_tot = B_y_positive .+ B_y_negative
    
    global B_x_tot = sum(B_x_tot, dims=2)
    global B_y_tot = sum(B_y_tot, dims=2)
    
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
                J_NN[i,j,k] = J_NN[j,i,k] = (-1)^rand(rng, Int64)                                   #for ising: 1, for spin glas: random (-1)^rand(rng, Int64)
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

#MATRIX TO STORE ENERGY DUE TO RKKY 
global energy_RKKY = zeros(N_sg*replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_RKKY_energy_spin_glass()
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

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_RKKY_energy_spin_glass()

    global energy_tot = (energy_RKKY .+ (dipole_field .* x_dir_sg) .+ (B_global*x_dir_sg))

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global glauber = Array(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#function to flip a spin using KMC subroutine
function one_MC_kmc(rng, N_sg, replica_num, Temp)
    compute_tot_energy_spin_glass()

    trans_rate = exp.(-2*energy_tot/Temp)
    global glauber = trans_rate./(1 .+ trans_rate)
    loc = reshape(mx_sg, (N_sg,replica_num)) |> Array

    for k in 1:replica_num
        loc[:,k] = shuffle!(loc[:,k])
    end

    glauber_cpu = glauber |> Array
    trans_prob = glauber_cpu[loc] |> Array
    trans_prob_ps = cumsum(trans_prob, dims=1)

    @CUDA.allowscalar for k in 1:replica_num
        chk = rand(rng, Float64)*trans_prob_ps[N_sg,k]
        for l in 1:N_sg
            if chk <= trans_prob_ps[l,k]
                x_dir_sg[loc[l,k]] = (-1)*x_dir_sg[loc[l,k]]
            break
            end
        end
    end

end

#------------------------------------------------------------------------------------------------------------------------------#

#function to create snap shot of transition rate distribution
function plot_transition_rate()
    histogram(glauber, label="Temp:$Temp, B loc:$field_intensity")
    xlabel!("transition rate value")
    ylabel!("Population of transition rate")
    ylims!(0,300)
end

#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY
calculate_dipole_field()                                                       #CALCULATION OF MAGNETIC FIELD LINES ONE TIME AND IT WILL NOT CHANGE OVER TIME

#MC BURN STEPS
for j in 1:MC_burns

    one_MC_kmc(rng, N_sg, replica_num, Temp)

end

#MC BURN STEPS
#MC steps 
anim = @animate for snaps in 1:10
    plot_transition_rate()
    for j in 1:(MC_steps/10 |> Int64)
        one_MC_kmc(rng, N_sg, replica_num, Temp) 
    end                                                #MONTE CARLO FUNCTION 
end
#------------------------------------------------------------------------------------------------------------------------------#

gif(anim, "transition_rate_T$(Temp)_Bloc$(field_intensity)_Bglob$(B_global)_$(n_x)x$(n_y)_close.gif", fps=1)

#Mc_ref = collect(1: MC_steps)
#window_size = length(MC_ref)/100.0

#magnetization = vec(movmean(magnetization, window_size))

#plot(Mc_ref, magnetization, label="Temp: $Temp", linewidth=2, legendfont=font(12))
#xlabel!("MC steps", guidefont=font(14), xtickfont=font(14))
#ylabel!("Avg. magnetization <m>",  guidefont=font(14), ytickfont=font(14))

#------------------------------------------------------------------------------------------------------------------------------#
