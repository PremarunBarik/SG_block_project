using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools, NaNStatistics


                                    # *******  GPU Code  ****** #

#This script plots the spin configuration and J-configuration for one replica only. Which does not take a lot of time to execute in CPU.


#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2

#FERROMAGNETIC BLOCK FIELD INTENSITY -- field intensity of locally appplied field
global field_intensity = 2.0
#GLOBALLY APPLIED FIELD -- field intensity of globally applied field
global B_global = 0.0   


rng = MersenneTwister()

#NUMBER OF REPLICAS 
replica_num = 50

#NUMBER OF MC MC STEPS 
MC_steps = 100000
MC_burns = 100000

#TEMPERATURE VALUES
min_Temp = 0.1
max_Temp = 2.0
Temp_step = 50
Temp_interval = (max_Temp - min_Temp)/Temp_step
Temp_values = collect(min_Temp:Temp_interval:max_Temp)
Temp_values = reverse(Temp_values)

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

#spin initialization -- for heisenberg spins -- need to chnage in the Array section -- need to change in the dummbell energy function
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
mx_sg = CuArray(collect(1:N_sg*replica_num)) |> CuArray

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = fill(n_z, N_sg)

for i in 1:N_sg
    x_pos_sg[i] = trunc((i-1)/n_x)+1    #10th position
    y_pos_sg[i] = ((i-1)%n_y)+1     #1th position
end

#------------------------------------------------------------------------------------------------------------------------------#

#INITIALIZATION OF FERROMAGNETIC BLOCKS
x_num = 1   #number of blocks along X axis 
y_num = 1   #number of blocks along Y axis
N_fm = x_num*y_num

x_dist = n_x/x_num  #distance between two blocks along x axis 
y_dist = n_y/y_num  #distance between two blocks along y axis 

#REFERENCE POSITION OF THE FERROMAGNETIC BLOCKS IN MATRIX
mx_fm = collect(1:N_fm) |> Array

#LENGTH OF FERROMAGNETIC DIPOLE 
fm_length = 3

#REFERENCE POSITION OF THE BLOCKS
x_pos_fm = zeros(N_fm, 1)
y_pos_fm = zeros(N_fm, 1)
z_pos_fm = fill(n_z + 1, N_fm) 

for i in 1:N_fm
    x_pos_fm[i] = trunc((i-1)/x_num)*(x_dist) + (x_dist/2)  #10th position
    y_pos_fm[i] = ((i-1)%y_num)*(y_dist) + (y_dist/2)   #1th position
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
    global period_num = 100     #number of periods to consider for ewald sum
    
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
    
                    #----------------------------------------------------------------#

#scatter(positive_x_pos_ES, positive_y_pos_ES, aspect_ratio=:equal)
#scatter!(negative_x_pos_ES, negative_y_pos_ES)
    
#plotting the central block
#display(plot!([0, n_x, n_x, 0, 0],[0, 0, n_y, n_y, 0], color=:red, legend=:none))

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO MAGNETIC BLOCKS
global dipole_field = zeros(N_sg*replica_num, 1) |> Array

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
    
    global dipole_field_x = sum(B_x_tot, dims=2) 
    global dipole_field_y = sum(B_y_tot, dims=2) 
    
    global dipole_field = repeat(dipole_field_x, replica_num, 1) |> CuArray

    return dipole_field
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
                J_NN[i,j,k] = J_NN[j,i,k] = (-1)^rand(rng, Int64)                                  #for ising: 1, for spin glas: random (-1)^rand(rng, Int64)
            end
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#CHANGING ALL THE MATRICES TO CU_ARRAY 
global x_dir_sg = CuArray(x_dir_sg)
#global y_dir_sg = Array(y_dir_sg)
#global z_dir_sg = Array(z_dir_sg)

global x_pos_sg = Array(x_pos_sg .- 0.5)                                  #fixing the exact position of 
global y_pos_sg = Array(y_pos_sg .- 0.5)
global z_pos_sg = Array(z_pos_sg)

global x_pos_fm = Array(x_pos_fm)
global y_pos_fm = Array(y_pos_fm)
global z_pos_fm = Array(z_pos_fm)

NN_s = CuArray{Int64}(NN_s)
NN_n = CuArray{Int64}(NN_n)
NN_e = CuArray{Int64}(NN_e)
NN_w = CuArray{Int64}(NN_w)

J_NN = CuArray(J_NN)

spin_rep_ref = CuArray{Int64}(spin_rep_ref)
rand_rep_ref = CuArray{Int64}(rand_rep_ref)

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO MAGNETIC BLOCKS
global dipole_field = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATION OF ENERGY DUE TO FERROMAGNETIC BLOCKS AS DIPOLES
function calculate_dipole_field()
    positive_distance = sqrt.( ((x_pos_sg .- positive_x_pos_fm').^2) .+ ((y_pos_sg .- positive_y_pos_fm').^2) .+ ((z_pos_sg .- positive_z_pos_fm').^2))
    negative_distance = sqrt.( ((x_pos_sg .- negative_x_pos_fm').^2) .+ ((y_pos_sg .- negative_y_pos_fm').^2) .+ ((z_pos_sg .- negative_z_pos_fm').^2))

    q_positive = field_intensity
    q_negative = -field_intensity

    B_x_positive = q_positive *(x_pos_sg .- positive_x_pos_fm')./(positive_distance.^3)
    B_x_negative = q_negative *(x_pos_sg .- negative_x_pos_fm')./(negative_distance.^3)

    B_x_tot = B_x_positive .+ B_x_negative

    dipole_field = sum(B_x_tot, dims=2) 
    global dipole_field = repeat(dipole_field, replica_num, 1) |> Array

    return dipole_field
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE ENERGY DUE TO RKKY 
global energy_RKKY = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE TOTAL ENERGY OF THE SYSTEM
function compute_RKKY_energy_spin_glass()
    r_s = (mx_sg.-1).*N_sg .+ NN_s
    r_n = (mx_sg.-1).*N_sg .+ NN_n 
    r_e = (mx_sg.-1).*N_sg .+ NN_e 
    r_w = (mx_sg.-1).*N_sg .+ NN_w 

    energy_x = x_dir_sg.*((J_NN[r_s].*x_dir_sg[NN_s .+ spin_rep_ref]) 
                        .+(J_NN[r_n].*x_dir_sg[NN_n .+ spin_rep_ref]) 
                        .+(J_NN[r_e].*x_dir_sg[NN_e .+ spin_rep_ref]) 
                        .+(J_NN[r_w].*x_dir_sg[NN_w .+ spin_rep_ref]))
   
    global energy_RKKY = energy_x

    return energy_RKKY
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE TOTAL ENERGY
global energy_tot = zeros(N_sg*replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_tot_energy_spin_glass()
    compute_RKKY_energy_spin_glass()

    global energy_tot = 2*(energy_RKKY .+ (dipole_field .* x_dir_sg) .+ (B_global*x_dir_sg))

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> CuArray

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos =  CuArray(rand(rng, (1:N_sg), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref

    global del_energy = energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#Matrix to keep track of which flipped how many times
#global flip_count = Array(zeros(N_sg*replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)  #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = CuArray(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)
    global x_dir_sg[r] = flipit.*x_dir_sg[r]

#    flipit = (abs.(flipit .- 1))/2
#    global flip_count[r] = flip_count[r] .+ flipit
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global glauber = CuArray(zeros(replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#function to flip a spin using KMC subroutine
function one_MC_kmc(rng, N_sg, replica_num, Temp)
    compute_tot_energy_spin_glass()

    trans_rate = exp.(-energy_tot/Temp)
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
                global flip_count[loc[l,k]] = flip_count[loc[l,k]] + 1
            break
            end
        end
    end

end

#------------------------------------------------------------------------------------------------------------------------------#

#Put inside the MC loop, after the MC function - Calculation of spatial correlation function terms
function spatial_correlation_term(N_sg, replica_num)
    spin_mux = reshape(x_dir_sg, (N_sg, replica_num))' |>  Array
    spin_mux = repeat(spin_mux, inner = (N_sg, 1))  #Scalar indexing - less time consum$
    spin_mux = spin_mux |> CuArray

    global corltn_term_sum += x_dir_sg .* spin_mux  #sum of sigma_i*sigma_j

end

#------------------------------------------------------------------------------------------------------------------------------#

#CALCULATION OF DISTANCE BETWEEN TWO SPINS 
function distance_calculation()
    x_j = x_pos_sg' |>  Array
    x_j = repeat(x_j, inner = (N_sg, 1))    #Scalar indexing - less time consuming
    x_j = x_j 

    y_j = y_pos_sg' |>  Array
    y_j = repeat(y_j, inner = (N_sg, 1))    #Scalar indexing - less time consuming
    y_j = y_j 

    global distance_ij = sqrt.((x_pos_sg .- x_j).^2 .+ (y_pos_sg .- y_j).^2)

end

#------------------------------------------------------------------------------------------------------------------------------#

#calculation of final coorelation VALUES
function calculate_spatial_correlation()
    
    global corltn_term = (corltn_term_sum/MC_steps) .^ 2    #thermal average of sigma_i*sigma_j : <sigma_i*sigma_j>
    global corltn_term = reshape(corltn_term', (N_sg, N_sg, replica_num)) |> Array  #sum of <sigma_i*sigma_j> over different replicas
    global corltn_term = sum(corltn_term, dims=3)/replica_num   #replica average of sigma_i*sigma_j : [<sigma_i*sigma_j>]
    
end
#------------------------------------------------------------------------------------------------------------------------------#

#function to calculate spatiaal correlation in some particular region -- just beneth the ferromagnetic elements
function calculate_spatial_correlation_Region_wise()

    global region_radius = 2    #define the radius of the region we want to investigate spatial correlation

    global distance_ij_region = Array{Float64}(undef,0)
    global corltn_term_region = Array{Float64}(undef,0)


    for fm_element in 1:N_fm
        for sg_element in 1:N_sg

            if x_pos_sg[sg_element]<=(x_pos_fm[fm_element]+region_radius) && x_pos_sg[sg_element]>=(x_pos_fm[fm_element]-region_radius) && y_pos_sg[sg_element]<=(y_pos_fm[fm_element]+region_radius) && y_pos_sg[sg_element]>=(y_pos_fm[fm_element]-region_radius)
                
            append!(distance_ij_region, distance_ij[sg_element,:])
            append!(corltn_term_region, corltn_term[sg_element,:])

            end
        end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#Binning the generated data
function data_binning()

    global distance_ij_region = vec(distance_ij_region)
    global corltn_term_region = vec(corltn_term_region)

    global spatial_corltn = Array{Float64}(undef,0)
    global spatial_distance = Array{Float64}(undef,0)

    global bin = collect(0:0.25:7.0)
    for bin_num in 1:(length(bin)-1)
        count = 0
        spatial_corltn_sum = 0
        for distance in 1:length(distance_ij_region)
            if distance_ij_region[distance] >= bin[bin_num] && distance_ij_region[distance] < bin[bin_num+1]
                count += 1
                spatial_corltn_sum += corltn_term_region[distance]
            end
        end

        if count!=0
            push!(spatial_corltn, (spatial_corltn_sum/count))
            push!(spatial_distance, (bin[bin_num]+bin[bin_num+1])/2)
        end
    end
    
end


#------------------------------------------------------------------------------------------------------------------------------#

#MAIN BODY

global Temp = 0.3
dipole_magnetic_field() #calculation of magnetic fields due to ferromegnetic blocks

#MC BURN STEPS
@CUDA.allowscalar for j in 1:MC_burns

    one_MC(rng, Temp)

end

#-----------------------------------------------------------#

#Initilization inside the temp loop, before MC loop - Calculation of spatial correlation function
global corltn_term_sum = zeros(N_sg*replica_num, N_sg) |> CuArray   #to store <sigma_i*sigma_j> over MC steps

#-----------------------------------------------------------#

@CUDA.allowscalar for j in 1:MC_steps

    one_MC(rng, Temp)   #MONTE CARLO FUNCTION 
    spatial_correlation_term(N_sg, replica_num)
end
#-----------------------------------------------------------#

distance_calculation()
calculate_spatial_correlation()
calculate_spatial_correlation_Region_wise()
data_binning()

plot!(spatial_distance, spatial_corltn, label="Temp: $Temp, B loc:$(field_intensity), boxlength:$(region_radius*2)", legendfont=font(12), legend=:topright, linewidth=2, framestyle=:box)
title!("2D-EA spatial correlation")
xlabel!("Distance (r)", guidefont=font(14), xtickfont=font(12))
ylabel!("Spatial Correlation function (g^2)", guidefont=font(14), xtickfont=font(12))
#savefig("2D_EA_SpCorrelation_g2_T$(Temp)_2.png")


#------------------------------------------------------------------------------------------------------------------------------#
