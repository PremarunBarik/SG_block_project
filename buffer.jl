
using CUDA, Random, Plots, LinearAlgebra, BenchmarkTools


                                    # *******  CPU Code  ****** #

#This script plots the spin configuration and J-configuration for one replica only. Which does not take a lot of time to execute in CPU.


#To change the magnetic field intensity: goto 'compute_dummbell_energy' function
#To change the density and orientation of ferromagnetic blocks go to 'Initialization of ferromagnetic blocks'
#This script applies only with 3 or more than 3 layers of spin glass material
#Although debatable - 3D EA model transition temperature is between 0.9 - 1.2

#FERROMAGNETIC BLOCK FIELD INTENSITY -- field intensity of locally appplied field
global field_intensity_mx = [0.0, 0.4, 1.2, 2.0]
#GLOBALLY APPLIED FIELD -- field intensity of globally applied field
global B_global = 0.0   

rng = MersenneTwister()

#NUMBER OF REPLICAS 
global replica_num = 1

#NUMBER OF MC MC STEPS 
global MC_steps = 100000
global MC_burns = 100000

#TEMPERATURE VALUES
global Temp_mx = [0.3, 0.7, 1.2, 1.5]

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

global x_dir_sg = repeat(x_dir_sg, replica_num, 1)
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
    
    global dipole_field = repeat(dipole_field_x, replica_num, 1) |> Array

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

global NN_s = repeat(NN_s, replica_num, 1)
global NN_n = repeat(NN_n, replica_num, 1)
global NN_e = repeat(NN_e, replica_num, 1)
global NN_w = repeat(NN_w, replica_num, 1)

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

    global energy_tot = 2*(energy_RKKY .+ (dipole_field .* x_dir_sg) .+ (B_global*x_dir_sg))

    return energy_tot
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global del_energy = zeros(replica_num, 1) |> Array

#------------------------------------------------------------------------------------------------------------------------------#

#COMPUTE THE ENERGY CHANGE OF THE SYSTEM
function compute_del_energy_spin_glass(rng)
    compute_tot_energy_spin_glass()

    global rand_pos =  Array(rand(rng, (1:N_sg), (replica_num, 1)))
    global r = rand_pos .+ rand_rep_ref

    global del_energy = energy_tot[r]

    return del_energy
end

#------------------------------------------------------------------------------------------------------------------------------#

#Matrix to keep track of which flipped how many times
global flip_count = Array(zeros(N_sg*replica_num, 1))

#------------------------------------------------------------------------------------------------------------------------------#

#ONE MC STEPS
function one_MC(rng, Temp)                                           #benchmark time: 1.89ms (smallest)
    compute_del_energy_spin_glass(rng)

    trans_rate = exp.(-del_energy/Temp)
    global rand_num_flip = Array(rand(rng, Float64, (replica_num, 1)))
    flipit = sign.(rand_num_flip .- trans_rate)
    global x_dir_sg[r] = flipit.*x_dir_sg[r]

    flipit = (abs.(flipit .- 1))/2
    global flip_count[r] = flip_count[r] .+ flipit
end

#------------------------------------------------------------------------------------------------------------------------------#

#MATRIX TO STORE DELTA ENERGY
global glauber = Array(zeros(replica_num, 1))

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

    for k in 1:replica_num
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

#function to label a cluster
function calculate_cluster_label(N_sg, replica_num)

    global largest_label = 0
    global cluster_label_positive = zeros(N_sg*replica_num, 1)
    global cluster_label_negative = zeros(N_sg*replica_num, 1)
    global trial_num = N_sg
    
    cluster_NN_s = NN_s .+ spin_rep_ref
    cluster_NN_w = NN_w .+ spin_rep_ref
    cluster_NN_e = NN_e .+ spin_rep_ref
    cluster_NN_n = NN_n .+ spin_rep_ref

    #-----------------------------------------------------------#

    for trials in 1:trial_num
    for spins in 1:N_sg*replica_num
        if x_dir_sg[spins] == 1
            neighbor_label = [cluster_label_positive[cluster_NN_e[spins]], cluster_label_positive[cluster_NN_s[spins]], cluster_label_positive[cluster_NN_w[spins]], cluster_label_positive[cluster_NN_n[spins]]]
            if (sum(neighbor_label) == 0) && (cluster_label_positive[spins] == 0)
                largest_label += 1
                cluster_label_positive[spins] = largest_label
            else
                sort!(neighbor_label)
                for neighbor in 1:4
                    if neighbor_label[neighbor] != 0
                        cluster_label_positive[spins] = neighbor_label[neighbor]
                        break
                    end
                end
            end
        end
    end 
    end

    #-----------------------------------------------------------#

    global largest_label = 0
    
    for trials in 1:trial_num
    for spins in 1:N_sg*replica_num
        if x_dir_sg[spins] == -1
            neighbor_label = [cluster_label_negative[cluster_NN_e[spins]], cluster_label_negative[cluster_NN_s[spins]], cluster_label_negative[cluster_NN_w[spins]], cluster_label_negative[cluster_NN_n[spins]]]
            if (sum(neighbor_label) == 0) && (cluster_label_negative[spins] == 0)
                largest_label += 1
                cluster_label_negative[spins] = largest_label
            else
                sort!(neighbor_label)
                for neighbor in 1:4
                    if neighbor_label[neighbor] != 0
                        cluster_label_negative[spins] = neighbor_label[neighbor]
                        break
                    end
                end
            end
        end
    end
    end
end

#------------------------------------------------------------------------------------------------------------------------------#

#function to calculate average cluster size
function calculate_cluster_size()
    calculate_cluster_label(N_sg, replica_num)

    global cluster_size_positive = Array{Int64}(undef,0)
    global cluster_label_number_positive = Array{Int64}(undef,0)
    global cluster_size_negative = Array{Int64}(undef,0)
    global cluster_label_number_negative = Array{Int64}(undef,0)

    #-----------------------------------------------------------#

    for clusters in 1:N_sg*replica_num
        count = 0
        for population in 1:N_sg*replica_num
            if clusters == cluster_label_positive[population]
                count += 1
            end
        end
        if count!=0
            push!(cluster_size_positive, count)
            push!(cluster_label_number_positive, clusters)
        end
    end
    
    number_of_clusters = length(cluster_label_number_positive)
    global cluster_label_number_positive_redefined = collect(1:number_of_clusters)

    for clusters in 1:number_of_clusters
        replace!(cluster_label_positive, cluster_label_number_positive[clusters]=>cluster_label_number_positive_redefined[clusters])
    end

    #-----------------------------------------------------------#

    for clusters in 1:N_sg*replica_num
        count = 0
        for population in 1:N_sg*replica_num
            if clusters == cluster_label_negative[population]
                count += 1
            end
        end
        if count!=0
            push!(cluster_size_negative, count)
            push!(cluster_label_number_negative, clusters)
        end
    end
    
    number_of_clusters = length(cluster_label_number_negative)
    global cluster_label_number_negative_redefined = collect(1:number_of_clusters)

    for clusters in 1:number_of_clusters
        replace!(cluster_label_negative, cluster_label_number_negative[clusters]=>cluster_label_number_negative_redefined[clusters])
    end

end

#------------------------------------------------------------------------------------------------------------------------------#

#function to plot flipping count og spin glass local moments
function cluster_plot()
    calculate_cluster_size()

    global x_pos_sg_plot = x_pos_sg[1:N_sg] |> Array
    global y_pos_sg_plot = y_pos_sg[1:N_sg] |> Array
    scatter(x_pos_sg_plot, y_pos_sg_plot, markerstrokewidth=0, markersize=14, markershape=:square, alpha=cluster_label_positive/cluster_label_number_positive_redefined[length(cluster_label_number_positive_redefined)]/2, color=:red, colorbar=true, size=(600,600), aspect_ratio=:equal, framestyle=:box, label=false)
    scatter!(x_pos_sg_plot, y_pos_sg_plot, markerstrokewidth=0, markersize=14, markershape=:square, alpha=cluster_label_negative/cluster_label_number_negative_redefined[length(cluster_label_number_negative_redefined)]/2, color=:blue, colorbar=true, size=(600,600), aspect_ratio=:equal, framestyle=:box, label=false)
   
    global x_pos_sg_plot = x_pos_sg[1:N_sg] |> Array
    global y_pos_sg_plot = y_pos_sg[1:N_sg] |> Array

    global B_start_x = x_pos_sg_plot .- (dipole_field_x/2)
    global B_start_y = y_pos_sg_plot .- (dipole_field_y/2)

    quiver!(B_start_x, B_start_y, quiver=(dipole_field_x, dipole_field_y), color=:gray)
    scatter!(positive_x_pos_fm, positive_y_pos_fm, markersize=10, label="positive charge", legend=:topright, markerstrokewidth=0,color=:green)
    scatter!(negative_x_pos_fm, negative_y_pos_fm, markersize=10, label="negative charge", markerstrokewidth=0, color=:purple)
    title!("    Defined clusters at (B local: $(field_intensity), Temp: $(Temp))")

    #savefig("Cluster_config_SG$(n_x)x$(n_y)_Temp$(Temp)_Bglobal$(B_global).png")
    
end

#------------------------------------------------------------------------------------------------------------------------------#

#function to plot cluster size
function cluster_size_BarPlot()
    calculate_cluster_size()

    lo = @layout [a b]
    plot1 = bar(cluster_label_number_positive_redefined, 
                sort(cluster_size_positive), 
                label="Positive clusters, Bglob$(B_global)",
                xlabel="Cluster number",
                ylabel="Cluster size (number of spins)",
                color=:red)
    plot2 = bar(cluster_label_number_negative_redefined, 
                sort(cluster_size_negative), 
                label="Negative clusters, Bglob$(B_global)",
                xlabel="Cluster number",
                ylabel="Cluster size (number of spins)",
                color=:blue)

    plot(plot1, plot2, layout= lo)
    title!("Cluster size at Bloc$(field_intensity)")
end

#------------------------------------------------------------------------------------------------------------------------------#
#MAIN BODY


for l in eachindex(Temp_mx)
    global Temp = Temp_mx[l]    
    for i in eachindex(field_intensity_mx)
        global field_intensity = field_intensity_mx[i]

dipole_magnetic_field()

for MC_burn in 1:MC_burns
    one_MC_kmc(rng, N_sg, replica_num, Temp)
end

anim = @animate for snaps in 1:10
    
    for j in 1:(MC_steps/10 |> Int64)
        one_MC_kmc(rng, N_sg, replica_num, Temp)
    end
    cluster_size_BarPlot()
    
end
gif(anim, "Cluster_size_BarPlot_T$(Temp)_Bloc$(field_intensity).gif", fps=1)

end
end

#------------------------------------------------------------------------------------------------------------------------------#
