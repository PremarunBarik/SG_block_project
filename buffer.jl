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
MC_steps = 100000
MC_burns = 50000

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
global field_intensity = 1.2                    

#------------------------------------------------------------------------------------------------------------------------------#

#NUMBER OF SPINGLASS ELEMENTS
n_x = 10
n_y = 10
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

#scatter(positive_x_pos_fm, positive_y_pos_fm)
#scatter!(negative_x_pos_fm, negative_y_pos_fm)

                #-----------------------------------------------------------#

#initialization of Ewald-sum
period_num = 3

simulation_box_num = (2*period_num + 1)^2

x_pos_ES = n_x*collect(-period_num:1:period_num)
x_pos_ES = repeat(x_pos_ES, inner=(2*period_num + 1))
x_pos_ES = repeat(x_pos_ES, outer=N_fm)

y_pos_ES = n_y*collect(-period_num:1:period_num)
y_pos_ES = repeat(y_pos_ES, outer=(2*period_num +1))
y_pos_ES = repeat(y_pos_ES, outer=N_fm)

                #-----------------------------------------------------------#

#initialization of image simulation boxes:
positive_x_pos_fm = repeat(positive_x_pos_fm, outer = simulation_box_num)
negative_x_pos_fm = repeat(negative_x_pos_fm, outer = simulation_box_num)

positive_y_pos_fm = repeat(positive_y_pos_fm, outer = simulation_box_num)
negative_y_pos_fm = repeat(negative_y_pos_fm, outer = simulation_box_num)

positive_z_pos_fm = repeat(positive_z_pos_fm, outer = simulation_box_num)
negative_z_pos_fm = repeat(negative_z_pos_fm, outer = simulation_box_num)

positive_x_pos_fm = positive_x_pos_fm - x_pos_ES
negative_x_pos_fm = negative_x_pos_fm - x_pos_ES

positive_y_pos_fm = positive_y_pos_fm - y_pos_ES
negative_y_pos_fm = negative_y_pos_fm - y_pos_ES

                #-----------------------------------------------------------#

scatter(positive_x_pos_fm, aspect_ratio=:equal, positive_y_pos_fm)
scatter!(negative_x_pos_fm, negative_y_pos_fm)

                #-----------------------------------------------------------#

#plotting the central block

plot!([0, n_x, n_x, 0, 0],[0, 0, n_y, n_y, 0], color=:red, legend=:none)

savefig("Ewald_sum.png")
