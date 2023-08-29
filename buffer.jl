using Plots, Random, LinearAlgebra, BenchmarkTools

#FERROMAGNETIC BLOCK FIELD INTENSITY
global field_intensity = 1.2

#NUMBER OF REPLICAS 
replica_num = 1

rng = MersenneTwister()

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
mx_sg = collect(1:N_sg*replica_num)

#REFERENCE POSITION OF THE SPIN ELEMENTS IN GEOMETRY
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = fill(n_z, N_sg)

for i in 1:N_sg
    x_pos_sg[i] = trunc((i-1)/n_x)+1 - 0.5                   #10th position
    y_pos_sg[i] = ((i-1)%n_y)+1 - 0.5                        #1th position
end

x_pos_sg = repeat(x_pos_sg, replica_num, 1)
y_pos_sg = repeat(y_pos_sg, replica_num, 1)
z_pos_sg = repeat(z_pos_sg, replica_num, 1)

#------------------------------------------------------------------------------------------------------------------------------#

#INITIALIZATION OF FERROMAGNETIC BLOCKS
x_num = 1                                                       #number of blocks along X axis 
y_num = 1                                                       #number of blocks along Y axis
N_fm = x_num*y_num

x_dist = n_x/x_num                                              #distance between two blocks along x axis 
y_dist = n_y/y_num                                              #distance between two blocks along y axis 

#REFERENCE POSITION OF THE FERROMAGNETIC BLOCKS IN MATRIX
mx_fm = collect(1:N_fm) 

#REFERENCE POSITION OF THE BLOCKS
x_pos_fm = zeros(N_fm, 1)
y_pos_fm = zeros(N_fm, 1)
z_pos_fm = fill(n_z + 1, N_fm) 

#LENGTH OF BLOCK 
fm_length = 3.0

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

dipole_magnetic_field()

B_start_x = x_pos_sg .- (B_x_tot/2)
B_start_y = y_pos_sg .- (B_y_tot/2)

#quiver(B_start_x, B_start_y, quiver=(B_x_tot, B_y_tot))
#scatter!(positive_x_pos_fm, positive_y_pos_fm, label="Positive magnetic charge", legendfont=font(14), xtickfont=font(12), ytickfont=font(12))
#display(scatter!(negative_x_pos_fm, negative_y_pos_fm, label="Negative magnetic charge"))
#title!("Field lines with period:$(period_num)")
#savefig("Field_distribution_B$(field_intensity)_period$(period_num).png")

#histogram(B_x_tot, label="Field_intensity:$field_intensity", legendfont=font(14))
#xlabel!("Magnetic field sterngth", guidefont=font(12), xtickfont=font(12))
#ylabel!("Population of magnetic field strength", guidefont=font(12), ytickfont=font(12))
#title!("Field intensity histogram with period:$(period_num)")
#savefig("Field_histogram_B$(field_intensity)_period$(period_num).png")

#------------------------------------------------------------------------------------------------------------------------------#
#magnetic field without Ewald sum
#global period_num = 0
#dipole_magnetic_field()
#B_0 = B_x_tot

# Comparing the change in the magnetic field due to increasing Ewald sum period
#for i in 0:10
#    global period_num = i 

#    dipole_magnetic_field()
#    println(sum(B_x_tot))
#end

scatter!(collect(1:N_sg), B_x_tot, markersize=2, markerstrokewidth=0, label="Period: $(period_num)")
ylims!(-0.1, 0.1)
xlabel!("Spin element position reference", guidefont=font(12), xtickfont=font(12))
ylabel!("Magnetic field strength", guidefont=font(12), ytickfont=font(12))
title!("Change in field intensity with increasing periodicity")
savefig("FieldIntensityWithEwaldSum_close.png")
