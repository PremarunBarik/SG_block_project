using Plots, Random, LinearAlgebra

rng = MersenneTwister(7895)

#INITIALIZATION OF SG CUBE
Lx_sg = 10
Ly_sg = 10
Lz_sg = 3

#NUMBER OF SPIN-GLASS SPINS
N_sg = Lx_sg*Ly_sg*Lz_sg

#SPIN-GLASS SPIN POSITION
x_pos_sg = Float64[ rand(rng, Float64)*Lx_sg for i in 1:N_sg]
y_pos_sg = Float64[ rand(rng, Float64)*Ly_sg for i in 1:N_sg]
z_pos_sg = Float64[ rand(rng, Float64)*Lz_sg for i in 1:N_sg]

#SPIN-GLASS SPIN VECTORS
x_dir_sg = zeros(N_sg,1)
y_dir_sg = zeros(N_sg,1)
z_dir_sg = zeros(N_sg,1)

for i in 1:N_sg
  theta = rand(rng, Float64)*2*pi
  phi = rand(rng,Float64)*pi
  x_dir_sg[i] = sin(theta)cos(phi)
  y_dir_sg[i] = sin(theta)sin(phi)
  z_dir_sg[i] = cos(theta)
end

x_dir_sg = vec(x_dir_sg)
y_dir_sg = vec(y_dir_sg)
z_dir_sg = vec(z_dir_sg)

#INITIALIZATION OF THE FM LATTICE
x_pos_fm = [1.5,5.5,9.5,]
y_pos_fm = [1.0,1.25,1.5,1.75,2.0,5.0,5.25,5.5,5.75,6.0,9.0,9.25,9.5,9.75,10.0]
z_pos_fm = [4.0]
Lx_fm = length(x_pos_fm)
Ly_fm = length(y_pos_fm)
Lz_fm = length(z_pos_fm)

#FERROMAGNET SPIN POSITIONS
x_pos_fm = repeat(x_pos_fm, inner=(Ly_fm,1))
x_pos_fm = repeat(x_pos_fm, outer=(Lz_fm,1))
y_pos_fm = repeat(y_pos_fm, outer=(Lx_fm,1))
y_pos_fm = repeat(y_pos_fm, outer=(Lz_fm,1))
z_pos_fm = repeat(z_pos_fm, inner=(Lx_fm*Ly_fm,1))

x_pos_fm = vec(x_pos_fm)
y_pos_fm = vec(y_pos_fm)
z_pos_fm = vec(z_pos_fm)
  
#NUMBER OF FERROMAGNETIC SPINS
N_fm = Lx_fm*Ly_fm*Lz_fm

#FERROMAGNETIC SPIN VECTORS
x_dir_fm = Float64[ 5.0 for i in 1:N_fm]
y_dir_fm = Float64[ 0.0 for i in 1:N_fm]
z_dir_fm = Float64[ 0.0 for i in 1:N_fm]

#NUMBER OF TOTAL SPINS
N_tot = N_sg+N_fm

#PRINTING INITIAL CONFIGURATION (using makie)
#println(Interac)
#Plotting the spins
#aspect = (10, 10, 5)
#perspectiveness = 0.5
#fig = Figure(; resolution=(1200, 1200))
#ax = Axis3(fig[1, 1]; aspect, perspectiveness)
#quiver!(ax, x_pos_sg, y_pos_sg, z_pos_sg, x_dir_sg, y_dir_sg, z_dir_sg)
#quiver!(ax, x_pos_fm, y_pos_fm, z_pos_fm, x_dir_fm, y_dir_fm, z_dir_fm, color= :red)
#Makie.save("Initial_spins.png", fig)

#RKKY INETRACTION J VALUE

global a = 3                               #near neighbour distance (Considering the matterial to be CuMn, with Mn density to be 10%)
global alpha =  15                         #2K_f*a the periodicity of RKKY

function RKKY_J(x_1, y_1, z_1, x_2, y_2, z_2, a, alpha)
  J_0 = 1/1000
  r_ij = sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2)/a            #distance between spins in terms of near neighbour distance
  term_1 = cos(alpha*r_ij)/r_ij^3
  term_2 = sin(alpha*r_ij)/(alpha*(r_ij)^4)
  J = J_0*(term_1-term_2)
  return J
end

#DUMMY RKKY INTERACTION FUNCTION
global omega = pi                          #pediodicity
global b = 0.5                             #damping_coefficient
function RKKY_dummy(x_1, y_1, z_1, x_2, y_2, z_2, omega, b)
    J_0 = 2
    r_ij = sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2)
    term_1 = cos(omega*r_ij)
    term_2 = exp(-b*r_ij)
    J = J_0*term_1*term_2
    return J
end

#RKKY INTERACTION MATRIX
interac_j = zeros(N_sg,N_sg)

for i in 1:N_sg
    for j in 1:N_sg
        if i==j
            interac_j[i,j]=0
        else
            interac_j[i,j] = RKKY_dummy(x_pos_sg[i], y_pos_sg[i], z_pos_sg[i], x_pos_sg[j], y_pos_sg[j], z_pos_sg[j], omega, b)
        end
    end
end

#PRINTING J DISTRIBUTION (using plots)
#interac_dist = reshape(interac_j, (N_sg*N_sg,1))
#interac_dist = vec(interac_dist)
#sort!(interac_dist, rev = true)
#N_dist = length(interac_dist)
#length_dist = collect(1:N_dist)

#plot(length_dist, interac_dist, label = "J distribution")
#ylabel!("Interaction coefficient (J)")
#xlims!(0,100)
#ylims!(-0.5,1)

#INTERACTION DUE TO FERROMAGNETIC BLOCKS
function dumbell_energy(x_fm, y_fm, z_fm, x_sg, y_sg, z_sg, s_sg_x, s_sg_y, s_sg_z, s_fm_x, s_fm_y, s_fm_z)
  E_0 = 0.001
  q_sg_plus = 1
  q_sg_minus = -1
  q_fm_plus = 5
  q_fm_minus = -5

  r_fm_d_plus = sqrt((x_fm+(s_fm_x/2))^2 + (y_fm+(s_fm_y)/2)^2 + (z_fm+(s_fm_z/2))^2)
  r_fm_d_minus = sqrt((x_fm-(s_fm_x/2))^2 + (y_fm-(s_fm_y)/2)^2 + (z_fm-(s_fm_z/2))^2)
  r_sg_d_plus = sqrt((x_sg+(s_sg_x/2))^2 + (y_sg+(s_sg_y)/2)^2 + (z_sg+(s_sg_z/2))^2)
  r_sg_d_minus = sqrt((x_sg-(s_sg_x/2))^2 + (y_sg-(s_sg_y)/2)^2 + (z_fm-(s_sg_z/2))^2)

  term_1 = q_fm_plus*q_sg_plus/sqrt((r_fm_d_plus-r_sg_d_plus)^2)
  term_2 = q_fm_plus*q_sg_minus/sqrt((r_fm_d_plus-r_sg_d_minus)^2)
  term_3 = q_fm_minus*q_sg_minus/sqrt((r_fm_d_minus-r_sg_d_minus)^2)
  term_4 = q_fm_minus*q_sg_plus/sqrt((r_fm_d_minus-r_sg_d_plus)^2)

  E = E_0*(term_1 + term_2 + term_3 + term_4)
end

function magnetic_field_block(x_0, y_0, z_0, x_r, y_r, z_r)              #COSIDERING MAGNETIC FIELD VARIES AS 1/r^3
  dipole_moment = 5
  r_ij = sqrt((x_r - x_0)^2 + (y_r - y_0)^2 + (z_r - z_0)^2)
  B = dipole_moment/(r_ij^3)

return B
end

#MAGNETIC FIELD DUE TO BLOCKS
b_block_x = vec(zeros(N_sg, 1))                                         #CONSIDERING MAGNETIC FIELD VARIES AS 1/r^3
block_energy_sg = zeros(N_sg, 1)

for i in 1:N_sg                       #for loop for all the spin glass materials
  for j in 1:N_fm                     #for loop for all the ferromagnetic dumbells
    #block_energy_sg[i] += dumbell_energy(x_pos_fm[j], y_pos_fm[j], z_pos_fm[j], x_pos_sg[i], y_pos_sg[i], z_pos_sg[i], x_dir_sg[i], y_dir_sg[i], z_dir_sg[i], x_dir_fm[j], y_dir_fm[j], z_dir_fm[j])          #ENERGY TERM COSIDERING DUMBELL MODEL
    b_block_x[i] += magnetic_field_block(x_pos_fm[j], y_pos_fm[j], z_pos_fm[j], x_pos_sg[i], y_pos_sg[i], z_pos_sg[i])                                                                                   #MAGNETIC FIELD CONSIDERING IT VARIES AS 1/r^3
  end
end

block_energy_sg .= x_dir_sg.*b_block_x          #MAGNETIC ENERGY CALCULATION COSIDERING MAGNETIC FIELD VARIES AS 1/r^3
block_energy_sg = vec(block_energy_sg)

#PRINTING HISTOGRAM OF ENERGY VALUES
#histogram(block_energy_sg)

#PRINTING THE ENERGY VALUES CORRESPONDING TO SPINGLASS POSITION
scatter(x_pos_sg, y_pos_sg, z_pos_sg, markersize=block_energy_sg, legend=false)
