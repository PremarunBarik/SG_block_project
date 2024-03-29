using Plots, Random, LinearAlgebra

rng = MersenneTwister(7635)

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
x_dir_fm = Float64[ 1.0 for i in 1:N_fm]
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

#RKKY INETRACTION FUNCTION
global a = 3                        #near neighbour distance (Considering the matterial to be CuMn, with Mn density to be 10%)
global alpha =  15                  #2K_f*a the periodicity of RKKY

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
function R_ij(x_1, y_1, z_1, x_2, y_2, z_2, omega, b)
    J_0 = 4
    r_ij = sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2)
    term_1 = cos(omega*r_ij)
    term_2 = exp(-b*r_ij)
    J = J_0*term_1*term_2
    return r_ij
end

function RKKY_dummy(r_ij, omega, b)
    J_0 = 2
    term_1 = cos(omega*r_ij)
    term_2 = exp(-b*r_ij)
    J = J_0*term_1*term_2
    return J
end

#RKKY INTERACTION MATRIX
interac_j = zeros(N_sg*N_sg,1)
interatomic_diatance = zeros(N_sg,N_sg)

for i in 1:N_sg
    for j in 1:N_sg
        if i==j
          interatomic_diatance[i,j] = 0
        else
          interatomic_diatance[i,j] = R_ij(x_pos_sg[i],y_pos_sg[i],z_pos_sg[i],x_pos_sg[j],y_pos_sg[j],z_pos_sg[j], a, alpha)
        end
    end
end

interatomic_diatance = reshape(interatomic_diatance, (N_sg*N_sg,1))
interatomic_diatance = vec(interatomic_diatance)
sort!(interatomic_diatance)
interac_j .= RKKY_dummy.(interatomic_diatance, omega, b)

#PRINTING J DISTRIBUTION (using plots)
#interac_dist = reshape(interac_j, (N_sg*N_sg,1))
#interac_dist = vec(interac_dist)
#sort!(interac_dist)
N_dist = length(interac_dist)
length_dist = collect(1:N_dist)

plot(interatomic_diatance, interac_j, label = "interatomic distance distribution")
ylabel!("Length unit")
#xlims!(0,100)
#ylims!(-1,0)
