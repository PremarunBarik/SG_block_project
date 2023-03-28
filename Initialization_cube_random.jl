using Makie, GLMakie, Random, LinearAlgebra


rng = MersenneTwister(1234)

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
  theta = rand(rng, Float64)2*pi
  phi = rand(rng,Float64)*pi
  x_dir_sg[i] = sin(theta)cos(phi)
  y_dir_sg[i] = sin(theta)sin(phi)
  z_dir_sg[i] = cos(theta)
end

#INITIALIZATION OF THE FM LATTICE
x_pos_fm = [1.0,1.25,1.5,1.75,2.0,5.0,5.25,5.5,5.75,6.0,9.0,9.25,9.5,9.75,10.0]
y_pos_fm = [1.0,1.25,1.5,1.75,2.0,5.0,5.25,5.5,5.75,6.0,9.0,9.25,9.5,9.75,10.0]
z_pos_fm = [4.0]
Lx_fm = length(x_pos_fm)
Ly_fm = length(y_pos_fm)
Lz_fm = length(z_pos_fm)

#FERROMAGNET SPIN POSITIONS
x_pos_fm = repeat(x_pos_fm, inner=(Lx_fm,1))
x_pos_fm = repeat(x_pos_fm, outer=(Lz_fm,1))
y_pos_fm = repeat(y_pos_fm, outer=(Ly_fm,1))
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

#INTERACTION MATRIX
Interac = zeros(N_tot,N_tot)

#Plotting the spins
aspect = (10, 10, 5)
perspectiveness = 0.5
fig = Figure(; resolution=(1200, 1200))
ax = Axis3(fig[1, 1]; aspect, perspectiveness)
quiver!(ax, x_pos_sg, y_pos_sg, z_pos_sg, x_dir_sg, y_dir_sg, z_dir_sg)
quiver!(ax, x_pos_fm, y_pos_fm, z_pos_fm, x_dir_fm, y_dir_fm, z_dir_fm, color= :red)
Makie.save("Initial_spins.png", fig)

