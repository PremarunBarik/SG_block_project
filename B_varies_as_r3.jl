using Plots, Random, LinearAlgebra

rng = MersenneTwister(7635)

#INITIALIZATION OF SG CUBE
Lx_sg = 10
Ly_sg = 10
Lz_sg = 3

#SPIN-GLASS SPIN POSITION
x_pos_sg = collect(1:Lx_sg)
y_pos_sg = collect(1:Ly_sg)
z_pos_sg = collect(1:Lz_sg)

Lx_sg = length(x_pos_sg)
Ly_sg = length(y_pos_sg)
Lz_sg = length(z_pos_sg)

x_pos_sg = repeat(x_pos_sg, inner=(Ly_sg,1))
x_pos_sg = repeat(x_pos_sg, outer=(Lz_sg,1))
y_pos_sg = repeat(y_pos_sg, outer=(Lx_sg,1))
y_pos_sg = repeat(y_pos_sg, outer=(Lz_sg,1))
z_pos_sg = repeat(z_pos_sg, inner=(Lx_sg*Ly_sg,1))

Lx_sg = length(x_pos_sg)
Ly_sg = length(y_pos_sg)
Lz_sg = length(z_pos_sg)

x_pos_sg = vec(x_pos_sg)
y_pos_sg = vec(y_pos_sg)
z_pos_sg = vec(z_pos_sg)

N_sg = Lx_sg

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

#ONE MAGNETIC BLOCK 
b_0 = 1                                  #field streength
block_center_x = 5
block_center_y = 5
block_center_z = 4

function magnetic_field_block(x_0, y_0, z_0, x_r, y_r, z_r, b_0)
    dipole_moment = 5
    r_ij = sqrt((x_r - x_0)^2 + (y_r - y_0)^2 + (z_r - z_0)^2)
    B = dipole_moment/(r_ij^3)
  
  return B
end

#MAGNETIC FIELD DUE TO BLOCKS
b_block_x = vec(zeros(N_sg, 1))

for i in 1:N_sg
  b_block_x[i] = magnetic_field_block(block_center_x, block_center_y, block_center_z, x_pos_sg[i], y_pos_sg[i], z_pos_sg[i], b_0)
end

#PRINTING ENERGY VALUES DUE TO FERROMAGNETIC BLOCKS
scatter!(x_pos_sg, y_pos_sg, z_pos_sg, markersize=b_block_x, aspect_ratio=:equal, legend=false)
