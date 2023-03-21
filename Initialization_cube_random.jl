using Makie, GLMakie, Random, LinearAlgebra

rng = MersenneTwister(1234)

#Initialization of the SG cube
Lx = 10
Ly = 10
Lz = 3

#initialization the number of spins
N = Lx*Ly*Lz

#Initialization of spin positions
x_pos = Float64[ rand(rng, Float64)*Lx for i in 1:N]
y_pos = Float64[ rand(rng, Float64)*Ly for i in 1:N]
z_pos = Float64[ rand(rng, Float64)*Lz for i in 1:N]

#Initialization of spin directions
x_dir = Float64[ 2*rand(rng, Float64)-1 for i in 1:N]
y_dir = Float64[ 2*rand(rng, Float64)-1 for i in 1:N]
z_dir = Float64[ 2*rand(rng, Float64)-1 for i in 1:N]

#Plotting the spins
aspect=(10, 10, 5)
perspectiveness=0.5
fig = Figure(; resolution=(1200, 1200))
ax = Axis3(fig[1, 1]; aspect, perspectiveness)
p = quiver!(ax, x_pos, y_pos, z_pos, x_dir, y_dir, z_dir)
Makie.save("Initial_spins.png", fig)

