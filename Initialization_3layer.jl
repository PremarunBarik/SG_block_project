using Plots
using Random

rng = MersenneTwister(1234)

#Initialization of the SG cube
Lx = 10
Ly = 10
Lz = 10

#initialization the number of spins
N = 300
mx = Int64[ (-1)^rand(rng, Int64) for i in 1:N]
x_pos = Float64[ rand(rng, Float64)*Lx for i in 1:Lx]
y_pos = Float64[ rand(rng, Float64)*Ly for i in 1:Ly]
z_pos = Float64[ rand(rng, Float64)*Lz for i in 1:Lz]


#Creating file to save the spin positions
open("spin_positions.txt", "w") do io
  for i in 1:Lx
    println(io,x_pos[i],"\t",y_pos[i],"\t",z_pos[i])
  end
end
