using Random

rng = MersenneTwister(1234)

#INITIALIZATION OF THE LATTICE
Lx = 10
Ly = 10
Lz = 3

#NUMBER OF SPINS
N = Lx*Ly*Lz

#SPIN-GLASS SPIN POSITIONS
x_pos = collect(1:Lx)
x_pos = repeat(x_pos, inner=(Lx,1))
x_pos = repeat(x_pos, outer=(Lz,1))
y_pos = collect(1:Ly)
y_pos = repeat(y_pos, outer=(Lx,1))
y_pos = repeat(y_pos, outer=(Lz,1))
z_pos = collect(1:Lz)
z_pos = repeat(z_pos, inner=(Lx*Ly,1))

#PRINTING SPIN POSITIONS
open("spin_lattice.txt", "w") do io
  for i in 1:N
    println(io,x_pos[i],"\t",y_pos[i],"\t",z_pos[i])
  end
end
