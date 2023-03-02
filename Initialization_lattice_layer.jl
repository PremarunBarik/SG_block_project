using Random

rng = MersenneTwister(1234)

#INITIALIZATION OF THE SG LATTICE
Lx_sg = 10
Ly_sg = 10
Lz_sg = 3

#NUMBER OF SPIN GLASS SPINS
N_sg = Lx_sg*Ly_sg*Lz_sg

#SPIN-GLASS SPIN POSITIONS
x_pos_sg = collect(1:Lx_sg)
x_pos_sg = repeat(x_pos_sg, inner=(Ly_sg,1))
x_pos_sg = repeat(x_pos_sg, outer=(Lz_sg,1))
y_pos_sg = collect(1:Ly_sg)
y_pos_sg = repeat(y_pos_sg, outer=(Lx_sg,1))
y_pos_sg = repeat(y_pos_sg, outer=(Lz_sg,1))
z_pos_sg = collect(1:Lz_sg)
z_pos_sg = repeat(z_pos_sg, inner=(Lx_sg*Ly_sg,1))
  
#INITIALIZATION OF THE FM LATTICE
x_pos_fm = [1,2,5,6,9,10]
y_pos_fm = [1,2,5,6,9,10]
Lx_fm = length(x_pos_fm)
Ly_fm = length(y_pos_fm)
Lz_sm = 1

#FERROMAGNET SPIN POSITIONS
x_pos_fm = repeat(x_pos_fm, inner=(Lx_fm,1))
x_pos_fm = repeat(x_pos_fm, outer=(Lz_fm,1))
y_pos_fm = repeat(y_pos_fm, outer=(Ly_fm,1))
y_pos_fm = repeat(y_pos_fm, outer=(Lz_fm,1))
z_pos_fm = collect(Lz_sg+1:Lz_sg+Lz_fm)
z_pos_fm = repeat(z_pos_fm, inner=(Lx_fm*Ly_fm,1))
  
#NUMBER OF FERROMAGNETIC SPINS
N_fm = Lx_fm*Ly_fm*Lz_fm

#TOTAL NUMBER OF SPINS
N_tot = N_sg+N_fm

#TOTAL SPIN POSITON OF SYSTEM
x_pos_tot = vcat(x_pos_sg, x_pos_fm)
y_pos_tot = vcat(y_pos_sg, y_pos_fm)
z_pos_tot = vcat(z_pos_sg, z_pos_fm)

 
#PRINTING SPIN POSITIONS
open("combined_spin_lattice.txt", "w") do io
  for i in 1:N_tot
    println(io,x_pos_tot[i],"\t",y_pos_tot[i],"\t",z_pos_tot[i])
  end
end
