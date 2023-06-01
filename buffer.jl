#REFERENCE POSITION OF THE SPIN ELEMENTS
x_pos_sg = zeros(N_sg, 1)
y_pos_sg = zeros(N_sg, 1)
z_pos_sg = zeros(N_sg, 1)

for i in 1:N_sg
    x_pos_sg[i] = trunc((((i-1)% (n_x*n_y)))/n_x)+1             #10th position
    y_pos_sg[i] = (((i-1)%(n_x*n_y))%n_y)+1                     #1th position
    z_pos_sg[i] = trunc((i-1)/(n_x*n_y)) +1                     #100th position
end

#x_pos_sg = repeat(x_pos_sg, 1, replica_num)
#y_pos_sg = repeat(y_pos_sg, 1, replica_num)
#z_pos_sg = repeat(z_pos_sg, 1, replica_num)

#INITIALIZATION OF THE FM LATTICE
x_pos_fm = [1.5,5.5,9.5,]
y_pos_fm = [1.0,1.5,2.0,5.0,5.5,6.0,9.0,9.5,10.0]
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

#NUMBER OF FERROMAGNETIC SPINS
N_fm = Lx_fm*Ly_fm*Lz_fm

#FERROMAGNETIC SPIN VECTORS
x_dir_fm = Float64[ 1.0 for i in 1:N_fm]
y_dir_fm = Float64[ 0.0 for i in 1:N_fm]
z_dir_fm = Float64[ 0.0 for i in 1:N_fm]
