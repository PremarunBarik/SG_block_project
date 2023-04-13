using GLMakie, Random, AbstractAlgebra

#VOLUME OF THE UNIT
L_x = 100
L_y = 100
L_z = 30

V = L_x*L_y*L_z

#PERCENTAGE AND NUMBER OF UNIT CELL
Lattice_const = 1
Unit_cell_along_x = convert(Int64, trunc(L_x/Lattice_const))
Unit_cell_along_y = convert(Int64, trunc(L_y/Lattice_const))
Unit_cell_along_z = convert(Int64, trunc(L_z/Lattice_const))
Unit_cell_num = convert(Int64, trunc(Unit_cell_along_x * Unit_cell_along_y * Unit_cell_along_z))

Imp_percent = 10                                                #WE CONSIDER THE IMPURITY PERCENTAGE TO BE 10%
Imp_num = convert(Int64, trunc((Unit_cell_num*10)/100))

#POSITION OF IMPURITY INSIDE AN UNIT CELL 
x_pos_unit_cell_centre = Lattice_const/2
y_pos_unit_cell_centre = Lattice_const/2 
z_pos_unit_cell_centre = Lattice_const/2

#UNIT CELL REFERENCE OF IMPUTITY POSITIONS
global rand_int = rand(0:(Unit_cell_num - 1), Imp_num)
global unique_int = unique(rand_int)

while length(unique_int) < Imp_num
    global rand_int = rand(1:(Unit_cell_num - 1), (Imp_num-length(unique_int))) 
    global unique_int = vcat(unique_int, unique(rand_int))
end

Unit_cell_ref_of_imp = unique_int

#POSITION OF IMPURITIES
x_pos_sg = vec(zeros(Imp_num, 1))
y_pos_sg = vec(zeros(Imp_num, 1))
z_pos_sg = vec(zeros(Imp_num, 1))

#POSITION OF BASE LATTICE 
x_pos_base = vec(zeros(Unit_cell_num, 1))
y_pos_base = vec(zeros(Unit_cell_num, 1))
z_pos_base = vec(zeros(Unit_cell_num, 1))

#CHANGE FROM CELL REFERENCE TO LATTICE POSITION
for i in 1:Imp_num
    x_pos_sg[i] = trunc(((Unit_cell_ref_of_imp[i] % (Unit_cell_along_x * Unit_cell_along_y))-1) / Unit_cell_along_x)*Lattice_const + x_pos_unit_cell_centre
    y_pos_sg[i] = (((Unit_cell_ref_of_imp[i] % (Unit_cell_along_y*Unit_cell_along_x)) - 1) % Unit_cell_along_y)*Lattice_const + y_pos_unit_cell_centre
    z_pos_sg[i] = trunc((Unit_cell_ref_of_imp[i]-1) / (Unit_cell_along_y *Unit_cell_along_x))*Lattice_const + z_pos_unit_cell_centre
end

for i in 1:Unit_cell_num
    x_pos_base[i] = trunc(((i % (Unit_cell_along_x * Unit_cell_along_y))-1) / Unit_cell_along_x)*Lattice_const
    y_pos_base[i] = (((i % (Unit_cell_along_y*Unit_cell_along_x)) - 1) % Unit_cell_along_y)*Lattice_const 
    z_pos_base[i] = trunc((i-1) / (Unit_cell_along_y *Unit_cell_along_x))*Lattice_const
end



#scatter!(x_pos_base, y_pos_base, z_pos_base, markersize=1, legend=false)
#scatter!(x_pos_sg, y_pos_sg, z_pos_sg, markersize=3, legend=false)


#PRINTING INITIAL CONFIGURATION (using makie)
aspect=(1, 1, 0.5)
perspectiveness=0.5
fig = Figure(; resolution=(1200, 400))
ax = Axis3(fig[1, 1]; aspect, perspectiveness)
scatter!(ax, x_pos_sg, y_pos_sg, z_pos_sg; markersize=5, color= :red)
scatter!(ax, x_pos_base, y_pos_base, z_pos_base; markersize=1, color= :black)
display(fig)
