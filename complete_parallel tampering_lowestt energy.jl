using GLMakie, Random, LinearAlgebra

rng = MersenneTwister(4321)

#NUMBER OF REPLICAS
replica_num = 10

#NUMBER OF MC MC STEPS 
MC_steps = 10000
MC_step_swap_interval = 10

#TEMPERATURE VALUES
min_temp = 0.1
max_temp = 3.1
temp_interval = (max_temp - min_temp)/replica_num
temp_replica = collect(min_temp:temp_interval:max_temp)

#VOLUME OF THE SPACE USED
L_x = 10
L_y = 10
L_z = 3

V = L_x*L_y*L_z

####################################################################################################################################################################################

#PERCENTAGE AND NUMBER OF UNIT CELL
Lattice_const = 1
Unit_cell_along_x = convert(Int64, trunc(L_x/Lattice_const))
Unit_cell_along_y = convert(Int64, trunc(L_y/Lattice_const))
Unit_cell_along_z = convert(Int64, trunc(L_z/Lattice_const))
Unit_cell_num = convert(Int64, trunc(Unit_cell_along_x * Unit_cell_along_y * Unit_cell_along_z))

Imp_percent = 10                                               #WE CONSIDER THE IMPURITY PERCENTAGE TO BE 10%
Imp_num = convert(Int64, trunc((Unit_cell_num*10)/100))

#POSITION OF IMPURITY INSIDE AN UNIT CELL 
x_pos_unit_cell_centre = Lattice_const/2
y_pos_unit_cell_centre = Lattice_const/2 
z_pos_unit_cell_centre = Lattice_const/2

#UNIT CELL REFERENCE OF IMPURITY 
Unit_cell_ref_of_imp = zeros(Imp_num, replica_num)

for i in 1:replica_num
  #UNIT CELL REFERENCE OF IMPUTITY POSITIONS
  global rand_int = rand(1:(Unit_cell_num - 1), Imp_num)
  global unique_int = unique(rand_int)

  while length(unique_int) < Imp_num
      global rand_int = rand(1:(Unit_cell_num - 1), (Imp_num-length(unique_int))) 
      global unique_int = vcat(unique_int, unique(rand_int))
  end

  Unit_cell_ref_of_imp[:,i] = unique_int
end


#POSITION OF IMPURITIES
x_pos_sg = zeros(Imp_num, replica_num)
y_pos_sg = zeros(Imp_num, replica_num)
z_pos_sg = zeros(Imp_num, replica_num)

#POSITION OF BASE LATTICE 
x_pos_base = zeros(Unit_cell_num, 1)
y_pos_base = zeros(Unit_cell_num, 1)
z_pos_base = zeros(Unit_cell_num, 1)

#CHANGE FROM CELL REFERENCE TO LATTICE POSITION
for j in 1:replica_num
  for i in 1:Imp_num
    x_pos_sg[i,j] = trunc(((Unit_cell_ref_of_imp[i,j] % (Unit_cell_along_x * Unit_cell_along_y))-1) / Unit_cell_along_x)*Lattice_const + x_pos_unit_cell_centre
    y_pos_sg[i,j] = (((Unit_cell_ref_of_imp[i,j] % (Unit_cell_along_y*Unit_cell_along_x)) - 1) % Unit_cell_along_y)*Lattice_const + y_pos_unit_cell_centre
    z_pos_sg[i,j] = trunc((Unit_cell_ref_of_imp[i,j]-1) / (Unit_cell_along_y *Unit_cell_along_x))*Lattice_const + z_pos_unit_cell_centre
  end
end

for i in 1:Unit_cell_num
    x_pos_base[i] = trunc(((i % (Unit_cell_along_x * Unit_cell_along_y))-1) / Unit_cell_along_x)*Lattice_const
    y_pos_base[i] = (((i % (Unit_cell_along_y*Unit_cell_along_x)) - 1) % Unit_cell_along_y)*Lattice_const 
    z_pos_base[i] = trunc((i-1) / (Unit_cell_along_y *Unit_cell_along_x))*Lattice_const
end

#IMPURITY SPIN VECTORS
N_sg = Imp_num                                              #CHANGING THE NOTATION FROM IMPURITY TO SPIN GLASS

x_dir_sg = zeros(N_sg, 1)
y_dir_sg = zeros(N_sg, 1)
z_dir_sg = zeros(N_sg, 1)


for i in 1:N_sg
    theta = rand(rng, Float64)*2*pi
    phi = rand(rng,Float64)*pi
    x_dir_sg[i] = sin(theta)cos(phi)
    y_dir_sg[i] = sin(theta)sin(phi)
    z_dir_sg[i] = cos(theta)
end

x_dir_sg = repeat(x_dir_sg, 1, replica_num)
y_dir_sg = repeat(y_dir_sg, 1, replica_num)
z_dir_sg = repeat(z_dir_sg, 1, replica_num)


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

x_pos_fm = repeat(x_pos_fm, 1, replica_num)
y_pos_fm = repeat(y_pos_fm, 1, replica_num)
z_pos_fm = repeat(z_pos_fm, 1, replica_num)
  
#NUMBER OF FERROMAGNETIC SPINS
N_fm = Lx_fm*Ly_fm*Lz_fm

#FERROMAGNETIC SPIN VECTORS
x_dir_fm = Float64[ 5.0 for i in 1:N_fm]
y_dir_fm = Float64[ 0.0 for i in 1:N_fm]
z_dir_fm = Float64[ 0.0 for i in 1:N_fm]

x_dir_fm = repeat(x_dir_fm, 1, replica_num)
y_dir_fm = repeat(y_dir_fm, 1, replica_num)
z_dir_fm = repeat(z_dir_fm, 1, replica_num)

#NUMBER OF TOTAL SPINS
N_tot = N_sg+N_fm

########################################################################################################################################################################################

#PRINTING INITIAL CONFIGURATION (using makie)
#aspect = (10, 10, 5)
#perspectiveness = 0.5
#fig = Figure(; resolution=(1200, 1200))
#ax = Axis3(fig[1, 1]; aspect, perspectiveness)
#quiver!(ax, x_pos_sg, y_pos_sg, z_pos_sg, x_dir_sg, y_dir_sg, z_dir_sg, color= :violet; arrowsize=0.1, )
#quiver!(ax, x_pos_fm, y_pos_fm, z_pos_fm, x_dir_fm, y_dir_fm, z_dir_fm, color= :black, arrowsize=0.1)
#scatter!(ax, x_pos_sg, y_pos_sg, z_pos_sg; markersize=5, color= :red)
#scatter!(ax, x_pos_base, y_pos_base, z_pos_base; markersize=1, color= :cyan)
#display(fig)
#Makie.save("Initial_spins.png", fig)

#PRINTING INITIAL CONFIG OF ALL REPLICAS USING PLOTS 

#for j in 1:replica_num
#  scatter!( x_pos_sg[:,j], y_pos_sg[:,j], z_pos_sg[:,j], markersize=3, color= :red)
#  display(scatter!( x_pos_base, y_pos_base, z_pos_base, markersize=1, color= :cyan))
#end

#####################################################################################################################################################################################

#RKKY INETRACTION J VALUE
global a = Lattice_const                               #near neighbour distance (Considering the matterial to be CuMn, with Mn density to be 10%)
global k_f = pi/a                         #2K_f*a the periodicity of RKKY
global alpha = 2*k_f*a

function RKKY_J(x_1, y_1, z_1, x_2, y_2, z_2, a, alpha)
  J_0 = -1
  r_ij = sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2)/a            #distance between spins in terms of near neighbour distance
  term_1 = cos(alpha*r_ij)/r_ij^3
  term_2 = sin(alpha*r_ij)/(alpha*(r_ij)^4)
  J = J_0*(term_1-term_2)
  return J
end

function RKKY_EA_NNN_fm_afm(x_1, y_1, z_1, x_2, y_2, z_2, a)      #NN FERROMAGNETIC AND NNN ANTIFEROMAGNETIC
  r_ij = sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2)            #distance between spins in terms of near neighbour distance
  if r_ij<(2*a)
    J = 1
  elseif r_ij>(2*a) && r_ij<(3*a)
    J = -1
  else
    J = 0
  end
  return J
end

function RKKY_EA_NNN_random(x_1, y_1, z_1, x_2, y_2, z_2, a, rng)
  r_ij = sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2)
  if r_ij<(3*a)
    J = (-1)^rand(rng, Int64)
  else
    J = 0
  end
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
interac_j = zeros(N_sg,N_sg,replica_num)

for k in 1:replica_num
  for i in 1:N_sg
    for j in 1:N_sg
        if i==j
            interac_j[i,j,k]=0
        else
            interac_j[i,j,k] = RKKY_EA_NNN_fm_afm(x_pos_sg[i], y_pos_sg[i], z_pos_sg[i], x_pos_sg[j], y_pos_sg[j], z_pos_sg[j], a)
        end
    end
  end
end

#####################################################################################################################################################################################

#plot(length_dist, interac_dist, label = "J distribution")
#j_dist = reshape(interac_j[:,:,5], (N_sg*N_sg,1))
#display(histogram(j_dist))
#ylabel!("Interaction coefficient (J)")
#xlims!(0,100)
#ylims!(-0.5,1)

####################################################################################################################################################################################


function dumbell_energy(x_fm, y_fm, z_fm, x_sg, y_sg, z_sg, s_sg_x, s_sg_y, s_sg_z, s_fm_x, s_fm_y, s_fm_z)
  E_0 = 1/10
  q_sg_plus = 1
  q_sg_minus = -1
  q_fm_plus = 5
  q_fm_minus = -5

  r_fm_d_plus_x = (x_fm+(s_fm_x/2)) 
  r_fm_d_plus_y = (y_fm+(s_fm_y)/2)
  r_fm_d_plus_z = (z_fm+(s_fm_z)/2)
  r_fm_d_minus_x = (x_fm-(s_fm_x/2))
  r_fm_d_minus_y = (y_fm-(s_fm_y/2))
  r_fm_d_minus_z = (z_fm-(s_fm_z/2))
  r_sg_d_plus_x = (x_sg+(s_sg_x/2))
  r_sg_d_plus_y = (y_sg+(s_sg_y)/2)
  r_sg_d_plus_z = (z_sg+(s_sg_z/2))
  r_sg_d_minus_x = (x_sg-(s_sg_x/2))
  r_sg_d_minus_y = (y_sg-(s_sg_y/2))
  r_sg_d_minus_z = (z_sg-(s_sg_z/2))

  term_1_denom = sqrt((r_fm_d_plus_x - r_sg_d_plus_x)^2 + (r_fm_d_plus_y - r_sg_d_plus_y)^2 + (r_fm_d_plus_z - r_sg_d_plus_z)^2)
  term_1 = q_fm_plus*q_sg_plus/term_1_denom
  term_2_denom = sqrt((r_fm_d_plus_x - r_sg_d_minus_x)^2 + (r_fm_d_plus_y - r_sg_d_minus_y)^2 + (r_fm_d_plus_z - r_sg_d_minus_z)^2)
  term_2 = q_fm_plus*q_sg_minus/term_2_denom
  term_3_denom = sqrt((r_fm_d_minus_x - r_sg_d_minus_x)^2 + (r_fm_d_minus_y - r_sg_d_minus_y)^2 + (r_fm_d_minus_z - r_sg_d_minus_z)^2)
  term_3 = q_fm_minus*q_sg_minus/term_3_denom
  term_4_denom = sqrt((r_fm_d_minus_x - r_sg_d_plus_x)^2 + (r_fm_d_minus_y - r_sg_d_plus_y)^2 + (r_fm_d_minus_z - r_sg_d_plus_z)^2)
  term_4 = q_fm_minus*q_sg_plus/term_4_denom

  E = E_0*(term_1 + term_2 + term_3 + term_4)

  return E
end

#MAGNETIC FIELD DUE TO BLOCKS
block_energy_sg = zeros(N_sg, replica_num)

for k in 1:replica_num
  for i in 1:N_sg                       #for loop for all the spin glass materials
    for j in 1:N_fm                     #for loop for all the ferromagnetic dumbells
      block_energy_sg[i,k] += dumbell_energy(x_pos_fm[j], y_pos_fm[j], z_pos_fm[j], x_pos_sg[i,k], y_pos_sg[i,k], z_pos_sg[i,k], x_dir_sg[i], y_dir_sg[i], z_dir_sg[i], x_dir_fm[j], y_dir_fm[j], z_dir_fm[j])          #ENERGY TERM COSIDERING DUMBELL MODEL
    end
  end
end

##################################################################################################################################################################################

#PRINTING HISTOGRAM OF ENERGY VALUES
#histogram(block_energy_sg[:,5])

#PRINTING THE ENERGY VALUES CORRESPONDING TO SPINGLASS POSITION
#for i in 1:replica_num
#  display(scatter(x_pos_sg[:,i], y_pos_sg[:,i], z_pos_sg[:,i], markersize=block_energy_sg[:,i], legend=false))
#end
#scatter(x_pos_fm, y_pos_fm, z_pos_fm)

#################################################################################################################################################################################
global count = 0
#MONTE CARLO FUNCTION
function one_MC(replica_index, N_sg, rng)
  rand_pos = rand(rng, (1:N_sg))
  rand_num_flip = rand(rng, Float64)
  interac_energy = 2*(x_dir_sg[rand_pos,replica_index]*(x_dir_sg[:,replica_index]'*interac_j[rand_pos,:,replica_index]) + y_dir_sg[rand_pos,replica_index]*(y_dir_sg[:,replica_index]'*interac_j[rand_pos,:,replica_index]) + z_dir_sg[rand_pos,replica_index]*(z_dir_sg[:,replica_index]'*interac_j[rand_pos,:,replica_index]))
  interac_energy_with_dumbell = interac_energy + block_energy_sg[rand_pos, replica_index]
  if interac_energy_with_dumbell<=0
    x_dir_sg[rand_pos, replica_index] = (-1)*x_dir_sg[rand_pos, replica_index]
    y_dir_sg[rand_pos, replica_index] = (-1)*y_dir_sg[rand_pos, replica_index]
    z_dir_sg[rand_pos, replica_index] = (-1)*z_dir_sg[rand_pos, replica_index]
  global  count = count + 1
  elseif (exp(-interac_energy_with_dumbell/temp_replica[replica_index])<rand_num_flip)
    x_dir_sg[rand_pos, replica_index] = (-1)*x_dir_sg[rand_pos, replica_index]
    y_dir_sg[rand_pos, replica_index] = (-1)*y_dir_sg[rand_pos, replica_index]
    z_dir_sg[rand_pos, replica_index] = (-1)*z_dir_sg[rand_pos, replica_index]
  global  count = count + 1
  end
  return interac_energy_with_dumbell
end

###################################################################################################################################################################################

#REPLICA SWAP FUNCTION
function replica_swap( replica_swap_index, rng)
  energy_1_x = x_dir_sg[:,replica_swap_index]'.*interac_j
  energy_1_x = sum(energy_1_x)
  energy_1_x = energy_1_x.*x_dir_sg[:, replica_swap_index]
  energy_1_y = y_dir_sg[:,replica_swap_index]'.*interac_j
  energy_1_y = sum(energy_1_y)
  energy_1_y = energy_1_y.*y_dir_sg[:, replica_swap_index]
  energy_1_z = z_dir_sg[:,replica_swap_index]'.*interac_j
  energy_1_z = sum(energy_1_z)
  energy_1_z = energy_1_z.*z_dir_sg[:, replica_swap_index]
  energy_1_tot = energy_1_x.+energy_1_y.+energy_1_z.+block_energy_sg[:, replica_swap_index]
  energy_1_tot = sum(energy_1_tot)

  energy_2_x = x_dir_sg[:,replica_swap_index+1]'.*interac_j
  energy_2_x = sum(energy_2_x)
  energy_2_x = energy_2_x.*x_dir_sg[:, replica_swap_index+1]
  energy_2_y = y_dir_sg[:,replica_swap_index+1]'.*interac_j
  energy_2_y = sum(energy_2_y)
  energy_2_y = energy_2_y.*y_dir_sg[:, replica_swap_index+1]
  energy_2_z = z_dir_sg[:,replica_swap_index+1]'.*interac_j
  energy_2_z = sum(energy_2_z)
  energy_2_z = energy_2_z.*z_dir_sg[:, replica_swap_index+1]
  energy_2_tot = energy_2_x.+energy_2_y.+energy_2_z.+block_energy_sg[:, replica_swap_index+1]
  energy_2_tot = sum(energy_2_tot)

  del_energy = energy_2_tot - energy_1_tot
  del_temp = (1/temp_replica[replica_swap_index+1]) - (1/temp_replica[replica_swap_index])

  swap_probability = del_energy*del_temp

  rand_num_swap = rand(rng, Float64)

  if swap_probability<=0
    swap_temp = temp_replica[replica_swap_index]
    temp_replica[replica_swap_index] = temp_replica[replica_swap_index+1]
    temp_replica[replica_swap_index] = swap_temp
  elseif exp(-swap_probability)<rand_num_swap
    swap_temp = temp_replica[replica_swap_index]
    temp_replica[replica_swap_index] = temp_replica[replica_swap_index+1]
    temp_replica[replica_swap_index] = swap_temp
  end
end

########################################################################################################################################################################################

#MAIN BODY

for i in 1:MC_steps
  for j in 1:replica_num
    replica_index = j
    one_MC(replica_index, N_sg, rng)
  end
  #TEMP SWAPPING BETWEEN REPLICAS 
  if (i%MC_step_swap_interval)==0
    swap_type = i%MC_step_swap_interval
    if (swap_type%2)==0
      for k in 2:2:(replica_num-2)
        replica_swap_index = k
        replica_swap( replica_swap_index, rng)
      end
    else
      for k in 1:2:(replica_num-1)
        replica_swap_index = k
        replica_swap( replica_swap_index, rng)
      end
    end
  end
end

#############################################################################################################################################################################################
 
#PRINTING FNITIAL CONFIGURATION (using makie)
replica_print= 1
x_pos_sg_print = vec(x_pos_sg[:,replica_print])
y_pos_sg_print = vec(y_pos_sg[:,replica_print])
z_pos_sg_print = vec(z_pos_sg[:,replica_print])
x_dir_sg_print = vec(x_dir_sg[:,replica_print])
y_dir_sg_print = vec(y_dir_sg[:,replica_print])
z_dir_sg_print = vec(z_dir_sg[:,replica_print])

aspect = (10, 10, 5)
perspectiveness = 0.5
fig = Figure(; resolution=(1200, 1200))
ax = Axis3(fig[1, 1]; aspect, perspectiveness)
quiver!(ax, x_pos_sg_print, y_pos_sg_print, z_pos_sg_print, x_dir_sg_print, y_dir_sg_print, z_dir_sg_print, color= :violet; arrowsize=0.1)
#quiver!(ax, x_pos_fm, y_pos_fm, z_pos_fm, x_dir_fm, y_dir_fm, z_dir_fm, color= :black, arrowsize=0.1)
display(fig)

##############################################################################################################################################################################################
