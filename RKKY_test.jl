
using Pkg
Pkg.add("Plots")
using Random, GLMakie, Plots

mx = collect(0:0.01:3)
N = length(mx)
a = 1/5
alpha = 4
function RKKY_J(x_1, x_2, a, alpha)
         J_0 = (a^2)*alpha
           #distance between spins in terms of near neighbour distance
         r_ij = sqrt((x_1-x_2)^2 )/a
         term_1 = cos(alpha*r_ij)/r_ij^3
           term_2 = sin(alpha*r_ij)/(alpha*(r_ij)^4)
             J = J_0*(term_1-term_2)
               return J
       end
interac = zeros(N,1)
for i in 1:N
interac[i] = RKKY_J(mx[1], mx[i], a, alpha)
end

mx = vec(mx)
interac = vec(interac)

plot(mx, interac)
#open("RKKY_J_value_alpha7.0.txt", "w") do io 					#creating a file to save data
#for i in 1:N
#	println(io,i,"\t",interac[i],"\t",mx[i])
#end
#end
#println("--COMPLETE--")
