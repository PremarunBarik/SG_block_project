using Plots

r = collect(0:0.001:1)
omega = 2.04279

function ro(omega, r)
  j_0 = sin(omega*r)/(omega*r)
  j_1 = (sin(omega*r)/(omega*r)^2) - (cos(omega*r)/(omega*r))
  N = (omega^4)/(omega^2 - sin(omega)^2)
  term_1 = sqrt(N)*j_0
  term_2 = (-1)*sqrt(N)*j_1
  
  final_term = (1/(4*pi))*(term_1^2 + term_2^2)
  return final_term
end

y = vec(zeros(length(r),1))
  
y .= ro.(omega, r)

plot(r, y, label = "omega: 2.04279")

savefig("myplot_1.png")
  
