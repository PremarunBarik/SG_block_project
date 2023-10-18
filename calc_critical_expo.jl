using Plots, DelimitedFiles

#define error margin
global error_margin = 0.021

#initializing the limit of bext fit coefficient
global neu_min = 2
global neu_max = 10

#---------------------------------------------------------------------------------------------#

#reading data ponts from text file
data_1 = readdlm("2D_EA_OverlapBinder_4x4_MC100K.txt")
data_2 = readdlm("2D_EA_OverlapBinder_6x6_MC100K.txt")
data_3 = readdlm("2D_EA_OverlapBinder_8x8_MC100K.txt")
#data_4 = readdlm("2D_EA_OverlapBinder_25x25_100K.txt")

#mentioning the system size of data files
L_1 = 4
L_2 = 6
L_3 = 8
#L_4 = 12

temp_1 = data_1[1:25,2]
temp_2 = data_2[1:25,2]
temp_3 = data_3[1:25,2]
#temp_4 = data_4[1:25,2]

overlap_binder_1 = data_1[1:25,3]
overlap_binder_2 = data_2[1:25,3]
overlap_binder_3 = data_3[1:25,3]
#overlap_binder_4 = data_4[1:36,3]

#---------------------------------------------------------------------------------------------#

#function to calculate the best slope fo fit data
function calculate_best_fit_slope(neu)
    x_1 = temp_1*(L_1^(1/neu))
    x_2 = temp_2*(L_2^(1/neu))
    x_3 = temp_3*(L_3^(1/neu))
#    x_4 = temp_4*(L_4^(1/neu))

    y_1 = overlap_binder_1
    y_2 = overlap_binder_2
    y_3 = overlap_binder_3
#    y_4 = overlap_binder_4

    global x_tot = vcat(x_1, x_2, x_3)
    global y_tot = vcat(y_1, y_2, y_3)

    x_av = sum(x_tot)/ length(x_tot)
    y_av = sum(y_tot)/ length(y_tot)

    global slope = sum((x_tot .- x_av) .* (y_tot .- y_av))/sum((x_tot .- x_av) .^ 2)
    global b = y_av - (slope * x_av)
    
end

#function to calculate variance of data points given a parameter value
function calculate_variance(neu)
    calculate_best_fit_slope(neu)

    y_best_fit = (x_tot * slope) .+ b
    global variance = sum(sqrt.((y_tot .- y_best_fit) .^2))/ length(y_tot)
    return variance
end

#function to calculate error margin for two different parameter value
function calculate_error(neu_1, neu_2)
    variance_1 = calculate_variance(neu_1)
    variance_2 = calculate_variance(neu_2)

    global error = variance_1 + variance_2
    return error
end

calculate_error(neu_min, neu_max)
global count = 0

while (error) >= error_margin

    global neu_half = (neu_max + neu_min)/2

    error_1 = calculate_error(neu_min, neu_half)
    error_2 = calculate_error(neu_max, neu_half)

    if (error_1 < error_2)
        global neu_max = neu_half
    elseif (error_1 > error_2)
        global neu_min = neu_half
    end

    global count += 1
end

println("neu: $((neu_max + neu_min)/2), delta_nue: $(neu_max - neu_min), count: $(count)")
