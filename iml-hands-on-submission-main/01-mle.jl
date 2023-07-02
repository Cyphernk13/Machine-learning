#Cypher
#Indian Institute of Technology, Jodhpur
N = 50;  # The total number of elements

# Now generate a range of values from 1 to N with a step size of 0.1
S = range(1., N, step=0.1);

# Generating a range of values from 0.1 to 0.9 with a length of 100
o = range(0.1, 0.9, length=100);

# now let us define a function L that calculates the likelihood function
L(S, o) = S * log(o) + (N - S) * log(1. - o);

# Import the Plots package
using Plots
gr()

# Generating a plot of the likelihood function using S and o as input
p2 = Plots.heatmap(S, o, (S, o) -> L(S, o), color=:jet, xlabel="S", ylabel="θ", title="Bird's Eye view");

# Now let us add a vertical line at S = 25 on the heatmap plot
SS = 25;
vline!([SS], label=false, color=:black);

# Generating a plot of the likelihood function at S = 25 using o as input
P3 = Plots.plot(o, o -> L(SS, o), label=false, xlabel='o', title="L(o|S=$SS)");

# Plotting the heatmap plot and the likelihood function
Plots.plot(p2, P3)

# Saving the combined plot as an image file
savefig("./assets/julia.png")
