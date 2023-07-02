#Cypher
#Indian Institute of Technology, Jodhpur
import numpy as np
import matplotlib.pyplot as plt

N = 50  # Total number of elements

# Creating an array of values from 1 to N with a step size of 0.1
S = np.arange(1, N, 0.1)

# Creating an array of values from 0.1 to 0.9 with 100 points
o = np.linspace(0.1, 0.9, 100)

# Defining the likelihood function
def L(S, o):
    return S * np.log(o) + (N - S) * np.log(1. - o)

# Creating a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Maximum Likelihood Estimation")

# Plotting the Bird's Eye View Heatmap
heatmap = ax1.imshow(L(np.repeat(S[:, np.newaxis], len(o), axis=1), np.repeat(o[np.newaxis, :], len(S), axis=0)),cmap='jet', origin='lower', aspect='auto', extent=[S.min(), S.max(), o.min(), o.max()])
ax1.set_xlabel('S')
ax1.set_ylabel('θ')
ax1.set_title("Bird's Eye View")

# Adding a vertical line at S = 12 on the heatmap plot
ax1.axvline(x=12, color='black')

# Plotting L(o|S=12)
ax2.plot(o, L(12, o), color='blue')
ax2.set_xlabel('o')
ax2.set_title("L(o|S=12)")

# Adjusting the spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Saving the figure as an image file
plt.savefig("./assets/stl-12.png")
