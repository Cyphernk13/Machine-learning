#Cypher
#Indian Institute of Technology, Jodhpur
# Importing necessary modules
import numpy as np
import matplotlib.pyplot as plt

N = 50  # The Total number of elements

# Creating an array of values from 1 to N
S = np.arange(1, N + 1)

# Creating an array of values from 0.1 to 0.9 with 100 points
theta = np.linspace(0.1, 0.9, 100)

# Performing Maximum Likelihood Estimation
# Creating a grid of values for S and theta
S_grid, theta_grid = np.meshgrid(S, theta)

# Calculating the likelihood function values for each combination of S and theta
L = S_grid * np.log(theta_grid) + (N - S_grid) * np.log(1 - theta_grid)

# Creating a new figure
fig = plt.figure()

# Adding a 3D subplot to the figure
ax = fig.add_subplot(111, projection='3d')

# Creating a surface plot using S_grid, theta_grid, and L
s = ax.plot_surface(S_grid, theta_grid, L, cmap='jet')

# Setting labels for the x, y, and z axes
ax.set_xlabel('S')
ax.set_ylabel('theta')
ax.set_zlabel('L(theta|S)')

# Adjusting the view angle of the plot
ax.view_init(65, 15)

# Saving the figure as an image file
plt.savefig("./assets/stl.png")
