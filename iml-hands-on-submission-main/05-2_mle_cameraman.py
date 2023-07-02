#Cypher
#Indian Institute of Technology, Jodhpur
#importing necessary modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Reading the hands.tif image using cv2.imread
lambd = cv2.imread('hands.tif')

# Converting the image to grayscale using cv2.cvtColor
lambd = cv2.cvtColor(lambd, cv2.COLOR_BGR2GRAY) / 255

T = 100  # Total number of iterations

# Repeating the lambd grayscale image to create lambdT with height, width, T
lambdT = np.repeat(lambd[:, :, np.newaxis], T, axis=2)

# Generating random samples
x = stats.poisson.rvs(lambdT)
y = (x >= 1).astype(float)

# Estimating the MLE parameter lambdhat 
lambdhat = -np.log(1 - np.mean(y, axis=2))

# Displaying the estimated parameter lambdhat as an image
plt.imshow(lambdhat, cmap='gray')

# Saving the figure as an image file
plt.savefig('./assets/hands.png')
