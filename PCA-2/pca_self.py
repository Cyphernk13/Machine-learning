import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('innercity.csv')

# Select the columns for the dataset
columns = ['room_bed', 'room_bath', 'living_measure', 'lot_measure', 'ceil', 'coast',
           'sight', 'condition', 'quality', 'ceil_measure', 'basement', 'yr_built',
           'yr_renovated', 'zipcode', 'lat', 'long', 'living_measure15', 'lot_measure15',
           'furnished', 'total_area']

# Convert columns to numeric types and handle missing values
dataset = data[columns].apply(pd.to_numeric, errors='coerce').values
column_means = np.nanmean(dataset, axis=0)
inds = np.where(np.isnan(dataset))
dataset[inds] = np.take(column_means, inds[1])

def pca(dataset, n_comp):
    # Normalize the data
    dataset = dataset - np.mean(dataset, axis=0)
    
    # Perform PCA
    pca = PCA(n_components=n_comp)
    reduced_data = pca.fit_transform(dataset)
    
    return reduced_data

# Perform PCA on the dataset
reduced_data = pca(dataset, 2)

# Convert the reduced data to a DataFrame
reduced_data = pd.DataFrame(reduced_data)

# Plot the reduced data
plt.scatter(reduced_data[0], reduced_data[1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA - Reduced Data')
# Save the plot at a specified location
plt.savefig('./assets/pca_plot.png')

# Display the plot
plt.show()
