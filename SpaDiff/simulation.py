import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
# from sklearn import datasets

# import functino for kscore
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from kscore.estimators import *
from kscore.kernels import *
import kscore

import argparse
import imageio # for creating gif
from io import BytesIO
# from scipy import sparse, io # for read sparse matrix
from scipy.spatial import KDTree # for finding neighbor
from scipy.spatial import cKDTree # for finding neighbor
from scipy.stats import gaussian_kde # for estimating the variance of the distance distribution

from utils import *

def simulation_data_convert(data, space_size=30, grid_size=1):
    """
    The function first rescales the space based on the minimum and maximum coordinates in the data and sets a grid
    to separate this adjusted space into small areas. Then, it counts the number of nodes within each small area,
    including empty areas with a count of 0.

    Parameters:
    space_size: The level for scaling the data.
    grid_size: The unit length of the grid.

    Returns:
    df: The count matrix recording the x and y coordinate of each small area and the count of nodes within it.
    """
    # Calculate the grid indices
    min_x, min_y = np.floor(data.min(axis=0) / grid_size).astype(int) * grid_size * space_size
    max_x, max_y = np.ceil(data.max(axis=0) / grid_size).astype(int) * grid_size * space_size

    grid_indices = np.floor(data * space_size / grid_size).astype(int)

    # Count nodes in each grid
    unique_indices, counts = np.unique(grid_indices, axis=0, return_counts=True)

    # Create DataFrame for counts
    df_counts = pd.DataFrame(unique_indices, columns=['x', 'y'])
    df_counts['count'] = counts

    # Generate all possible grid coordinates
    all_indices = pd.DataFrame(np.array(np.meshgrid(range(min_x, max_x + 1), range(min_y, max_y + 1))).T.reshape(-1, 2), columns=['x', 'y'])

    # Merge count data with all grid coordinates
    df = pd.merge(all_indices, df_counts, on=['x', 'y'], how='left')
    df['count'].fillna(0, inplace=True)  # Replace NaN with 0 for grids with no nodes

    return df




def distribution_plot(data, figure_size=(8,8), size=10, x_limit=None, y_limit=None):
  """
  Plot the distribution of the count matrix.

  Paramters:
  data: The spatial and count iformation generated from function simulation_data_convert.
  figure_size: A tuple recording the width and height of the figure.
  size: The spot size in the figure.

  Returns:
  None
  """
  # Transfer count vector to coordinate set
  data_loc = count_to_coordinate(data['count'], data[['x','y']])

  # Plot the score field
  plt.figure(figsize=figure_size)
  plt.scatter(data_loc['x'], data_loc['y'], s=size, c='steelblue', alpha=0.2)
  
  if x_limit is not None:
    plt.xlim(x_limit)
  if y_limit is not None:
    plt.ylim(y_limit)
  
  plt.show()

def simulation_add_noise(data, sigma=1, seed=None):
  """
  Add noise to the simulated data.

  Parameters:
  data: The spatial and count iformation generated from function simulation_data_convert.
  sigma: Noise level.

  Returns:
  data: The data after adding noise.
  """
  # Transfer count vector to coordinate set
  data_loc = count_to_coordinate(data['count'], data[['x','y']])
  
  if seed is not None:
        tf.random.set_seed(seed)

  # Add noise to the data
  x = np.array(data_loc, np.float32)
  rand = tf.random.normal((x.shape[0], 2), mean=0.0, stddev=1)
  x = x + sigma*rand

  # Transfer coordinate set to count vector
  data_loc = pd.DataFrame(x)
  data_loc.columns = ['x','y']
  data_update = coordinate_to_count(data_loc, data[['x','y']])

  return data_update