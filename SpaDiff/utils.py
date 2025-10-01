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


# # Kernel Class
# class CurlFreeIMQ:
#     def __init__(self, alpha=1.0):
#         self.alpha = alpha

#     def pairwise_differences(self, x, y):
#         return tf.expand_dims(x, 1) - tf.expand_dims(y, 0)

#     def kappa(self, x, y, kernel_hyperparams):
#         diff = self.pairwise_differences(x, y)
#         norm_sq = tf.reduce_sum(tf.square(diff), axis=-1)
#         return 1.0 / tf.sqrt(norm_sq + kernel_hyperparams)

#     def grad_kappa(self, x, y, kernel_hyperparams):
#         diff = self.pairwise_differences(x, y)
#         norm_sq = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)
#         base = (norm_sq + kernel_hyperparams) ** (-1.5)
#         return -diff * base

# # Score function estimator
# def compute_score_gradients(x_query, x_samples, lam=1.0, kernel_width=1.0):
#     kernel = CurlFreeIMQ()

#     grad_kappa = kernel.grad_kappa(x_query, x_samples, kernel_width)
#     kappa = kernel.kappa(x_query, x_samples, kernel_width)

#     score_estimate = tf.reduce_mean(grad_kappa, axis=1) + lam * tf.reduce_mean(kappa[..., None] * (x_query[:, None, :] - x_samples[None, :, :]), axis=1)

#     return score_estimate



def count_to_coordinate(count, spatial):
  """
  Transfer the count matrix to coordinate set.

  Parameters:
  count: The count vector of one gene on all spots.
  spatial: The spatial location information. Two columns are x and y coordinate respectively.

  Returns:
  result: The converted coordinate sets, where each spot's coordinate will have number of count replications.
  """
  data = pd.concat([spatial, count], axis=1)
  data.columns = ['x', 'y', 'n']

  # Filter non-zero counts
  non_zero_data = data[data['n'] != 0]

  # Replicating rows based on the count and preparing final output
  result = non_zero_data.loc[non_zero_data.index.repeat(non_zero_data['n'])]
  result.reset_index(drop=True, inplace=True)
  result = result[['x', 'y']]

  return result


def coordinate_to_count(data_loc, spatial):
    """
    Convert a set of coordinates back to a count vector based on the nearest spatial locations.
    Safely merges count data to avoid index mismatch errors.

    Parameters:
    data_loc: DataFrame with columns ['x', 'y'] representing the coordinates to transform.
    spatial: DataFrame with columns ['x', 'y'] representing spatial locations.

    Returns:
    result: DataFrame with the original spatial coordinates and a new 'count' column indicating
            the number of coordinates nearest to each spatial location.
    """

    # Build a KDTree for spatial locations
    tree = cKDTree(spatial[['x', 'y']])

    # Query this tree for the nearest neighbor to each point in the coordinate set
    distances, indices = tree.query(data_loc[['x', 'y']], k=1)

    # Count the occurrences of each index (i.e., each spatial location)
    counts = pd.Series(indices).value_counts().sort_index()

    # Prepare the output DataFrame
    result = spatial.copy()
    result['count'] = 0  # Initialize all counts to zero
    result = result.reset_index(drop=True) # replace the row index from barcode to index
    result.loc[counts.index, 'count'] = counts.values  # Assign the counts

    return result



# def calculate_min_distances(data_loc, target_loc):
#   """
#   Calculate the distance between the given set of spots and the given target spots
#   (could be boudary or known spatial domain).

#   Parameters:
#   data_loc: The coordinate set of the outside spots.
#   target_loc: The coordinate set of the target spots (usually is boundary or known spatial domain).

#   Returns:
#   min_distances: The distances of each spot to the target domain.
#   """
#   # Convert dataframes to numpy arrays for distance calculation
#   data = data_loc.to_numpy()
#   target = target_loc.to_numpy()

#   # Calculate all pairwise distances using broadcasting
#   dist_matrix = np.sqrt((data[:, np.newaxis, 0] - target[:, 0])**2 + (data[:, np.newaxis, 1] - target[:, 1])**2)

#   # Find the minimum distance for each point to the boundary
#   min_distances = np.min(dist_matrix, axis=1)

#   return min_distances


# def density_comparison_plot(data1, data2, label=['before', 'after'], figure_size=(8,6)):
#   """
#   Plot the histgram and dennsity of two campared datasets.

#   Parameters:
#   data1: First compared data.
#   data2: Second compared data.
#   label: A two-elements list recording the label for data1 and data2.
#   figure_size: figure_size: A tuple recording the width and height of the figure.

#   Returns:
#   None
#   """
#   # Create the histogram plot
#   plt.figure(figsize=figure_size)
#   plt.hist(data1, bins=30, density=True, alpha=0.5, color='blue')
#   plt.hist(data2, bins=30, density=True, alpha=0.5, color='red')

#   # Adding labels and title
#   plt.xlabel('Distances')
#   plt.ylabel('Density')
#   plt.title('Comparison of Two Distributions')

#   # Calculate KDEs
#   kde1 = gaussian_kde(data1)
#   kde2 = gaussian_kde(data2)
#   x1 = np.linspace(min(data1), max(data1), 1000)
#   x2 = np.linspace(min(data2), max(data2), 1000)

#   # Plot KDEs on top of the histograms
#   plt.plot(x1, kde1(x1), color='blue', label=label[0])
#   plt.plot(x2, kde2(x2), color='red', label=label[1])
#   plt.legend()

#   # Show plot
#   plt.show()


def density_scaler(x, neighbor_range=20.0):
  """
  Calculate the scaler for the score function to make the spot with a dense neighborhood have a smaller score function
  and the spot with a sparse neighborhood have a bigger score function to accelerate the process.

  Parameters:
  x: The coordinate set of one gene.
  neigbor_range: The range for neighbors.

  Returns:
  A multipliers to rescale the gradient.
  """
  # Count the number of neighbors for each spot using KD-Tree
  tree = KDTree(x)
  neighbors_list = tree.query_ball_tree(tree, r=neighbor_range)
  neighbors_count = [len(neighbors) - 1 for neighbors in neighbors_list]

  # Construct the tensorflow object
  neighbors_count_tensor = tf.constant(neighbors_count, dtype=tf.float32)

  # Find the biggest density
  max_value_tensor = tf.fill(neighbors_count_tensor.shape, tf.reduce_max(neighbors_count_tensor))

  # Find the location that have count
  safe_denominator = tf.where(neighbors_count_tensor > 0, neighbors_count_tensor, 1)

  # Calculate the multiplier
  multipliers = max_value_tensor / safe_denominator
  multipliers_reshaped = tf.reshape(multipliers, (-1, 1))

  return multipliers_reshaped


def gradient_calculation(data_loc, lam=1.0, kernel_width=80.0, subsample=True, fraction=0.2, sample_count=500, seed=1234):
  """
  Calculate the gradient of the given coordinate sets.

  Parameters:
  data_loc: The coordinate sets of one gene.
  lam: The parameter lambda for nu estimator.
  kernel_width: The width of the kernel in the gradient calculation.
  subsample: Indicator of whether to use subsample to accelerate calculation.
  fraction: The proportion of the subsample.
  sample_count: The exact number of subsamples to take.
  seed: Random seed.

  Returns:
  gradient: A two dimensional dataframe recording the gradient along x and y direction.
  """
  # Set random seed
  tf.compat.v1.set_random_seed(seed)
  np.random.seed(seed)

  # Sample from the whole data to accelerate computation
  if subsample:
    if len(data_loc)*fraction > sample_count:
      x_samples = data_loc.sample(n=sample_count)
    else:
      x_samples = data_loc.sample(frac=0.2)
  else:
    x_samples = data_loc

  # Convert location to tensor
  samples = tf.constant(x_samples.values)
  samples = tf.cast(samples, dtype=tf.float32)

  # # Calculate score function
  estimator = kscore.estimators.NuMethod(lam=lam, kernel=CurlFreeIMQ())
  estimator.fit(samples, kernel_hyperparams=kernel_width)

  # rescale the gradient according to the density
  x = np.array(data_loc, np.float32)
  multipliers_reshaped = density_scaler(x)

  # calculate gradient
  gradient = estimator.compute_gradients(x) * multipliers_reshaped
  # gradient = -compute_score_gradients(x, samples, lam, kernel_width) * multipliers_reshaped

  return gradient


def gradient_scaler(gradient, max_gradient=5):
  """
  Calculate a scaler for rescaling the gradient by the max length of score function.
  This step is used for standarlizing the parameters selection in the subsequent analysis.

  Parameters:
  gradient: The two dimensional tensor recording the gradient along x and y directions.
  max_gradient: The max gradient to set.

  Returns:
  gradient_scale: A scaler for rescaling the gradient by the max length of score function.
  """
  # calculate the scale of the
  norms = tf.norm(gradient, axis=1)

  # Calculate the max length of score function
  max_length = tf.reduce_max(norms)

  # Scale the gradient by the max length of score function
  gradient_scale = max_gradient/max_length
  gradient_scale = gradient_scale.numpy()

  return gradient_scale


def safe_denoise(gene_data):
  try:
    print(f"Starting denoising for gene")
    result = denoise_single_gene(gene_data)
    print(f"Completed denoising for gene")
    return result
  except Exception as e:
    print(f"Error processing gene data: {e}")
    return gene_data 


def split_random_balanced_chunks(df, max_chunk_size, seed=1234):
    """
    Shuffle and split a DataFrame into evenly sized chunks, each <= max_chunk_size
    """
    total = len(df)

    if total <= max_chunk_size:
      return [df]

    else:
      rng = np.random.default_rng(seed)
      shuffled_idx = rng.permutation(df.index)
      df_shuffled = df.loc[shuffled_idx].reset_index(drop=True)

      num_chunks = math.ceil(total / max_chunk_size)
      chunk_size = math.ceil(total / num_chunks)

      return [df_shuffled.iloc[i*chunk_size : (i+1)*chunk_size] for i in range(num_chunks)]
  

def denoise_single_gene(args):
        """
        Denoise the selected data using the diffusion process. This step is the main step of the process.

        Parameters:
        gene_index: The idex of the target gene. If None, then apply the denoise procedure for all genes.
        step: The number of steps of the diffusion process.
        step_size: The length of each step.
        sigma: The dispersion level of the diffusion process.
        lam: The parameter lambda for nu estimator.
        kernel_width: The width of the kernel in the gradient calculation.
        subsample: Indicator of whether to use subsample to accelerate calculation.
        fraction: The proportion of the subsample.
        gradient_scale: The scaler for rescaling the gradient. If None, the algorithm will automatically calculate the scaler for each gene.
        max_gradient: The max gradient to set.
        sample_count: The exact number of subsamples to take.
        if_output: The indicator of whether return the updated data.
        seed: Random seed.

        Retruns:
        data_update: The updated count matrix.
        """
        # Unpack parameters
        gene_data, spatial_data, params, threshold = args
        step, step_size, sigma, lam, kernel_width, subsample, fraction, sample_count, gradient_scale, max_gradient, seed = params
    
        gene_data = pd.DataFrame(gene_data)
        spatial_data = pd.DataFrame(spatial_data)
        gene_data.index = spatial_data.index
    
        data_update = gene_data.copy()
        data_loc = count_to_coordinate(gene_data, spatial_data)

        chunks = split_random_balanced_chunks(data_loc, max_chunk_size=threshold, seed=seed)
        denoised_chunks = []
        
        # TensorFlow random seed initialization inside process
        # tf.compat.v1.set_random_seed(seed)

        for chunk_df in chunks:
          x = np.array(chunk_df[['x', 'y']].values, dtype=np.float32)

          # Optional: Gradient scale per chunk (auto or passed)
          for update in range(step):
              data_tmp = pd.DataFrame(x, columns=['x', 'y'])

              gradient = gradient_calculation(
                  data_tmp, lam=lam, kernel_width=kernel_width,
                  subsample=subsample, fraction=fraction,
                  sample_count=sample_count, seed=seed
              )

              if gradient_scale is None:
                  g_scale = gradient_scaler(gradient, max_gradient=max_gradient)
              else:
                  g_scale = gradient_scale

              rand = tf.random.normal((x.shape[0], 2), mean=0.0, stddev=1)
              x = x + step_size * (sigma*gradient*g_scale + math.sqrt(sigma)*rand)

          denoised_chunks.append(pd.DataFrame(x, columns=['x', 'y']))

        # Record the updated data
        data_loc = pd.concat(denoised_chunks, ignore_index=True)
        data_loc.columns = ['x', 'y']
        # if convert_to_grid:
        #   data_update = coordinate_to_count(data_loc, spatial_data)['count'].astype(np.int64)
        # else:
        data_update = data_loc.astype(np.int64)

        return data_update