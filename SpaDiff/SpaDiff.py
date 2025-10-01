import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
# from sklearn import datasets
import pickle

# import functino for kscore
import sys
# sys.path.append('/content/drive/MyDrive/Colab Notebooks')
from kscore.estimators import *
from kscore.kernels import *
import kscore

from utils import *

import argparse
import imageio # for creating gif
from io import BytesIO
# from scipy import sparse, io # for read sparse matrix
from scipy.spatial import KDTree # for finding neighbor
from scipy.spatial import cKDTree # for finding neighbor
from scipy.stats import gaussian_kde # for estimating the variance of the distance distribution

from multiprocessing import cpu_count # for parallel processing
import multiprocessing
# multiprocessing.set_start_method('spawn')
from pathos.multiprocessing import ProcessingPool as Pool

from tqdm import tqdm # for process bar


class SpaDiff:

    def __init__(self, count, spatial, meta=None):
        """
        Initialize a SpaDiff object.

        Parameters:
        count: The count matrix, where the row represents spot and column represents gene.
        spatial: The spatial matrix, where the row represents spot and column represents x and y coordinates.
        meta: The meta data that records some extra information of each spot, e.g., boundary, annotation.

        Returns:
        None
        """
        count = count.loc[spatial.index]
        self.X = count
        spatial.columns=['x', 'y']
        self.S_org = spatial
        self.meta = meta


    def spatial_scale(self):
        """
        Scale the spatial coordinates so that each spot takes one unit. This step will convert the data from different platforms to the same scale.
        It will make the choice of parameters in the subsequent steps easier.

        Parameters:
        None

        Returns:
        None
        """
        data = self.S_org

        x_range = max(data['x']) - min(data['x'])
        y_range = max(data['y']) - min(data['y'])
        x_count = data['x'].unique().size
        y_count = data['y'].unique().size
        data['x'] = data['x'] / x_range * x_count
        data['y'] = data['y'] / y_range * y_count

        self.S = data


    def scatter_plot(self, if_update=False, gene_index=None, lam=1.0, kernel_width=80.0, subsample=True, fraction=0.2, sample_count=500, seed=1234, 
                     figure_size=(8,8), gradient_return=False, gradient_plot=True, boundary=False, background=True, size=100, x_limit=None, y_limit=None):
        """
        Plot the score function pattern.

        Paramters:
        gene_index: The gene index for the target gene. If None, then the count matrix should only contains only one column.
        lam: The parameter lambda for nu estimator.
        kernel_width: The width of the kernel in the gradient calculation.
        subsample: The indicator of whether to use subsample to accelerate calculation.
        fraction: The proportion of the subsample.
        sample_count: The exact number of subsamples to take.
        seed: Random seed.
        figure_size: A tuple recording the width and height of the figure.
        gradient: The indicator of whether plot the score function.
        boundary: The indicator of whether plot the bondary.
        background: The indicator of whether plot the background spot.
        size: The spot size in the figure.

        Returns:
        gradient: The calculated gradient.
        """
        # Transfer count vector to coordinate set
        if if_update:
            if gene_index is None:
                count = self.X_update.squeeze()
            else:
                count = self.X_update.iloc[:, gene_index]
        else:
            if gene_index is None:
                count = self.X.squeeze()
            else:
                count = self.X.iloc[:, gene_index]
            
        spatial = self.S
        count.index = spatial.index
        data_loc = count_to_coordinate(count, spatial)

        # plot the score field
        plt.figure(figsize=figure_size)
        if background:
            plt.scatter(spatial['x'], spatial['y'], s=size, c='whitesmoke')
        plt.scatter(data_loc['x'], data_loc['y'], s=size, c='steelblue', alpha=0.2)
        if gradient_plot:
            gradient = gradient_calculation(data_loc, lam, kernel_width, subsample, fraction, sample_count, seed)
            plt.quiver(data_loc.iloc[:,0], data_loc.iloc[:,1], gradient[:,0], gradient[:,1])
        if boundary:
            meta = self.meta
            meta_data = pd.concat([spatial, meta], axis=1)
            boundary_loc = meta_data[meta_data['boundary']==1][['x', 'y']]
            plt.scatter(boundary_loc['x'], boundary_loc['y'], s=size/2, c='red')
        if x_limit is not None:
            plt.xlim(x_limit)
        if y_limit is not None:
            plt.ylim(y_limit)
        plt.show()

        if gradient_return:
            return gradient


    # def variance_calculation_boundary(self, if_update=False, if_plot=True, bins=30):
    #     """
    #     Calculate variance of the distribution of the distances between the off-tissue spots and the boundary.

    #     Parameters:
    #     if_update: The indicator of whether using the updated data.
    #     if_plot: The indicator of whether plot the histgram of the distances.

    #     Returns:
    #     variance_kde: The calculated variance.
    #     """
    #     # Extract data
    #     if if_update:
    #         count = self.X_update
    #     else:
    #         count = self.X
    #         spatial = self.S
    #         meta = self.meta
    #         data = pd.concat([spatial, meta], axis=1)
    #         data['read_count'] = count.sum(axis=1)

    #     # Extract boundary and off-tissue spatial data
    #     boundary_loc = data[data['boundary']==1][['x', 'y']]
    #     data_background = data[data['tissue']==0]
    #     data_loc = count_to_coordinate(data_background['read_count'], data_background[['x', 'y']])

    #     # Calculate the min distance between off-tissue spots and boundary spots
    #     distances = calculate_min_distances(data_loc, boundary_loc)

    #     # Plot the histgram of the distances
    #     if if_plot:
    #         plt.hist(distances, density=True, bins=bins, color='blue', alpha=0.5)
    #         plt.title('Histogram of Distances')
    #         plt.xlabel('Distances')
    #         plt.ylabel('Frequency')
    #         plt.show()

    #     # Creating a Gaussian KDE instance
    #     kde = gaussian_kde(distances)

    #     # Estimate variance
    #     variance_kde = kde.covariance[0, 0]

    #     return ({'variance': variance_kde, 'distance': distances})


    # def variance_calculation_target(self, gene_index, if_update=False, if_plot=True, bins=30):
    #     """
    #     Calculate the variance of the distribution of the distances between all biomarker gene reads and the target domains.

    #     Parameters:
    #     biomarker: The name of the biomarker gene.
    #     if_update: The indicator of whether using the updated data.
    #     if_plot: The indicator of whether plot the histgram of the distances.

    #     Returns:
    #     variance_kde: The calculated variance.
    #     """
    #     # Extract data
    #     if if_update:
    #         count = pd.DataFrame(self.X_update).iloc[:, gene_index]
    #     else:
    #         count = pd.DataFrame(self.X).iloc[:, gene_index]
    #         spatial = self.S
    #         meta = self.meta['target']
    #         data = pd.concat([spatial, count, meta], axis=1)
    #         data.columns = ['x','y','count','target']

    #     # Extract boundary and off-tissue spatial data
    #     target_loc = data[data['target']==1][['x', 'y']]
    #     data_loc = count_to_coordinate(data['count'], data[['x', 'y']])

    #     # Calculate the min distance between off-tissue spots and boundary spots
    #     distances = calculate_min_distances(data_loc, target_loc)

    #     # Plot the histgram of the distances
    #     if if_plot:
    #         plt.hist(distances, density=True, bins=bins, color='blue', alpha=0.5)
    #         plt.title('Histogram of Distances')
    #         plt.xlabel('Distances')
    #         plt.ylabel('Frequency')
    #         plt.show()

    #     # Creating a Gaussian KDE instance
    #     kde = gaussian_kde(distances)

    #     # Estimate variance
    #     variance_kde = kde.covariance[0, 0]

    #     return ({'variance': variance_kde, 'distance': distances})


    def denoise_single(self, gene_index=None, step=20, step_size=1, sigma=0.01, lam=1.0, kernel_width=80.0, subsample=True,
                       fraction=0.2, sample_count=500, gradient_scale=None, max_gradient=5, threshold=10000, if_output=False, 
                       generate_gif=False, save_path='../gif/process.gif', duration=1, figure_size=(8,8), size=100, seed=1234):
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

        Returns:
        data_update: The updated count matrix.
        """
        # Save the parameter
        self.step = step
        self.step_size = step_size
        self.sigma = sigma
        self.lam = lam
        self.kernel_width = kernel_width
        self.subsample = subsample
        self.fraction = fraction
        self.sample_count = sample_count
        self.gradient_scale = gradient_scale
        self.max_gradient = max_gradient
        self.seed = seed

        # Extract the target data according to gene_index
        if gene_index is None:
            data = pd.DataFrame(self.X)
        elif isinstance(gene_index, int):
            data = self.X.iloc[:, gene_index]
        elif isinstance(gene_index, str):
            data = self.X.loc[:, gene_index]
        else:
            raise ValueError("Invalid gene index")

        data_update = data.copy()

        # Transfer count vector to coordinate set
        count_vector = data
        data_loc = count_to_coordinate(count_vector, self.S)
        spatial = self.S






        # chunks = split_random_balanced_chunks(data_loc, max_chunk_size=threshold, seed=seed)
        # denoised_chunks = []

        # for chunk_df in chunks:
        #     x = np.array(chunk_df, dtype=np.float32)

        #   # Optional: Gradient scale per chunk (auto or passed)
        #     for update in range(step):
        #         data_tmp = pd.DataFrame(x, columns=['x', 'y'])

        #         gradient = gradient_calculation(
        #             data_tmp, lam=lam, kernel_width=kernel_width,
        #             subsample=subsample, fraction=fraction,
        #             sample_count=sample_count, seed=seed
        #         )

        #         if gradient_scale is None:
        #             g_scale = gradient_scaler(gradient, max_gradient=max_gradient)
        #         else:
        #             g_scale = gradient_scale

        #         rand = tf.random.normal((x.shape[0], 2), mean=0.0, stddev=1)
        #         x = x + step_size * (sigma*gradient*g_scale + math.sqrt(sigma)*rand)

        #     denoised_chunks.append(pd.DataFrame(x, columns=['x', 'y']))

        # data_loc = pd.concat(denoised_chunks)





        # Calculate the gradient for the orginal data
        gradient = gradient_calculation(data_loc, lam, kernel_width, subsample, fraction, sample_count, seed)

        # Calculate the gradient scaler
        if gradient_scale is None:
            g_scale = gradient_scaler(gradient, max_gradient)

        # Denoise process
        x = np.array(data_loc, np.float32)
        
        # Initialize gif
        if generate_gif:
            images = []

        for update in range(step):

            # Calculate gradient
            data_tmp = pd.DataFrame(x, columns=['x', 'y'])
            gradient = gradient_calculation(data_tmp, lam, kernel_width, subsample, fraction, sample_count, seed)

            # Update the status
            rand = tf.random.normal((x.shape[0], 2), mean=0.0, stddev=1)
            x = x + step_size * (sigma*gradient*g_scale + math.sqrt(sigma)*rand)
            
            # Generate GIF
            if generate_gif:
                # Plot the distribution of all reads at the current step
                plt.figure(figsize=figure_size)
                plt.scatter(spatial['x'], spatial['y'], s=size, c='whitesmoke')
                plt.scatter(x[:,0], x[:,1], s=size, alpha=0.2, c='steelblue')
                plt.title(f'Time = {update:.2f}')

                # Create a BytesIO buffer to save the plot
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                plt.close()  # Close the figure to prevent it from displaying in the notebook
                buf.seek(0)  # Go to the beginning of the BytesIO buffer
                images.append(imageio.imread(buf))
                buf.close()  # Close the buffer

        # Record the updated data
        data_loc = pd.DataFrame(x)
        data_loc.columns = ['x', 'y']
        count_vector = coordinate_to_count(data_loc, self.S)['count']
        data_update = count_vector

        self.X_update = data_update
        
        # Create a GIF from the images stored in memory
        if generate_gif:
            filename = save_path
            imageio.mimsave(filename, images, duration=duration)

        if if_output:
            return data_update
    
    

    def denoise(self, step=20, step_size=1, sigma=0.01, lam=1.0, kernel_width=10.0, subsample=True, 
                fraction=0.2, sample_count=500, gradient_scale=None, max_gradient=5, if_output=False, seed=1234, if_parallel=True,
                threshold=10000, convert_to_grid=True):
        """
        Denoise all genes in parallel.

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
        # Save the parameter
        self.step = step
        self.step_size = step_size
        self.sigma = sigma
        self.lam = lam
        self.kernel_width = kernel_width
        self.subsample = subsample
        self.fraction = fraction
        self.sample_count = sample_count
        self.gradient_scale = gradient_scale
        self.max_gradient = max_gradient
        self.seed = seed
        self.threshold = threshold

        params = (step, step_size, sigma, lam, kernel_width, subsample, fraction, sample_count, gradient_scale, max_gradient, seed)

        # Extract the data
        count_data = pd.DataFrame(self.X) 
        spatial_data = pd.DataFrame(self.S) 
        if convert_to_grid:
            update_data = count_data.copy()
        else:
            update_data = []  # Will store one DataFrame of coordinates per gene

        # update_data = count_data.copy()
        
        # Prepare tuples for parallel processing
        data_list = [(count_data.iloc[:, i], spatial_data, params, threshold) for i in range(count_data.shape[1])]
        
        try:
            pickle.dumps(data_list[0])
            pickle.dumps(denoise_single_gene)
            print("Pickling successful.")
        except Exception as e:
            print(f"Pickling error: {e}")
        
        if if_parallel:
            
            print("Starting parallel processing...")
            # Parallel processing with progress bar
            with Pool(processes=20) as pool:
                print(f"Created pool with {20} processes")
                # results = list(tqdm(pool.imap(denoise_single_gene, data_list), total=len(data_list), desc="Denoising genes"))
                # results = list(tqdm(pool.imap(safe_denoise, data_list), total=len(data_list), desc="Denoising genes"))
                results = list(tqdm(pool.imap(denoise_single_gene, data_list), total=len(data_list)))
                # results = denoise_single_gene(data_list[0])

            # Collect results and update DataFrame
            for i, gene_coords in enumerate(results):
                if convert_to_grid:
                    gene_counts = coordinate_to_count(gene_coords, spatial_data)['count']
                    update_data.iloc[:, i] = gene_counts.astype(np.int64).values
                else:
                    update_data.append(gene_coords)  # Just append coords to the list

            self.X_update = update_data
        
        else:
            for i in range(count_data.shape[1]):
                updated_gene_data = denoise_single_gene(data_list[i])

                if convert_to_grid:
                    gene_counts = coordinate_to_count(updated_gene_data, spatial_data)['count']
                    update_data.iloc[:, i] = gene_counts.astype(np.int64).values
                else:
                    update_data.append(updated_gene_data)  # if you want to store coordinates instead
                
            self.X_update = update_data

        if if_output:
            return update_data
            

   

