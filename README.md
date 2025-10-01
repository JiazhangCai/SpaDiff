# SpaDiff: Denoising for Sequence-based Spatial Transcriptomics via Diffusion Process

## 1. Introduction 
Spatial transcriptomics enables transcriptome-scale analysis with spatial resolution but suffers from spot-swapping, 
where RNA molecules drift from their true locations, introducing noise and reducing spatial specificity. We introduce SpaDiff, 
a denoising method that treats spot-swapping as a diffusion process. 
SpaDiff simulates the displacement of RNA molecules and reverses it to restore their original spatial distribution while preserving molecular counts. 
Evaluations on simulated and real data demonstrate that SpaDiff enhances the spatial specificity of gene expression, improves data integrity, 
and supports more accurate downstream analyses, such as clustering and spatial domain identification.

<img width="468" height="382" alt="image" src="https://github.com/user-attachments/assets/b5213986-0adf-4b89-9712-761ee988c169" />


## 2. Environment 
Python=3.12.4 
### Required packages: 
The detailed requirements can be found [here](https://github.com/JiazhangCai/SpaDiff/blob/main/requirements.txt).

## 3.Tutorial 
The [tutorial](https://github.com/JiazhangCai/SpaDiff/blob/main/tutorial.ipynb) provides a step-by-step guide to using SpaDiff on the DLPFC spatial transcriptomics dataset. It covers data loading and preprocessing, optional gene filtering, constructing the SpaDiff object and scaling coordinates, visualizing the score function for parameter selection, running denoising on single genes and the full dataset, evaluating performance through visualization, and finally exporting the denoised data in standard Visium format for downstream analysis.

## 4. Data used in the paper 

### 4.1 Simulation data

The code for generating simulation data is provided. [Here](https://github.com/JiazhangCai/SpaDiff/blob/main/SpaDiff/simulation.py) is the code 
for generating simulation data using the [Xenium data](https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain) of the mouse brain.

### 4.2 Real data

The DLPFC data used in the paper can be found [here](http://research.libd.org/spatialLIBD/).     
The 10X Genomics Visium image data of the mouse-human chimeric data used in the paper can be found [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178221).

 
