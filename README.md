# Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ∆∆G Prediction
## Graphinity: Equivariant Graph Neural Network Architecture for Predicting Change in Antibody-Antigen Binding Affinity
<p align="center">
<img src="Graphinity_architecture_ddg_horizontal.png" alt="Graphinity architecture for ∆∆G prediction" width="90%">
</p>

Code to accompany the paper titled: "Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ∆∆G Prediction"

Equivariant graph neural network (EGNN) code developed by Constantin Schneider and Alissa Hummer.


## Abstract
Antibody-antigen binding affinity lies at the heart of therapeutic antibody development: efficacy is guided by specific binding and control of affinity. Here we present Graphinity, an equivariant graph neural network architecture built directly from antibody-antigen structures that achieves state-of-the-art performance on experimental ∆∆G prediction. However, our model, like previous methods, appears to be overtraining on the few hundred experimental data points available. To investigate the amount and type of data required to generalizably predict ∆∆G, we built synthetic datasets of nearly 1 million FoldX-generated and >20,000 Rosetta Flex ddG-generated ∆∆G values. Our results indicate there is currently insufficient experimental data to accurately and robustly predict ∆∆G, with orders of magnitude more likely needed. Dataset size is not the only consideration – our tests demonstrate the importance of diversity. We also show that Graphinity can learn the distributions of experimental data, as opposed to synthetic data, using a set of >36,000 Trastuzumab variants.


## Synthetic ∆∆G Datasets
We generated synthetic ∆∆G datasets by mutating the interfaces of structurally-resolved complexes from SAbDab (Dunbar et al., 2014; Schneider et al., 2021) using FoldX (942,723 mutations; Schymkowitz et al., 2005) and Rosetta Flex ddG (20,829 mutations; Barlow et al., 2018). For more detail, please see the paper.

<p align="center">
<img src="Synthetic_ddG_dataset_generation.png" alt="Synthetic ∆∆G dataset generation" width="50%">
</p>


The PDBs can be downloaded from: https://opig.stats.ox.ac.uk/data/downloads/affinity_dataset/
  - FoldX (942,723 mutations):
    - WT: synthetic_foldx_ddg_wt_pdbs.tar.gz (303 MB compressed; 2.6 GB uncompressed)  
    - Mutant: synthetic_foldx_ddg_mutated_pdbs.tar.gz (195 GB compressed; 768 GB uncompressed)  
  - Flex ddG (20,829 mutations):
    - WT: synthetic_flexddg_ddg_wt_pdbs.tar.gz (7.8 GB compressed; 37 GB uncompressed)
    - Mutant: synthetic_flexddg_ddg_mutated_pdbs.tar.gz (7.8 GB compressed; 37 GB uncompressed)


## Requirements
The requirements to run the EGNN model code are included in the graphinity\_env\_cuda102.yaml file. A conda environment can be created from this file with
```
conda env create -f graphinity_env_cuda102.yaml
```

Installing the environment from the yaml file can take up to several hours.

If errors are encountered with the PyTorch Geometric installation, we recommend uninstalling and reinstalling the following packages (in the following order, with the versions specified):

```
pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster torch-spline-conv

pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
pip install torch-geometric==1.6.3 --no-cache-dir
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
```

An alternative to installing from the yaml file, the environment can also be created by running the commands below. This approach is faster, expected to take on the order of 10-30 minutes, depending on the machine.
```
conda create --name graphinity_env_cuda102 python=3.7.10
conda activate graphinity_env_cuda102

conda install pytorch==1.8.0 cudatoolkit=10.2 -c pytorch

pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
pip install torch-geometric==1.6.3 --no-cache-dir
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.8.0+cu102.html --no-cache-dir

conda install conda-forge::openbabel
pip install biopython
pip install numpy
pip install pandas
pip install pyarrow
pip install pyyaml
pip install biopandas
pip install pytorch-lightning==1.2.10
pip install tqdm
pip install wandb
```

Graphinity can also be run without a GPU. The environment can be created without CUDA as follows:
```
conda create --name graphinity_env_no_cuda python=3.7.10
conda activate graphinity_env_no_cuda

conda install pytorch==1.8.0 -c pytorch

pip install torch-scatter==2.0.8 --no-cache-dir
pip install torch-sparse==0.6.12 --no-cache-dir
pip install torch-geometric==1.6.3 --no-cache-dir
pip install torch-cluster==1.5.9 --no-cache-dir
pip install torch-spline-conv==1.2.1 --no-cache-dir

conda install conda-forge::openbabel
pip install biopython
pip install numpy
pip install pandas
pip install pyarrow
pip install pyyaml
pip install biopandas
pip install pytorch-lightning==1.2.10
pip install tqdm
pip install wandb
```

Graphinity has been tested on Linux and Mac. The results in the paper are from Graphinity trained and tested on Linux with 1 GPU (NVIDIA RTX 6000) and 4 CPUs.


## Citation

```
@article{Hummer2023,
	title = {Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ∆∆G Prediction},
	author = {Alissa M. Hummer and Constantin Schneider and Lewis Chinery and Charlotte M. Deane},
	journal = {bioRxiv},
	doi = {10.1101/2023.05.17.541222},
	URL = {https://www.biorxiv.org/content/early/2023/05/19/2023.05.17.541222},
	year = {2023},
}
```
