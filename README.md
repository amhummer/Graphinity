# Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ∆∆G Prediction
## Graphinity: Equivariant Graph Neural Network Architecture for Predicting Change in Antibody-Antigen Binding Affinity
<p align="center">
<img src="Graphinity_architecture_ddg_horizontal.png" alt="Graphinity architecture for ∆∆G prediction" width="90%">
</p>

Code to accompany the paper titled: "Investigating the Volume and Diversity of Data Needed for Generalizable Antibody-Antigen ∆∆G Prediction"

Equivariant graph neural network (EGNN) code developed by Constantin Schneider and Alissa Hummer.

## Abstract
Antibody-antigen binding affinity lies at the heart of therapeutic antibody development: efficacy is guided by specific binding and control of affinity. Here we present Graphinity, an equivariant graph neural network architecture built directly from antibody-antigen structures that achieves state-of-the-art performance on experimental ∆∆G prediction. However, our model, like previous methods, appears to be overtraining on the few hundred experimental data points available. To test if we could overcome this problem, we built a synthetic dataset of nearly 1 million FoldX-generated ∆∆G values. Graphinity achieved Pearson’s correlations nearing 0.9 and was robust to train-test cutoffs and noise on this dataset. The synthetic dataset also allowed us to investigate the role of dataset size and diversity in model performance. Our results indicate there is currently insufficient experimental data to accurately and robustly predict ∆∆G, with orders of magnitude more likely needed. Dataset size is not the only consideration – our tests demonstrate the importance of diversity. We also confirm that Graphinity can be used for experimental binding prediction by applying it to a dataset of >36,000 Trastuzumab variants.

## Requirements
The requirements to run the EGNN model code are included in the graphinity\_env\_cuda102.yaml file. A conda environment can be created from this file with
```
conda env create -f graphinity_env_cuda102.yaml
```

## Synthetic FoldX ∆∆G Dataset
We generated a synthetic ∆∆G dataset consisting of 942,723 data points by exhaustively mutating the interfaces of structurally-resolved complexes from SAbDab (Dunbar et al., 2014; Schneider et al., 2021) using FoldX (Schymkowitz et al., 2005). For more detail, please see the paper.

<p align="center">
<img src="Synthetic_ddG_dataset_generation.png" alt="Synthetic ∆∆G dataset generation" width="50%">
</p>


The PDBs can be downloaded from: https://opig.stats.ox.ac.uk/data/downloads/affinity_dataset/
  - WT: synthetic_ddg_wt_pdbs.tar.gz (303 MB compressed; 2.6 GB uncompressed)  
  - Mutant: synthetic_ddg_mutated_pdbs.tar.gz (195 GB compressed; 768 GB uncompressed)  

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
