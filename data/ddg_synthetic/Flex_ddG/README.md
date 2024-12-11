# Synthetic\_FlexddG\_ddG\_20829

Dataset of 20829 single-point antibody-antigen mutations generated using Flex ddG (Barlow et al., 2018). Structurally-resolved antibody-antigen complexes were obtained from SAbDab (Dunbar et al., 2014) and interface residues exhaustively mutated using Flex ddG. For more details on the method, please see the manuscript (Figure 1b, Methods section).

## Files

- Synthetic\_FlexddG\_ddG\_20829.csv
- cdr\_seqid\_cutoffs/Synthetic\_FlexddG\_ddG\_20829/Synthetic\_FlexddG\_ddG\_20829-cutoff\_{none/100/90/70}-10foldcv.csv: the files were generated using the folds from the FoldX dataset (Synthetic\_FoldX\_ddG\_942723/cdr\_seqid\_cutoffs)
- varying\_dataset\_size/subset\_{size}/Synthetic\_FlexddG\_ddG-varying\_dataset\_size-{size}-{train/val}.csv, varying\_dataset\_size/Synthetic\_FlexddG\_ddG-varying\_dataset\_size-test.csv

## Columns CSV

- pdb: pdb id
- complex: in the format {pdb}\_{wt\_aa}\_{chain\_id}\_{residue\_number}\_{mut\_aa}
- labels: Flex ddG label (kcal/mol), ddG\_fa\_talaris2014-gam (used as the labels in this study)
- ab\_chain: antigen chain id(s); not used by the model code, provided for information
- ag\_chain: antigen chain id(s); not used by the model code, provided for information
- ddG\_fa\_talaris2014-gam: Flex ddG prediction using Talaris 2014 energy function with generalized additive model (GAM) reweighting (identical to the labels column; used as the labels in this study)
- ddG\_fa\_talaris2014: Flex ddG prediction using Talaris 2014 energy function without generalized additive model (GAM) reweighting
- wt\_dg\_fa\_talaris2014: Flex ddG dG value for the WT complex (kcal/mol) using Talaris 2014 energy function without generalized additive model (GAM) reweighting
- mut\_dg\_fa\_talaris2014: Flex ddG dG value for the WT complex (kcal/mol) using Talaris 2014 energy function without generalized additive model (GAM) reweighting

## PDB files
The PDBs can be downloaded from: https://opig.stats.ox.ac.uk/data/downloads/affinity_dataset/
  - WT: synthetic_flexddg_ddg_wt_pdbs.tar.gz (7.8 GB compressed; 37 GB uncompressed)
  - Mutant: synthetic_flexddg_ddg_mutated_pdbs.tar.gz (7.8 GB compressed; 37 GB uncompressed)

The PDB files are named [pdb]\_[antibody\_chains]\_[antigen\_chains]\_[mutation\_id]-[wt/mut].pdb (eg 3w2d\_HL\_A\_EL1A-mut.pdb)
