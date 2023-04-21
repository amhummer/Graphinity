# Synthetic\_ddG\_942723

Dataset of 942723 single-point antibody-antigen mutations generated using FoldX (Schymkowitz et al., 2005). Structurally-resolved antibody-antigen complexes were obtained from SAbDab (Dunbar et al., 2014) and interface residues exhaustively mutated using FoldX. For more details on the method, please see the manuscript (Figure 1b, Methods section).

## Files

- Synthetic\_ddG\_942723.csv
- cdr\_seqid\_cutoffs/Synthetic\_ddG\_942723/Synthetic\_ddG\_942723-cutoff\_{none/100/90/70}-10foldcv.csv
- varying\_dataset\_size/subset\_{size}/Synthetic\_ddG-varying\_dataset\_size-{size}-{train/val}.csv, varying\_dataset\_size/Synthetic\_ddG-varying\_dataset\_size-test.csv
- varying\_dataset\_diversity/{diversity\_type}/{min/max}/Synthetic\_ddG-varying\_dataset\_diversity-{diversity\_type}-{min/max}-{train/val}.csv, varying\_dataset\_diversity/Synthetic\_ddG-varying\_dataset\_diversity-test.csv
- noise\_robustness/shuffling/sh\_{percent}pct/synth\_ddg-shuffled\_{percent}\_pct-train.csv
- noise\_robustness/gaussian\_noise/scale_{scale}/Synthetic_ddG-gaussian_noise-scale_{scale}-train.csv
- noise\_robustness/Synthetic\_ddG-noise\_robustness-test.csv

## Columns CSV

- pdb: pdb id
- complex: in the format {pdb}\_{wt\_aa}\_{chain\_id}\_{residue\_number}\_{mut\_aa}
- chain\_prot1: chain id(s) of one binding partner
- chain\_prot2: chain id(s) of the other binding partner
- labels: ddG label (kcal/mol)
- ab\_chain: antigen chain id(s); not used by the model code, provided for information
- ag\_chain: antigen chain id(s); not used by the model code, provided for information

also in cdr\_seqid\_cutoffs directory files
- fold\_id: id for the fold a mutation is in (split based on the length-matched CDR sequence identity cutoff provided in the file name)

## PDB files
We have made the PDB files available at the following links.
  - WT: https://opig.stats.ox.ac.uk/data/downloads/synthetic\_ddg\_wt\_pdbs.tar.gz (303 MB compressed; 2.6 GB uncompressed)
  - Mutant: https://opig.stats.ox.ac.uk/data/downloads/synthetic\_ddg\_mutated\_pdbs.tar.gz (195 GB compressed; 768 GB uncompressed)

The WT pdb files are named [pdb].pdb (eg 3w2d.pdb). The mutant PDB files are named [pdb]\_[antibody\_chains]\_[antigen\_chains]\_[mutation\_id].pdb (eg 3w2d\_HL\_A\_EL1A.pdb)
