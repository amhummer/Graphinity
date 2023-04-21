# Trastuzumab Variants

Dataset from paper: Mason et al., 2021 - https://www.nature.com/articles/s41551-021-00699-9

Deep mutational scanning-guided mutagenesis of Trastuzumab CDRH3 - mutations made to 10 positions (WT = WGGDGFYAMD). Binary binding values for 36,391 variants - classified based on FACS. Binder: HER2.

## Files

- random/trastuzumab_variants-random-{train/val/test}.csv (random train-val-test split)
- split_by_clonotype_seqid_{90/70}/trastuzumab_variants-split_by_clonotype_seqid_{90/70}-{train/val/test}.csv (train-val-test split by clonotype and sequence identity)

## Columns CSV

- cdrh3_seq: sequence of the CDRH3
- labels: non-binding (0), binding (1)
- ab_chains: antibody chains
- ag_chains: antigen chains

## Notes

- Some variants were classified as both binding and non-binding. In this dataset, these were set to be binding (only) as is done in the Mason paper.
- WT Trastuzumab sequence:
    - VH: EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS
    - VL: DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK
- Prototypical PDB structure: 1N8Z (antibody chains: AB, antigen chain: C)
