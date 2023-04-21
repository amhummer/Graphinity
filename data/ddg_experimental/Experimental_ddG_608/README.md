# Experimental_ddG_608

Dataset of 608 single-point mutations and corresponding ddG values for structurally-resolved antibody-antigen complexes, filtered from the SKEMPI 2.0 database (Jankauskaite et al., 2019).

## Files

- Experimental_ddG_608.csv
- cdr_seqid_cutoffs/: files where the train, validation and test datasets are separated by a certain length-matched CDR sequence identity cutoff
	- Experimental_ddG_608_-Reverse_Mutations/Experimental_ddG_608_-Reverse_Mutations-cutoff_{none/100/90/70}-10foldcv.csv
	- Experimental_ddG_608_+Reverse_Mutations/Experimental_ddG_608_+Reverse_Mutations-cutoff_{none/100/90/70}-10foldcv.csv: hypothetical reverse mutations included in the dataset

## Columns CSV

- pdb: pdb id
- complex: in the format {pdb}\_{wt_aa}\_{chain_id}\_{residue_number}\_{mut_aa}
- chain_prot1: chain id(s) of one binding partner
- chain_prot2: chain id(s) of the other binding partner
- labels: ddG label (kcal/mol)
- ab_chain: antigen chain id(s); not used by the model code, provided for information
- ag_chain: antigen chain id(s); not used by the model code, provided for information

also in cdr_seqid_cutoffs directory files
- fold_id: id for the fold a mutation is in (split based on the length-matched CDR sequence identity cutoff provided in the file name)
