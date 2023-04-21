import pandas as pd

def format_ds(cutoff):

    if cutoff == "none": df = pd.read_csv("/data/localhost/not-backed-up/nvme00/hummer/synth_aff_data/ddg_pred/dataset/seqid_cutoffs/random/synth_ddg-942723-random-10foldcv.csv")
    else: df = pd.read_csv(f"/data/localhost/not-backed-up/nvme00/hummer/synth_aff_data/ddg_pred/dataset/seqid_cutoffs/cutoff_{cutoff}/synth_ddg-942723-cdr_seqid_cutoff_{cutoff}-10foldcv.csv")

    df['complex'] = df['complex'].str.split("_", expand=True)[0] + "_" + df['complex'].str.split("_", expand=True)[3]
    df['pdb'] = df['complex'].str[:4]
    df['ab_chain'] = df['chain_prot1'].copy()
    df['ag_chain'] = df['chain_prot2'].copy()

    df['pdb_wt'] = "synthetic_ddg_wt_pdbs/" + df['pdb_wt'].str.split("/", expand=True)[10]
    df['pdb_mut'] = "synthetic_ddg_mutated_pdbs/" + df['pdb_mut'].str.split("/", expand=True)[10]

    df[['pdb', 'complex', 'chain_prot1', 'chain_prot2', 'labels', 'ab_chain', 'ag_chain']].to_csv(f"Synthetic_ddG_942723-cutoff_{cutoff}-10foldcv.csv.csv", index=False)

for cutoff in ["none",100,90,70]:
    format_ds(cutoff)
