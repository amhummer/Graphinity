import pandas as pd

#def format_ds(size):
#
#    for file in ["train","val"]:
#        df = pd.read_csv(f"/data/localhost/not-backed-up/nvme00/hummer/synth_aff_data/ddg_pred/dataset/varying_amounts/sub_{size}/synth_ddg-subset_{size}-{file}.csv")
#    
#        df['complex'] = df['complex'].str.split("_", expand=True)[0] + "_" + df['complex'].str.split("_", expand=True)[3]
#        df['pdb'] = df['complex'].str[:4]
#        df['ab_chain'] = df['chain_prot1'].copy()
#        df['ag_chain'] = df['chain_prot2'].copy()
#        
#        df[['pdb', 'complex', 'labels', 'chain_prot1', 'chain_prot2', 'ab_chain', 'ag_chain']].to_csv(f"subset_{size}/Synthetic_ddG-varying_dataset_size-{size}-{file}.csv", index=False)
#
#for size in [645,1000,5000,10000,50000,100000,500000]:
#    format_ds(size)
#
#
#test = pd.read_csv("/data/nagagpu03/not-backed-up/nvme00/hummer/synth_aff_data/ddg_pred/dataset/synth_ddg-942723-test.csv")
#test['complex'] = test['complex'].str.split("_", expand=True)[0] + "_" + test['complex'].str.split("_", expand=True)[3]
#test['pdb'] = test['complex'].str[:4]
#test['ab_chain'] = test['chain_prot1'].copy()
#test['ag_chain'] = test['chain_prot2'].copy()
#
#test[['pdb', 'complex', 'labels', 'chain_prot1', 'chain_prot2', 'ab_chain', 'ag_chain']].to_csv(f"Synthetic_ddG-varying_dataset_size-test.csv", index=False)

for file in ["train","val","test"]:
    df = pd.read_csv(f"/data/nagagpu03/not-backed-up/nvme00/hummer/synth_aff_data/ddg_pred/dataset/synth_ddg-942723-{file}.csv")
    df['complex'] = df['complex'].str.split("_", expand=True)[0] + "_" + df['complex'].str.split("_", expand=True)[3]
    df['pdb'] = df['complex'].str[:4]
    df['ab_chain'] = df['chain_prot1'].copy()
    df['ag_chain'] = df['chain_prot2'].copy()

    df[['pdb', 'complex', 'labels', 'chain_prot1', 'chain_prot2', 'ab_chain', 'ag_chain']].to_csv(f"full_942723/Synthetic_ddG-full_942723-{file}.csv", index=False)
