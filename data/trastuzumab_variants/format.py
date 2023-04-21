import pandas as pd

for file in ["train","val","test"]:
    # random
    df = pd.read_csv(f"/data/localhost/not-backed-up/nvme00/hummer/greiff_data/reddy_data/dataset/random/reddy_variants-random-{file}.csv")
    df[["aa_cdr3_seq","labels","ab_chains","ag_chains"]].rename(columns={"aa_cdr3_seq":"cdrh3_seq"}).to_csv(f"random/trastuzumab_variants-random-{file}.csv", index=False)

    # clonotype, seqid split
    for cutoff in [90,70]:
        df = pd.read_csv(f"/data/localhost/not-backed-up/nvme00/hummer/greiff_data/reddy_data/dataset/split_by_clonotype/Reddy_train_val_test_split_by_clonotype_{cutoff}pc_VJ_70_15_15/reddy_variants-split_by_clonotype_seqid_{cutoff}-{file}.csv")
        df[["aa_cdr3_seq","labels","ab_chains","ag_chains"]].rename(columns={"aa_cdr3_seq":"cdrh3_seq"}).to_csv(f"split_by_clonotype_seqid_{cutoff}/trastuzumab_variants-split_by_clonotype_seqid_{cutoff}-{file}.csv", index=False)
