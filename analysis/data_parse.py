import pandas as pd

names = ["data/MSNA feb 18 bsl.txt", "data/MSNA feb 18 slow breathing 1.txt", "data/MSNA feb 18 slow breathing 2.txt", "data/MSNA Feb 19 bsl.txt", "data/MSNA Feb 19 slow breathing.txt", "data/MSNA Feb 19 slow breathing 2.txt", "data/TGIF4 baseline.txt", "data/TGIF38 bsl.txt", "data/TGIF40 bsl.txt", "data/TGIF45 bsl.txt","data/TGIF90 baseline .txt", "data/vagus pilot comment 7 bsl.txt", "data/vagus pilot comment 9 slow breathing.txt", "data/vagus pilot comment 47 slow breathing.txt"]
newnames = ["data_parq/MSNA feb 18 bsl.parquet", "data_parq/MSNA feb 18 slow breathing 1.parquet", "data_parq/MSNA feb 18 slow breathing 2.parquet", "data_parq/MSNA Feb 19 bsl.parquet", "data_parq/MSNA Feb 19 slow breathing.parquet", "data_parq/MSNA Feb 19 slow breathing 2.parquet", "data_parq/TGIF4 baseline.parquet", "data_parq/TGIF38 bsl.parquet", "data_parq/TGIF40 bsl.parquet", "data_parq/TGIF45 bsl.parquet","data_parq/TGIF90 baseline .parquet", "data_parq/vagus pilot comment 7 bsl.parquet", "data_parq/vagus pilot comment 9 slow breathing.parquet", "data_parq/vagus pilot comment 47 slow breathing.parquet"]
for i in range(len(names)):
    fname = "../" + names[i]

    def load_data_old_4col(): 
        df = pd.read_csv(fname, sep=r"\s+", header=None)
        df.columns = ["time", "breath", "ecg", "vagus"]
        return df

    def load_data_msna_5col():  
        rows = []
        with open(fname, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # changed == to >=
                    try:
                        [float(x) for x in parts[:5]]  # validate only first 5
                        rows.append(parts[:5])           # take only first 5
                    except ValueError:
                        pass
        df = pd.DataFrame(rows, columns=["time", "resp", "ecg", "bp", "raw_arm"]).astype(float)
        return df

    def load_data_msna_4col():  
        rows = []
        with open(fname, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        [float(x) for x in parts]
                        rows.append(parts)
                    except ValueError:
                        pass
        df = pd.DataFrame(rows, columns=["time", "ecg", "raw_arm", "resp"]).astype(float)
        return df


    def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a NEW df with consistent column names:
        time, ecg, resp, nerve, (optional) bp
        Works with both:
        old: time breath ecg vagus
        msna: time ecg bp resp raw_arm
        """
        cols = set(df.columns)

        # vagus format
        if {"time", "breath", "ecg", "vagus"}.issubset(cols):
            out = df.rename(columns={"breath": "resp", "vagus": "nerve"}).copy()
            return out

        # msna format
        if {"time", "ecg", "raw_arm", "resp"}.issubset(cols):
            out = df.rename(columns={"raw_arm": "nerve"}).copy()
            return out

        raise ValueError(f"Unknown data format. Columns found: {sorted(df.columns)}")

    def load_any():
        """
        Tries old format first (fast). If it fails, falls back to MSNA-style loader.
        """
        try:
            df = load_data_old_4col()
            df = canonicalize(df)
            df = df.dropna().reset_index(drop=True)  # remove any NaN rows
            return df
        except Exception:
            df = load_data_msna_5col()
            df = canonicalize(df)
            df = df.dropna().reset_index(drop=True)  # remove any NaN rows
            return df

    df = load_any()
    df.to_parquet("../" + newnames[i], index=False)