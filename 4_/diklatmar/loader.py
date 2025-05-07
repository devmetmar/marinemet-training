import numpy as np
import datetime
import os
import pandas as pd

class loader:
    global MSG_INFO, MSG_WARN, MSG_ERR
    MSG_INFO = 1
    MSG_WARN = 2
    MSG_ERR = 3

    def __init__(self, data_dir=None):
        try:
            self.data_dir = data_dir
        except Exception as e:
            self.logging(MSG_ERR, f"Initialization failed: {e}")
            raise            

    def logging(self, messType, messText):
        dtNow = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S :")
        code = {MSG_INFO: "I", MSG_WARN: "W", MSG_ERR: "E"}.get(messType, "E")
        print(f"{dtNow}\t{code}\t{messText}")
        
    def meta(self, filename=None):
        if not filename:
            self.logging(MSG_ERR, "No file specified to load")
            return None
        
        fpath_meta = os.path.join(self.data_dir, filename)

        if not os.path.exists(fpath_meta):
            self.logging(MSG_ERR, f"File not found: {fpath_meta}")
            return None

        try:
            df = pd.read_csv(fpath_meta, index_col=["id"]).iloc[:, :3]
            self.logging(MSG_INFO, f"Loaded metadata from {filename}")
            return df
        except Exception as e:
            self.logging(MSG_ERR, f"Failed to read file: {e}")
            return None
        
    def maws_data(self, id_aws=None, year=None, month=None, save_data=False, sv_dir=None):
        if id_aws is None or year is None or month is None:
            self.logging(MSG_ERR, "Check your input parameter. id_aws, year, and month must exist")
            return None

        id_aws = str(id_aws)
        year_str = f"{year:04d}"
        month_str = f"{month:02d}"

        def find_subdir(base_path, keyword, level_name):
            matches = [d for d in os.listdir(base_path) if keyword in d]
            if matches:
                self.logging(MSG_INFO, f"Found {level_name}: {keyword}")
                return os.path.join(base_path, matches[0])
            else:
                self.logging(MSG_ERR, f"{level_name.capitalize()} not found: {keyword}")
                return None


        path2id = find_subdir(self.data_dir, id_aws, "AWS id")
        if not path2id: return None

        path2year = find_subdir(path2id, year_str, "year")
        if not path2year: return None

        path2month = find_subdir(path2year, month_str, "month")
        if not path2month: return None
        # self.logging(MSG_INFO, f"Path to directory: {path2month}")
        
        parquet_files = sorted(
            [os.path.join(path2month, f) for f in os.listdir(path2month) if f.endswith('.parquet')]
        )

        if not parquet_files:
            self.logging(MSG_ERR, f"No .parquet files found in {path2month}")
            return None

        df_month = pd.concat([pd.read_parquet(fl) for fl in parquet_files], ignore_index=False)
        self.logging(MSG_INFO, f"Loaded {len(parquet_files)} files with {len(df_month)} records.")
        
        if save_data:
            dir2sv = os.makedirs(sv_dir, exist_ok=True)
            pth2sv = f"{sv_dir}/{id_aws}_{year_str}_{month_str}.csv"
            df_month.to_csv(pth2sv)
            self.logging(MSG_INFO, f"Saving file to: {pth2sv}")

        
        return df_month
            
            

        