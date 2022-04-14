import pandas as pd
import numpy as np
from scipy import stats 

def add_infinity_boundaries(bins, left=False, right=True):
    # Silly hack to allow infinity to be included in bins
    new_bins = []
    
    if left:
        new_bins.append(-np.inf)

    for bin in bins.values.categories:
        new_bins.append(bin.left)

    if right:
        new_bins.append(np.inf)

    return new_bins

def numeric_get_optimal_bins(df, col, bins_max=20):
    bin_size_threshold = int(len(df) * 0.05)

    for bins_num in range(bins_max, 1, -1):
        binned_col = pd.qcut(df[col], bins_num, duplicates='drop')
        
        WoE_df = calculate_WoE(binned_col, df["Label"])     

        r, _ = stats.spearmanr(WoE_df[col], WoE_df["WoE"])
        bin_min_size = WoE_df[["0","1"]].sum(axis=1).min() 

        if (
                abs(r)==1 and                       # check if WoE for bins are monotonic
                bin_min_size > bin_size_threshold   # check if bin size is greater than 5%
        ):
            return WoE_df[col]

def calculate_WoE(feature, label):
    # Count occurences of events and non-events per category of given feature
    df = pd.crosstab(feature, label)

    # Adjusting table format
    df = df.reset_index().rename_axis(None, axis=1) 
    df.columns = df.columns.map(str)

    df["Distribution 0"] = df["0"] / df["0"].sum()
    df["Distribution 1"] = df["1"] / df["1"].sum()

    # Replace 0 with very small probability to handle edge case 
    # where a large bin contains only 
    cols = ["Distribution 0", "Distribution 1"]
    df[cols] = df[cols].replace(0, 0.00001)

    df['WoE'] = np.log(df["Distribution 1"] / df["Distribution 0"])

    return df

def calculate_IV(feature, label):
    df = calculate_WoE(feature, label)
    df['IV'] = (df['Distribution 1'] - df['Distribution 0']) * df['WoE']

    return df

def get_all_IVs(df, cols):
    iv_dict = {}

    for col in cols:
        iv_df = calculate_IV(df[col], df['Label'])
        iv_dict[col] = iv_df['IV'].sum()
    df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV'])
    df = df.sort_values(by='IV', ascending=False)

    return df

def get_WoE_encodings(df, cols):
    WoE_encodings = {}
    for col in cols:
        
        woe_df = calculate_WoE(df[col], df["Label"])
        woe_df = woe_df.set_index(col)

        WoE_encodings[col] = woe_df['WoE'].to_dict()
    
    return WoE_encodings

def encode_WoE(df, WoE_encodings, drop_other=False):

    for col in WoE_encodings.keys():
        df[col + "WoE"] = df[col].map(WoE_encodings[col]).astype(float)

    if drop_other:
        cols = [col+"WoE" for col in WoE_encodings.keys()]
        cols.append("Label")
        df = df[cols]

    return df