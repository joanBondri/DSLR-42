#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.width', 1000)

def print_wide_dataframe(df, cols_per_block=5):
    """Affiche un DataFrame large par blocs de colonnes"""
    total_cols = len(df.columns)
    if (total_cols <= cols_per_block):
        print(df)
        return
    for i in range(0, total_cols, cols_per_block):
        end = min(i + cols_per_block, total_cols)
        print(f"\n{'='*80}")
        print(f"Colonnes {i} à {end-1}:")
        print(f"{'='*80}")
        print(df.iloc[:, i:end])
        print()


def print_details(arr : np.ndarray, label, max_column=100):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    count = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    min = np.nanmin(arr, axis=0)
    p25 = np.nanpercentile(arr, 25, axis=0)
    p50 = np.nanpercentile(arr, 50, axis=0)
    p75 = np.nanpercentile(arr, 75, axis=0)
    max = np.nanmax(arr, axis=0)
    data = {
        'Count': count,
        'Mean': mean,
        'Std': std,
        'Min': min,
        '25%': p25,
        '50%': p50,
        '75%': p75,
        'Max': max
	}
    df = pd.DataFrame(data, index=label)
    df = df.T
    print_wide_dataframe(df, cols_per_block=max_column)

def get_arr(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return
        arr = df.apply(pd.to_numeric, errors='coerce')
        arr.dropna(axis=1, how='all', inplace=True)
        if arr.empty:
            return
        return (arr.astype(float).to_numpy(), arr.columns.to_list())
    except Exception as e:
        print(e)

def main():
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=str, help='Path to csv file to analyze')
    parser.add_argument('--max-column', type=int, default = 100, help='visual option to manage the print')
    args = parser.parse_args()
    arr = get_arr(args.file)
    if arr is not None:
        print_details(arr[0], arr[1], args.max_column)
    else:
        print("No data or invalid data.")

if __name__ == '__main__':
    main()