

import pandas as pd
import json
import ast


def read_csv_with_tuples(filepath, tuple_columns=None):
    """
    Read a CSV file and convert specified columns from string representation of tuples back to actual tuples.
    
    Parameters:
    filepath (str): Path to the CSV file
    tuple_columns (list): List of column names that should be converted to tuples. 
                         If None, attempts to convert any column that contains tuple-like strings.
    
    Returns:
    pandas.DataFrame: DataFrame with tuple columns properly converted
    """
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    if tuple_columns is None:
        # Try to identify columns that contain tuple-like strings
        tuple_columns = []
        for col in df.columns:
            # Check first non-null value in each column
            first_val = df[col].dropna().iloc[0] if not df[col].empty else None
            if isinstance(first_val, str) and first_val.startswith('(') and first_val.endswith(')'):
                tuple_columns.append(col)
    
    # Convert string representations of tuples back to actual tuples
    for col in tuple_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    
    return df


def to_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
