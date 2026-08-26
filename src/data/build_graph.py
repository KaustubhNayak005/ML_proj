import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import argparse

# Configuration
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
TRANSACTION_FILE = os.path.join(RAW_DATA_DIR, 'train_transaction.csv')
IDENTITY_FILE = os.path.join(RAW_DATA_DIR, 'train_identity.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'graph.pt')

# Relations to build and degree cap (to prevent hub explosion)
RELATION_COLS = ['card1', 'card2', 'addr1', 'addr2', 'P_emaildomain', 'DeviceInfo']
DEGREE_CAP = 100  # If an entity appears more than 100 times, do not build edges between all its transactions

def check_files():
    if not os.path.exists(TRANSACTION_FILE) or not os.path.exists(IDENTITY_FILE):
        print("Error: IEEE-CIS dataset files not found.")
        print(f"Please download 'train_transaction.csv' and 'train_identity.csv' from Kaggle (IEEE-CIS Fraud Detection)")
        print(f"and place them in the '{RAW_DATA_DIR}' directory.")
        return False
    return True

def build_graph(limit=None):
    print("Loading data...")
    # Read data (limiting rows for initial testing can be done here if needed)
    if limit is not None:
        df_trans = pd.read_csv(TRANSACTION_FILE, nrows=limit)
        df_id = pd.read_csv(IDENTITY_FILE, nrows=limit)
    else:
        df_trans = pd.read_csv(TRANSACTION_FILE)
        df_id = pd.read_csv(IDENTITY_FILE)
    
    # Merge on TransactionID
    df = df_trans.merge(df_id, on='TransactionID', how='left')
    
    # Sort by time to allow time-based split
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    print(f"Total transactions: {len(df)}")
    
    labels = df['isFraud'].values
    y = torch.tensor(labels, dtype=torch.long)
    
    print("Processing node features...")
    # Very basic feature processing (impute NaNs, encode categoricals, scale)
    # Exclude IDs, target, and relation columns from node features
    exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT'] + RELATION_COLS
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    features_df = df[feature_cols].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = features_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = features_df.select_dtypes(include=['object']).columns
    
    # Impute numeric with 0 (a simplistic approach, could be improved)
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    train_end = int(len(df) * 0.7)
    
    # Encode categorical
    features_df[categorical_cols] = features_df[categorical_cols].fillna('MISSING')
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if len(categorical_cols) > 0:
        encoder.fit(features_df.iloc[:train_end][categorical_cols])
        features_df[categorical_cols] = encoder.transform(features_df[categorical_cols])
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(features_df.iloc[:train_end])
    x_scaled = scaler.transform(features_df)
    x = torch.tensor(x_scaled, dtype=torch.float)
    
    print("Building edges...")
    edge_indices = {}
    
    for col in RELATION_COLS:
        if col not in df.columns:
            continue
            
        print(f"  Processing relation: {col}")
        # Drop missing values for the relation
        valid_nodes = df[df[col].notna()][['TransactionID', col]]
        valid_nodes['node_idx'] = valid_nodes.index
        
        # Group by the entity value
        grouped = valid_nodes.groupby(col)['node_idx'].apply(list)
        
        # Filter out groups that exceed DEGREE_CAP to avoid hub explosion
        grouped = grouped[grouped.apply(len) <= DEGREE_CAP]
        grouped = grouped[grouped.apply(len) > 1] # Need at least 2 nodes to make an edge
        
        # Build edges (fully connected within each group)
        src = []
        dst = []
        for indices in grouped:
            for i in indices:
                for j in indices:
                    if i != j:
                        src.append(i)
                        dst.append(j)
                        
        if len(src) > 0:
            edge_indices[col] = torch.tensor([src, dst], dtype=torch.long)
            print(f"    Created {len(src)} edges for {col}")
        else:
            print(f"    No edges created for {col} (all groups exceeded cap or were size 1)")
    
    # Create single combined edge_index for basic GraphSAGE/GAT
    # (Advanced model in Phase 7 might use the separate relations)
    all_src = []
    all_dst = []
    for ei in edge_indices.values():
        all_src.extend(ei[0].tolist())
        all_dst.extend(ei[1].tolist())
        
    if len(all_src) > 0:
        combined_edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    else:
        combined_edge_index = torch.empty((2, 0), dtype=torch.long)
        
    print(f"Total combined edges: {combined_edge_index.shape[1]}")
    
    # Create train/val/test masks based on time (TransactionDT sorting)
    # E.g., 70% train, 15% val, 15% test
    n_nodes = len(df)
    train_end = int(n_nodes * 0.7)
    val_end = int(n_nodes * 0.85)
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    
    data = Data(x=x, edge_index=combined_edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    # Store individual relation edge indices as extra attributes
    for col, ei in edge_indices.items():
        setattr(data, f'edge_index_{col}', ei)
        
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    torch.save(data, OUTPUT_FILE)
    print(f"Graph saved to {OUTPUT_FILE}")
    print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows to load")
    args = parser.parse_args()

    # Change working directory to project root if executed from elsewhere
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    os.chdir(proj_root)
    
    if check_files():
        build_graph(limit=args.limit)
