import torch
import numpy as np
import pandas as pd

import torch_geometric
from torch_geometric.data import Data

from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def shift_positions_to_box(pos: np.ndarray):
    """Shift coordinates so each axis begins at 0 and infer periodic box lengths."""
    origin = pos.min(axis=0)
    shifted = pos - origin
    span = shifted.max(axis=0) - shifted.min(axis=0)
    span = np.maximum(span, 1e-6)
    upper = shifted.max(axis=0)
    box_lengths = np.maximum(span, upper * (1.0 + 1e-9) + 1e-6)
    return shifted, origin, box_lengths


def compute_edge_attributes(edge_index_np: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                            periodic: bool = False, box_lengths: np.ndarray | None = None):
    """Compute edge distance and velocity-difference attributes for a directed edge list."""
    if edge_index_np.size == 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    src = edge_index_np[0]
    dst = edge_index_np[1]
    pos_t = torch.tensor(pos, dtype=torch.float32)
    vel_t = torch.tensor(vel, dtype=torch.float32)
    row = torch.tensor(src, dtype=torch.long)
    col = torch.tensor(dst, dtype=torch.long)
    diff = pos_t[row] - pos_t[col]

    if periodic and box_lengths is not None:
        L = torch.tensor(box_lengths, dtype=torch.float32, device=diff.device)
        for axis in range(3):
            d = diff[:, axis]
            L_axis = L[axis]
            diff[:, axis] = d - L_axis * torch.round(d / L_axis)

    dist = torch.linalg.norm(diff, dim=1)
    vel_diff = vel_t[row] - vel_t[col]
    vel_norm = torch.linalg.norm(vel_diff, dim=1)
    return torch.stack([dist, vel_norm], dim=1)


def compute_overdensity(edge_index_np: np.ndarray, stellar_log: np.ndarray, n_nodes: int):
    """Compute log10 stellar mass sum over neighbor links for each node."""
    if edge_index_np.size == 0:
        return np.full(n_nodes, np.nan, dtype=np.float64)

    src = edge_index_np[0]
    dst = edge_index_np[1]
    mass_lin = (10.0 ** stellar_log[src]).astype(np.float64)
    sums = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(sums, dst, mass_lin)
    return np.log10(np.maximum(sums, 1e-30))


class TNGDataProcessor():
    """
    Load IllustrisTNG subhalo parquet and expose FIREbox-compatible graph construction.
    """

    def __init__(self, file_path='../data/tng-data/TNG300-1-subhalos_99.parquet', subhalos='both'):
        """
        Initialize paths and load the renamed TNG subhalo table.

        Parameters:
        -----------
        file_path : str
            Path to the TNG subhalos parquet file.

        Returns:
        --------
        None
            Sets ``self.file_path``, ``self.df``, ``self.data``, and ``self.df_filtered``.
        """
        self.file_path = file_path
        self.df = None
        self.data = Data()
        self.subhalos = subhalos
        self.df_filtered = self.load_data()

    def load_data(self):
        """
        Load TNG subhalos data from parquet file and process it.

        Parameters:
        -----------
        None
            Reads from ``self.file_path``.

        Returns:
        --------
        pd.DataFrame
            Processed dataframe with renamed columns to match FIREbox convention.
            Also assigns the raw table to ``self.df``.
        """
        # Read the parquet file
        self.df = pd.read_parquet(self.file_path)

        if self.subhalos=='both':
            # Include both subhalos and main halos
            pass
        elif self.subhalos==True:
            # Filter to only include subhalos (subhalo_id > 0)
            self.df = self.df[self.df['is_central'] == False].copy()
        elif self.subhalos==False:
            # Filter to only include main halos (subhalo_id == 0)
            self.df = self.df[self.df['is_central'] == True].copy()

        # Rename columns to match FIREbox naming convention
        df_filtered = self.df.rename(columns={
            'subhalo_x': 'pos_x',
            'subhalo_y': 'pos_y',
            'subhalo_z': 'pos_z',
            'subhalo_vx': 'vel_x',
            'subhalo_vy': 'vel_y',
            'subhalo_vz': 'vel_z',
            'subhalo_logstellarmass': 'stellar_mass'
        }).copy()
        
        return df_filtered

    def create_graph_data(self, k=None, r=1, leaf_size=40, test_size=0.1, val_size=0.1,
                        standardize=True, random_state=42, stratify_bins=10, 
                        upper_mass=14, periodic=True):
        """
        Create a PyTorch Geometric graph from the TNG data.

        Parameters:
        -----------
        k : int or None
            Number of nearest neighbors for graph connectivity (None → radius-based).
        r : float
            Radius for radius-based connectivity (used when ``k`` is None).
        leaf_size : int
            Unused; reserved for API compatibility.
        test_size : float
            Fraction of data held out as the final test set (default 0.10).
        val_size : float
            Fraction of data used for validation / early-stopping (default 0.10).
        standardize : bool
            Whether to standardize features (train-fit scaler).
        random_state : int
            Random seed for reproducibility.
        stratify_bins : int or None
            Number of mass-percentile bins used to stratify all splits.

        Returns:
        --------
        torch_geometric.data.Data
            Object with ``x``, ``edge_index``, ``y``, ``pos``, and train/val/test masks.
        """

        feature_cols = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'stellar_mass']
        target_col = 'subhalo_loghalomass'

        self.df_filtered = self.df_filtered.dropna(subset=feature_cols + [target_col])
        self.df_filtered = self.df_filtered[self.df_filtered[target_col] <= upper_mass]
        X = self.df_filtered[feature_cols].values.astype(np.float64)
        y = self.df_filtered[target_col].values

        N = len(self.df_filtered)
        idx = np.arange(N)

        # ── 3-way stratified split (train / val / test) ──────────────────────
        # Split indices BEFORE fitting the scaler to prevent data leakage.
        if stratify_bins is not None:
            bins = np.percentile(y, np.linspace(0, 100, stratify_bins + 1))
            y_bins = np.digitize(y, bins[1:-1])
        else:
            y_bins = None

        temp_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state,
            stratify=y_bins)

        y_bins_temp = y_bins[temp_idx] if y_bins is not None else None
        val_frac_of_temp = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            temp_idx, test_size=val_frac_of_temp, random_state=random_state,
            stratify=y_bins_temp)

        # ── Fit scaler on training rows ONLY, then transform all rows ────────
        if standardize:
            self.scaler = StandardScaler()
            X[train_idx] = self.scaler.fit_transform(X[train_idx])
            X[val_idx]   = self.scaler.transform(X[val_idx])
            X[test_idx]  = self.scaler.transform(X[test_idx])
        X_scaled = X

        # ── Build graph on standardized positions (all nodes, transductive setting) ─
        pos_arr = X_scaled[:, 0:3]
        pos_arr, _, box_lengths = shift_positions_to_box(pos_arr)

        vel_arr = self.df_filtered[['vel_x', 'vel_y', 'vel_z']].values.astype(np.float64)
        stellar_log = self.df_filtered['stellar_mass'].values.astype(np.float64)

        if k:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(pos_arr)
            _, indices = nbrs.kneighbors(pos_arr)
            pairs = []
            for i, neigh in enumerate(indices):
                for j in neigh[1:]:
                    pairs.append((i, j))
            if pairs:
                pairs = np.asarray(pairs, dtype=np.int64)
                src = pairs[:, 0]
                dst = pairs[:, 1]
                edge_index_np = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
            else:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)
        else:
            tree = cKDTree(pos_arr, boxsize=box_lengths) if periodic else cKDTree(pos_arr)
            pairs = tree.query_pairs(r=r, output_type='ndarray')
            if pairs.size == 0:
                edge_index_np = np.zeros((2, 0), dtype=np.int64)
            else:
                src = np.concatenate([pairs[:, 0], pairs[:, 1]])
                dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
                edge_index_np = np.stack([src, dst], axis=0)

        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        edge_attr = compute_edge_attributes(edge_index_np, pos_arr, vel_arr, periodic=periodic, box_lengths=box_lengths)
        overdensity = compute_overdensity(edge_index_np, stellar_log, N)

        # ── Build masks ───────────────────────────────────────────────────────
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx]     = True
        test_mask[test_idx]   = True

        self.data = Data(
            x=torch.tensor(X_scaled, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y, dtype=torch.float).unsqueeze(1),
            pos=torch.tensor(pos_arr, dtype=torch.float),
        )
        self.data.train_mask = train_mask
        self.data.val_mask   = val_mask
        self.data.test_mask  = test_mask
        self.data.overdensity = torch.tensor(overdensity, dtype=torch.float32)
        self.data.periodic = periodic
        self.data.box_lengths = torch.tensor(box_lengths, dtype=torch.float32)

        return self.data

