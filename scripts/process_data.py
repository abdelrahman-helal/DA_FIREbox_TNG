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


def sample_pyg_subgraph(data: Data, num_nodes_sample: int, random_state: int = 42) -> Data:
    """Sample a random subset of nodes from a PyG Data object and return the induced subgraph."""
    if num_nodes_sample <= 0:
        raise ValueError("num_nodes_sample must be a positive integer")
    if num_nodes_sample > data.num_nodes:
        raise ValueError("num_nodes_sample cannot exceed the number of nodes in the graph")

    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)

    sampled_indices = rng.choice(data.num_nodes, size=num_nodes_sample, replace=False)
    sampled_indices_sorted = np.sort(sampled_indices)
    index_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(sampled_indices_sorted)}
    sampled_node_set = set(sampled_indices_sorted.tolist())

    edge_index = data.edge_index
    if edge_index.numel() == 0:
        filtered_edges = edge_index
        filtered_edge_attr = None
    else:
        edge_mask = []
        for src, dst in edge_index.t().cpu().tolist():
            edge_mask.append(int(src) in sampled_node_set and int(dst) in sampled_node_set)
        edge_mask = torch.tensor(edge_mask, dtype=torch.bool, device=edge_index.device)
        filtered_edges = edge_index[:, edge_mask]

        filtered_edge_attr = None
        if hasattr(data, "edge_attr") and getattr(data, "edge_attr", None) is not None:
            filtered_edge_attr = data.edge_attr[edge_mask]

    new_edge_index = torch.zeros_like(filtered_edges)
    for i in range(filtered_edges.shape[1]):
        old_src = int(filtered_edges[0, i].item())
        old_dst = int(filtered_edges[1, i].item())
        new_edge_index[0, i] = index_map[old_src]
        new_edge_index[1, i] = index_map[old_dst]

    sampled_x = data.x[sampled_indices_sorted]
    sampled_y = data.y[sampled_indices_sorted]
    sampled_pos = data.pos[sampled_indices_sorted] if hasattr(data, "pos") and data.pos is not None else None

    sampled_data = Data(
        x=sampled_x,
        edge_index=new_edge_index,
        y=sampled_y,
        pos=sampled_pos,
    )

    if filtered_edge_attr is not None:
        sampled_data.edge_attr = filtered_edge_attr

    # FIX (train-test-split): preserve each sampled node's ORIGINAL train/val/test
    # membership by indexing the original masks with the same sampled_indices_sorted
    # used for x/y/pos above -- the same pattern the notebook's own
    # induced_subgraph_by_mask() already uses correctly for the overlap restriction.
    #
    # The previous version discarded this membership entirely: it computed
    # train/val counts from the ORIGINAL split's ratio, then reassigned them to a
    # freshly `rng.shuffle`d, UNSTRATIFIED ordering of the num_nodes_sample sampled
    # nodes. That (a) is not stratified by halo mass, unlike the stratified split
    # DataProcessor/TNGDataProcessor.create_graph_data() originally produced, and
    # (b) lets a node that was e.g. in the original test split end up reassigned to
    # train (or vice versa) purely by the reshuffle, for no reason tied to the data.
    # Indexing directly inherits the original (correctly stratified) partition, so
    # the sampled graph's split is a faithful subsample of the full graph's split
    # rather than an independently re-randomized one.
    idx_t = torch.as_tensor(sampled_indices_sorted, dtype=torch.long)
    if hasattr(data, "train_mask") and getattr(data, "train_mask", None) is not None:
        train_mask = data.train_mask[idx_t]
    else:
        train_mask = torch.zeros(num_nodes_sample, dtype=torch.bool)

    if hasattr(data, "val_mask") and getattr(data, "val_mask", None) is not None:
        val_mask = data.val_mask[idx_t]
    else:
        val_mask = torch.zeros(num_nodes_sample, dtype=torch.bool)

    if hasattr(data, "test_mask") and getattr(data, "test_mask", None) is not None:
        test_mask = data.test_mask[idx_t]
    else:
        test_mask = torch.zeros(num_nodes_sample, dtype=torch.bool)

    sampled_data.train_mask = train_mask
    sampled_data.val_mask = val_mask
    sampled_data.test_mask = test_mask

    if hasattr(data, "overdensity") and getattr(data, "overdensity", None) is not None:
        sampled_data.overdensity = data.overdensity[sampled_indices_sorted]
    if hasattr(data, "periodic") and getattr(data, "periodic", None) is not None:
        sampled_data.periodic = data.periodic
    if hasattr(data, "box_lengths") and getattr(data, "box_lengths", None) is not None:
        sampled_data.box_lengths = data.box_lengths

    return sampled_data


class DataProcessor():
    """
    Load FIREbox host halos and build a standard PyG graph (kNN or radius) for halo-mass regression.
    """

    def __init__(self, file_path='../data/FIREbox_z=0.txt', subhalos=False):
        """
        Initialize paths and load the filtered host-halo table.

        Parameters:
        -----------
        file_path : str
            Path to the FIREbox text catalog (whitespace-separated, ``#`` comments).

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
        Load FIREbox data from text file and process it.

        Parameters:
        -----------
        None
            Reads from ``self.file_path``.

        Returns:
        --------
        pd.DataFrame
            Processed dataframe with:
            - Only rows where hostHaloID = -1
            - Xc, Yc, Zc columns renamed to pos_x, pos_y, pos_z
            Also assigns the full table to ``self.df``.
        """
        # Read the data file, skipping comment lines (starting with #)
        self.df = pd.read_csv(self.file_path, sep=r'\s+', comment='#')
        
        # Filter for rows where hostHaloID = -1
        if self.subhalos == "both":
            df_filtered = self.df.copy()
        elif self.subhalos:
            df_filtered = self.df[self.df['hostHaloID'] != -1].copy()
        else:
            df_filtered = self.df[self.df['hostHaloID'] == -1].copy()
        
        # Rename position and velocity columns 
        df_filtered = df_filtered.rename(columns={
            'Xc': 'pos_x',
            'Yc': 'pos_y', 
            'Zc': 'pos_z', 
            'VXc': 'vel_x',
            'VYc': 'vel_y',
            'VZc': 'vel_z'
        })
        
        # Divide position values by 10^3
        # df_filtered['pos_x'] = df_filtered['pos_x'] / 100
        # df_filtered['pos_y'] = df_filtered['pos_y'] / 100
        # df_filtered['pos_z'] = df_filtered['pos_z'] / 100
        
        return df_filtered

    def create_graph_data(self, k=None, r=1, leaf_size=40, test_size=0.1, val_size=0.1,
                        standardize=True, random_state=42, include_Rhalo=False,
                        stratify_bins=10, lower_mass=9.5, upper_mass=14, periodic=True):
        """
        Filter galaxies, stratify train/val/test, optionally add Rhalo, and build kNN or radius edges.

        Parameters:
        -----------
        k : int or None
            Number of nearest neighbors (excluding self); if None, use radius graph with ``r``.
        r : float
            Pairwise distance threshold for radius graph when ``k`` is None.
        leaf_size : int
            Unused (reserved for tree backends); kept for API compatibility.
        test_size : float
            Fraction of nodes in the held-out test split.
        val_size : float
            Fraction of nodes in the validation split (relative to full graph).
        standardize : bool
            If True, fit ``StandardScaler`` on training rows only, then transform all rows.
        random_state : int
            Seed for stratified train/val/test splits.
        include_Rhalo : bool
            If True, append ``Rhalo`` to node features.
        stratify_bins : int or None
            Number of halo-mass percentile bins for stratified splitting; None disables stratification.

        Returns:
        --------
        torch_geometric.data.Data
            Graph with ``x``, ``edge_index``, ``y``, ``pos``, and ``train_mask`` / ``val_mask`` / ``test_mask``.
        """

        # Define feature columns and target
        self.df_filtered = self.df_filtered[self.df_filtered['lg_Mstar_<Rhalo'] > 0]

        feature_cols = ['pos_x', 'pos_y', 'pos_z',
                        'vel_x', 'vel_y', 'vel_z', 'lg_Mstar_<Rhalo']
        if include_Rhalo:
            feature_cols += ['Rhalo']

        target_col = 'lg_Mhalo'

        # Halo mass cutoff for galaxy formation
        self.df_filtered = self.df_filtered[(self.df_filtered['lg_Mhalo'] > lower_mass) & (self.df_filtered['lg_Mhalo'] < upper_mass)]

        X = self.df_filtered[feature_cols].values.astype(np.float64)
        y = self.df_filtered[target_col].values
        print(f"Max {target_col}: {max(self.df_filtered[target_col])}")
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
        # val_size fraction of the full dataset → fraction of temp set
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
        stellar_log = self.df_filtered['lg_Mstar_<Rhalo'].values.astype(np.float64)

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


# if __name__ == "__main__":
#     data_processor = DataProcessor()
#     data_processor.create_graph_data()
#     print(data_processor.data)
#     print(data_processor.data.train_mask)
