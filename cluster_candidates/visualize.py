from dataclasses import dataclass
from typing import List, Optional, Tuple
from copy import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from syd import make_viewer, Viewer
from .base import SameCellProcessor, get_connected_groups, SameCellClusterParameters, get_best_roi
from .support import compute_cross_correlations


def plot_correlation_vs_distance(processor: SameCellProcessor) -> Viewer:
    """Make correlation vs distance scatter plot with interactive parameters.

    Parameters
    ----------
    processor : SameCellProcessor
        Processor containing ROI pair data

    Returns
    -------
    viewer : syd.Viewer
        Viewer object
    """
    from matplotlib.colors import LogNorm

    def plot(state):
        keep_planes = state["keep_planes"]
        distance_cutoff = state["distance_cutoff"]
        plane_category = state["plane_category"]

        pair_idx = processor.get_pair_filter(keep_planes=keep_planes, distance_cutoff=distance_cutoff)
        filtered_data = processor.filter_pairs(pair_idx)

        if plane_category == "all":
            idx_category = np.ones(len(filtered_data["pairwise_distances"]), dtype=bool)
        elif plane_category == "same plane":
            idx_category = filtered_data["plane_pair1"] == filtered_data["plane_pair2"]
        elif plane_category == "neighbor":
            idx_category = np.abs(filtered_data["plane_pair1"] - filtered_data["plane_pair2"]) == 1
        elif plane_category == "far plane":
            idx_category = np.abs(filtered_data["plane_pair1"] - filtered_data["plane_pair2"]) > 1
        else:
            raise ValueError(f"Invalid plane category: {plane_category}")

        title = f"Corr vs Distance for {plane_category} ROI pairs"
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        # ax.scatter(filtered_data["pairwise_distances"][idx_category], filtered_data["pairwise_correlations"][idx_category], c="k", alpha=alpha)
        h = ax.hist2d(
            filtered_data["pairwise_distances"][idx_category],
            filtered_data["pairwise_correlations"][idx_category],
            bins=[50, 50],
            norm=LogNorm(),
            cmap="viridis",
            density=True,
        )
        plt.colorbar(h[3], ax=ax, label="Log - Density")
        ax.set_xlabel("Distance (μm)")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        return fig

    planes = list(processor.params.keep_planes)
    max_distance = np.max(processor.pairwise_distances)

    viewer = make_viewer(plot)
    viewer.add_multiple_selection("keep_planes", value=planes, options=planes)
    viewer.add_float("distance_cutoff", value=max_distance, min=0, max=max_distance, step=1.0)
    viewer.add_selection("plane_category", value="all", options=["all", "same plane", "neighbor", "far plane"])
    return viewer


def plot_plane_pair_histograms(
    processor: SameCellProcessor,
    corr_cutoffs: List[float] = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    distance_cutoff: Optional[float] = None,
    figsize: Tuple[int, int] = (5, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot histograms of ROI pairs across planes for different correlation thresholds.

    Parameters
    ----------
    processor : SameCellProcessor
        Processor containing ROI pair data
    corr_cutoffs : List[float]
        List of correlation thresholds to analyze
    distance_cutoff : float, optional
        Maximum distance between ROI pairs in μm
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get all available planes
    all_planes = sorted(processor.keep_planes)
    n_planes = len(all_planes)

    # Create figure
    fig, axes = plt.subplots(len(corr_cutoffs), 1, figsize=figsize, sharex=True)
    if len(corr_cutoffs) == 1:
        axes = [axes]

    # Create plane pair combinations
    plane_pairs = []
    for i in range(n_planes):
        for j in range(i, n_planes):
            plane_pairs.append((all_planes[i], all_planes[j]))

    # Create x-axis labels
    x_labels = [f"{p1}-{p2}" for p1, p2 in plane_pairs]
    x_pos = np.arange(len(plane_pairs))

    # Create colormap
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(corr_cutoffs))]

    # Plot histograms for each correlation threshold
    for i, corr_cutoff in enumerate(corr_cutoffs):
        # Get filtered pairs
        pair_idx = processor.get_pair_filter(corr_cutoff=corr_cutoff, distance_cutoff=distance_cutoff)
        filtered_data = processor.filter_pairs(pair_idx)

        # Count pairs for each plane combination
        counts = np.zeros(len(plane_pairs))
        for j, (p1, p2) in enumerate(plane_pairs):
            mask = ((filtered_data["plane_pair1"] == p1) & (filtered_data["plane_pair2"] == p2)) | (
                (filtered_data["plane_pair1"] == p2) & (filtered_data["plane_pair2"] == p1)
            )
            counts[j] = np.sum(mask)

        # Plot histogram
        axes[i].bar(x_pos, counts, color=colors[i], alpha=0.7)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"Correlation > {corr_cutoff:.2f}")

        # Add count labels
        for j, count in enumerate(counts):
            if count > 0:
                axes[i].text(j, count + 0.5, str(int(count)), ha="center")

    # Set x-axis labels on bottom plot
    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Plane Pairs")

    # Adjust layout
    plt.tight_layout()

    return fig, axes


def plot_cluster_size_distribution(
    processor: SameCellProcessor,
    corr_cutoffs: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    distance_cutoff: float = 30,
    min_distance: Optional[float] = None,
    keep_planes: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (5, 5),
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot distribution of cluster sizes for different correlation thresholds.

    Parameters
    ----------
    processor : SameCellProcessor
        Processor containing ROI pair data
    corr_cutoffs : List[float]
        List of correlation thresholds to analyze
    distance_cutoff : float
        Maximum distance between ROI pairs in μm
    min_distance : float, optional
        Minimum distance between ROI pairs in μm
    keep_planes : List[int], optional
        List of plane indices to include
    figsize : Tuple[int, int]
        Figure size
    verbose : bool
        Whether to print clustering statistics

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(corr_cutoffs))]

    # Store max cluster size for setting x-axis limit
    max_cluster_size = 0

    # Process each correlation threshold
    for i, corr_cutoff in enumerate(corr_cutoffs):
        # Get filtered pairs
        pair_idx = processor.get_pair_filter(
            corr_cutoff=corr_cutoff, distance_cutoff=distance_cutoff, keep_planes=keep_planes
        )

        if min_distance is not None:
            # Add minimum distance filter
            pair_idx &= processor.pairwise_distances >= min_distance

        # If no pairs meet criteria, skip
        if not np.any(pair_idx):
            continue

        # Create adjacency matrix for connected components analysis
        n_rois = processor.num_rois
        adj_matrix = np.zeros((n_rois, n_rois), dtype=bool)

        # Fill adjacency matrix
        filtered_data = processor.filter_pairs(pair_idx)
        for idx1, idx2 in zip(filtered_data["idx_roi1"], filtered_data["idx_roi2"]):
            adj_matrix[int(idx1), int(idx2)] = True
            adj_matrix[int(idx2), int(idx1)] = True

        # Find connected components
        clusters = get_connected_groups(adj_matrix)

        # Count cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters]
        size_counts = Counter(cluster_sizes)

        # Update max cluster size
        if cluster_sizes and max(cluster_sizes) > max_cluster_size:
            max_cluster_size = max(cluster_sizes)

        # Plot histogram
        sizes = sorted(size_counts.keys())
        counts = [size_counts[s] for s in sizes]

        ax.plot(sizes, counts, "o-", color=colors[i], label=f"r > {corr_cutoff:.2f}")

        # Print statistics
        if verbose:
            print(f"Correlation > {corr_cutoff:.2f}:")
            print(f"  Total clusters: {len(clusters)}")
            print(f"  Mean cluster size: {np.mean(cluster_sizes):.2f}")
            print(f"  Max cluster size: {max(cluster_sizes) if cluster_sizes else 0}")
            print(f"  Size distribution: {dict(size_counts)}")
            print()

    # Set axis labels and title
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Size Distribution")

    # Set x-axis limit and use integer ticks
    ax.set_xlim(0.5, max_cluster_size + 0.5)
    ax.set_xticks(range(1, max_cluster_size + 1))

    # Use log scale for y-axis if range is large
    if ax.get_ylim()[1] / ax.get_ylim()[0] > 10:
        ax.set_yscale("log")

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_distance_distribution(
    processor: SameCellProcessor,
    corr_cutoffs: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6],
    max_distance: float = 200,
    normalize: str = "counts",
    keep_planes: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (5, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot distribution of distances between correlated ROIs.

    Parameters
    ----------
    processor : SameCellProcessor
        Processor containing ROI pair data
    corr_cutoffs : List[float]
        List of correlation thresholds to analyze
    max_distance : float
        Maximum distance to include in histogram
    normalize : str
        Normalization method: 'counts', 'density', or 'probability'
    keep_planes : List[int], optional
        List of plane indices to include
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, len(corr_cutoffs))]

    # Create distance bins
    bins = np.linspace(0, max_distance, 41)  # 40 bins

    # Process each correlation threshold
    for i, corr_cutoff in enumerate(corr_cutoffs):
        # Get filtered pairs
        pair_idx = processor.get_pair_filter(corr_cutoff=corr_cutoff, keep_planes=keep_planes)

        # If no pairs meet criteria, skip
        if not np.any(pair_idx):
            continue

        # Get distances for filtered pairs
        distances = processor.pairwise_distances[pair_idx]

        counts, bin_edges = np.histogram(distances, bins=bins, density=True)
        if normalize == "probability":
            counts = counts / counts.sum()

        ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, counts, "-", color=colors[i], label=f"r > {corr_cutoff:.2f}")

    # Set axis labels and title
    ax.set_xlabel("Distance (μm)")
    if normalize == "counts":
        ax.set_ylabel("Count")
    elif normalize == "density":
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Probability")
    ax.set_title("Distance Distribution Between Correlated ROIs")

    # Set x-axis limit
    ax.set_xlim(0, max_distance)

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_roi_removal_analysis(
    processor: SameCellProcessor,
    corr_cutoffs: List[float] = np.linspace(0, 1, 11),
    max_bin_connections: int = 25,
    keep_planes: Optional[List[int]] = None,
    distance_cutoff: float = 40,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot analysis of ROI removal strategies and connection distributions.

    Parameters
    ----------
    processor : SameCellProcessor
        Processor containing ROI pair data
    corr_cutoffs : List[float]
        List of correlation thresholds to analyze
    max_bin_connections : int
        Maximum number of connections to show in histogram
    keep_planes : List[int], optional
        List of plane indices to include
    distance_cutoff : float
        Maximum distance between ROI pairs in μm
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Function to count ROIs that need to be removed using simple strategy
    def remove_count_simple(adj_matrix):
        """Count ROIs removed by removing any ROI in a pair."""
        n_rois = adj_matrix.shape[0]
        removed = np.zeros(n_rois, dtype=bool)

        # For each pair of connected ROIs, remove one
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                if adj_matrix[i, j] and not (removed[i] or removed[j]):
                    removed[j] = True

        return np.sum(removed)

    # Function to count ROIs using Maximal Independent Set (MIS) strategy
    def remove_count_mis(adj_matrix):
        """Count removed by removing only ROIs necessary to make a maximal independent set."""
        n_rois = adj_matrix.shape[0]
        removed = np.zeros(n_rois, dtype=bool)

        # Create list of ROIs sorted by number of connections (descending)
        roi_connections = np.sum(adj_matrix, axis=1)
        roi_order = np.argsort(-roi_connections)

        # Process ROIs in order
        for roi in roi_order:
            # If this ROI is already marked for removal, skip
            if removed[roi]:
                continue

            # Find all ROIs connected to this one
            connected = np.where(adj_matrix[roi])[0]

            # Mark all connected ROIs for removal
            removed[connected] = True

        return np.sum(removed)

    # Function to count connections per ROI
    def count_connections(adj_matrix):
        """Count number of connections per ROI."""
        return np.sum(adj_matrix, axis=1)

    # Store results
    simple_counts = []
    mis_counts = []
    connection_distributions = []

    # Process each correlation threshold
    for corr_cutoff in corr_cutoffs:
        # Get filtered pairs
        pair_idx = processor.get_pair_filter(
            corr_cutoff=corr_cutoff, distance_cutoff=distance_cutoff, keep_planes=keep_planes
        )

        # If no pairs meet criteria, add zeros
        if not np.any(pair_idx):
            simple_counts.append(0)
            mis_counts.append(0)
            connection_distributions.append(np.zeros(max_bin_connections + 1))
            continue

        # Create adjacency matrix
        n_rois = processor.num_rois
        adj_matrix = np.zeros((n_rois, n_rois), dtype=bool)

        # Fill adjacency matrix
        filtered_data = processor.filter_pairs(pair_idx)
        for idx1, idx2 in zip(filtered_data["idx_roi1"], filtered_data["idx_roi2"]):
            adj_matrix[int(idx1), int(idx2)] = True
            adj_matrix[int(idx2), int(idx1)] = True

        # Count ROIs to remove with simple strategy
        simple_count = remove_count_simple(adj_matrix)
        simple_counts.append(simple_count)

        # Count ROIs to remove with MIS strategy
        mis_count = remove_count_mis(adj_matrix)
        mis_counts.append(mis_count)

        # Count connections per ROI
        connections = count_connections(adj_matrix)

        # Create histogram of connections
        hist, _ = np.histogram(connections, bins=np.arange(max_bin_connections + 2) - 0.5)
        connection_distributions.append(hist)

    # Plot ROI removal counts
    axes[0].plot(corr_cutoffs, simple_counts, "o-", label="Simple")
    axes[0].plot(corr_cutoffs, mis_counts, "s-", label="MIS")
    axes[0].set_xlabel("Correlation Threshold")
    axes[0].set_ylabel("ROIs Removed")
    axes[0].set_title("ROI Removal Strategies")
    axes[0].legend()

    # Plot ROI removal percentage
    total_rois = processor.num_rois
    axes[1].plot(corr_cutoffs, np.array(simple_counts) / total_rois * 100, "o-", label="Simple")
    axes[1].plot(corr_cutoffs, np.array(mis_counts) / total_rois * 100, "s-", label="MIS")
    axes[1].set_xlabel("Correlation Threshold")
    axes[1].set_ylabel("% ROIs Removed")
    axes[1].set_title("ROI Removal Percentage")
    axes[1].legend()

    # Plot connection distribution heatmap
    connection_matrix = np.array(connection_distributions).T
    im = axes[2].imshow(
        connection_matrix,
        aspect="auto",
        origin="lower",
        extent=[min(corr_cutoffs), max(corr_cutoffs), -0.5, max_bin_connections + 0.5],
        cmap="viridis",
    )
    axes[2].set_xlabel("Correlation Threshold")
    axes[2].set_ylabel("Number of Connections")
    axes[2].set_title("Connection Distribution")
    plt.colorbar(im, ax=axes[2], label="Count")

    # Adjust layout
    plt.tight_layout()

    return fig, axes


@dataclass
class ClusterParameters:
    corr_cutoff: float
    distance_cutoff: float
    keep_planes: List[int]
    npix_cutoff: float
    min_distance: float
    max_correlation: float


class ClusterExplorer(Viewer):
    def __init__(
        self,
        same_cell_processor: SameCellProcessor,
        stat: list[dict],
        ops: dict,
        neuropil: Optional[np.ndarray] = None,
    ):
        self.scp = same_cell_processor
        self.stat = stat
        self.ops = ops
        self.neuropil = neuropil
        self.Ly = self.ops["Ly"]
        self.Lx = self.ops["Lx"]
        self.roi_plane_idx = self.scp.roi_plane_idx

        initial_keep_planes = list(self.scp.keep_planes)

        self.cluster_params = SameCellClusterParameters()
        corr_cutoff = (self.cluster_params.corr_cutoff, self.cluster_params.max_correlation or 1.0)
        distance_cutoff = (self.cluster_params.min_distance or 0.0, self.cluster_params.distance_cutoff)
        npix_cutoff = self.cluster_params.npix_cutoff
        keep_planes = self.cluster_params.keep_planes

        self.add_integer("cluster_idx", value=0, min=0, max=0)
        self.add_boolean("zscore_traces", value=False)
        self.add_boolean("show_neuropil", value=False)
        self.add_boolean("show_cross_correlations", value=False)
        self.add_integer("roi_idx", value=0, min=0, max=10)
        self.add_float_range("corr_cutoff", value=corr_cutoff, min=0.0, max=1.0, step=0.05)
        self.add_float_range("distance_cutoffs", value=distance_cutoff, min=0.0, max=100.0, step=1.0)
        self.add_multiple_selection("keep_planes", value=initial_keep_planes, options=keep_planes)
        self.add_float("npix_cutoff", value=npix_cutoff, min=0, max=10000)
        self.add_boolean("color_by_plane", value=False)
        self.add_integer("max_rois", value=10, min=1, max=1000)
        self.add_button("print_rois", label="Print ROIs", callback=self.print_rois)

        self.on_change(["corr_cutoff", "distance_cutoffs", "keep_planes", "npix_cutoff"], self.change_cluster_params)
        self.on_change("cluster_idx", self.change_cluster)
        self.on_change("show_neuropil", self.check_neuropil)

        # Use callback to set initial clusters
        self.change_cluster_params(self.state)

    def check_neuropil(self, state):
        if state["show_neuropil"] and self.neuropil is None:
            print(
                "Neuropil is not loaded. Please reload with neuropil data or set explorer.neuropil = neuropil then try again."
            )

    def set_clusters(self, corr_cutoff, distance_cutoff, keep_planes, npix_cutoff, min_distance, max_correlation):
        """Update the clusters based on the given parameters. Doesn't update if they match previous parameters."""
        # Check if the parameters are different
        need_update = False
        if not hasattr(self, "params") or not hasattr(self, "clusters"):
            need_update = True
            self.params = ClusterParameters(
                corr_cutoff, distance_cutoff, keep_planes, npix_cutoff, min_distance, max_correlation
            )
        else:
            new_params = ClusterParameters(
                corr_cutoff, distance_cutoff, keep_planes, npix_cutoff, min_distance, max_correlation
            )
            if self.params != new_params:
                need_update = True
                self.params = new_params

        if not need_update:
            return self.clusters

        # Get pair filter
        extra_filter = np.ones(self.scp.num_pairs, dtype=bool)
        if self.params.min_distance > 0.0:
            extra_filter &= self.scp.pairwise_distances >= self.params.min_distance
        if self.params.max_correlation < 1.0:
            extra_filter &= self.scp.pairwise_correlations <= self.params.max_correlation

        pair_filter = self.scp.get_pair_filter(
            corr_cutoff=self.params.corr_cutoff,
            distance_cutoff=self.params.distance_cutoff,
            keep_planes=self.params.keep_planes,
            npix_cutoff=self.params.npix_cutoff,
            extra_filter=extra_filter,
        )

        n_rois = self.scp.num_rois
        adj_matrix = np.zeros((n_rois, n_rois), dtype=bool)
        filtered_data = self.scp.filter_pairs(pair_filter)
        for idx1, idx2 in zip(filtered_data["idx_roi1"], filtered_data["idx_roi2"]):
            adj_matrix[int(idx1), int(idx2)] = True
            adj_matrix[int(idx2), int(idx1)] = True

        # Get connected groups
        self.clusters = get_connected_groups(adj_matrix)

    def get_mask(self, roi_idx):
        ypix = self.stat[roi_idx]["ypix"]
        xpix = self.stat[roi_idx]["xpix"]
        lam = self.stat[roi_idx]["lam"]
        image = np.zeros((self.Ly, self.Lx), dtype=np.uint8)
        image[ypix, xpix] = lam
        return image

    def get_hull(self, mask):
        binary = (255 * (mask > 0)).astype(np.uint8)
        dilated = cv2.dilate(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        complete_contour = np.concatenate([contour, contour[[0]]], axis=0)
        return complete_contour

    def change_cluster_params(self, state):
        corr_cutoff, max_correlation = state["corr_cutoff"]
        min_distance, distance_cutoff = state["distance_cutoffs"]
        keep_planes = state["keep_planes"]
        npix_cutoff = state["npix_cutoff"]
        self.set_clusters(corr_cutoff, distance_cutoff, keep_planes, npix_cutoff, min_distance, max_correlation)
        num_clusters = len(self.clusters)
        self.update_integer("cluster_idx", max=num_clusters - 1)
        self.change_cluster(self.state)

    def change_cluster(self, state):
        num_rois = len(self.clusters[state["cluster_idx"]])
        self.update_integer("roi_idx", value=0, max=num_rois - 1)

    def print_rois(self, state):
        cluster = self.clusters[state["cluster_idx"]]
        selected = cluster[state["roi_idx"]]
        print("Current Cluster:", cluster, "Selected ROI:", selected)

    def plot(self, state: dict):
        # Get clusters (should have been update already by on_change)
        cluster = self.clusters[state["cluster_idx"]]

        # Get the selected ROI from within this cluster
        num_rois = len(cluster)
        roi_idx = state["roi_idx"]

        # Get the activity and neuropil for the selected ROI
        times = np.arange(self.scp.spks.shape[0])
        roi_activity = self.scp.spks[:, cluster]
        if state["show_neuropil"]:
            roi_neuropil = self.neuropil[:, cluster]

        # Choose the best ROI
        best_in_cluster = get_best_roi(self.scp, cluster)
        best_message = f"Best ROI={best_in_cluster}"

        # Select ROIs if there are too many
        if len(cluster) > state["max_rois"]:
            _old_cluster = copy(cluster)
            cluster = [cluster[best_in_cluster]]
            num_to_add = state["max_rois"] - len(cluster)
            _cluster_to_add = copy(_old_cluster)
            _cluster_to_add.pop(best_in_cluster)
            for ii in np.random.permutation(len(_cluster_to_add))[:num_to_add]:
                cluster.append(_cluster_to_add[ii])
            icluster = [_old_cluster.index(roi) for roi in cluster]
            roi_activity = roi_activity[:, icluster]
            if state["show_neuropil"]:
                roi_neuropil = roi_neuropil[:, icluster]
            num_rois = len(cluster)
            best_in_cluster = 0

        # Normalize the traces
        if state["zscore_traces"]:
            roi_activity = roi_activity - np.median(roi_activity, axis=0)
            roi_activity = roi_activity / np.std(roi_activity, axis=0)
            if state["show_neuropil"]:
                roi_neuropil = roi_neuropil - np.median(roi_neuropil, axis=0)
                roi_neuropil = roi_neuropil / np.std(roi_neuropil, axis=0)

        # Measure ROI Contours
        masks = [self.get_mask(roi_idx) for roi_idx in cluster]
        contours = [self.get_hull(mask) for mask in masks]
        plane_idx = self.roi_plane_idx[cluster]

        # Set up the figure and axes
        num_cols = 2 + state["show_neuropil"] + state["show_cross_correlations"]
        mask_title = "ROI Mask"

        # Color settings
        if state["color_by_plane"]:
            cmap = mpl.colormaps["jet"].resampled(len(self.scp.keep_planes))
            relative_plane_idx = [np.where(self.scp.keep_planes == plane)[0][0] for plane in plane_idx]
            colors = [cmap(i) for i in relative_plane_idx]
            color_array = np.array([cmap(i) for i in range(len(self.scp.keep_planes))])
            color_array = color_array.reshape(1, -1, 4)  # Reshape colors to horizontal stack
            xticks = self.scp.keep_planes
            selected_plane = plane_idx[roi_idx]
            xtick_labels = [
                f"Plane {plane}\n(selected)" if plane == selected_plane else f"Plane {plane}" for plane in xticks
            ]
        else:
            cmap = mpl.colormaps["jet"].resampled(num_rois)
            colors = [cmap(i) for i in range(num_rois)]
            color_array = np.array(colors).reshape(1, -1, 4)  # Reshape colors to horizontal stack
            xticks = np.arange(num_rois)
            xtick_labels = [f"ROI {i}\n(selected)" if i == roi_idx else f"ROI {i}" for i in xticks]

        # Create the figure
        fig = plt.figure(figsize=(8, 5), layout="constrained")
        width_ratios = [1] * (num_cols - 1) + [0.4]
        gs = fig.add_gridspec(3, num_cols, height_ratios=[1, 1, 0.1], width_ratios=width_ratios)
        ax_traces = fig.add_subplot(gs[0, 0])
        ax_imtraces = fig.add_subplot(gs[1, 0])
        ax_imtraces.sharex(ax_traces)
        if state["show_neuropil"]:
            ax_neuropil = fig.add_subplot(gs[0, 1])
            ax_neuropil.sharex(ax_traces)
            ax_imneuropil = fig.add_subplot(gs[1, 1])
            ax_imneuropil.sharex(ax_traces)
        if state["show_cross_correlations"]:
            ax_cross_correlations = fig.add_subplot(gs[0:2, -2])
        ax_hulls = fig.add_subplot(gs[0, -1])
        ax_masks = fig.add_subplot(gs[1, -1])
        ax_cbar = fig.add_subplot(gs[2, :])

        # Plot the activity traces
        for iroi, cluster_id in enumerate(cluster):
            color = "black" if iroi == roi_idx else colors[iroi]
            linewidth = 0.8 if iroi == roi_idx else 0.5
            zorder = 10 if iroi == roi_idx else 1
            ax_traces.plot(times, roi_activity[:, iroi], color=color, linewidth=linewidth, zorder=zorder)
        ax_traces.set_title(f"ROI Activity\n{best_message}")
        ax_traces.set_xlabel("Time (s)")
        ax_traces.set_ylabel("Z-score")

        # Plot all activity
        ax_imtraces.imshow(
            roi_activity.T,
            aspect="auto",
            cmap="gray_r",
            origin="lower",
            extent=(times[0], times[-1], 0, num_rois),
            interpolation="none",
        )
        ax_imtraces.set_xlabel("Time (s)")
        ax_imtraces.set_ylabel("ROI Index")
        ax_imtraces.set_yticks(np.arange(num_rois) + 0.5)
        yticklabels = [f"--> {i}" if i == roi_idx else f"{i}" for i in range(num_rois)]
        ax_imtraces.set_yticklabels(yticklabels)

        if state["show_neuropil"]:
            for iroi, cluster_id in enumerate(cluster):
                color = "black" if iroi == roi_idx else colors[iroi]
                linewidth = 0.8 if iroi == roi_idx else 0.5
                zorder = 10 if iroi == roi_idx else 1
                ax_neuropil.plot(times, roi_neuropil[:, iroi], color=color, linewidth=linewidth, zorder=zorder)
            ax_neuropil.set_title(f"Neuropil Activity - {cluster_id}")
            ax_neuropil.set_xlabel("Time (s)")
            ax_neuropil.set_ylabel("Z-score")

            # Plot all neuropil
            ax_imneuropil.imshow(
                roi_neuropil.T,
                aspect="auto",
                cmap="gray_r",
                origin="lower",
                extent=(times[0], times[-1], 0, num_rois),
                interpolation="none",
            )
            ax_imneuropil.set_xlabel("Time (s)")
            ax_imneuropil.set_ylabel("ROI Index")
            ax_imneuropil.set_yticks(np.arange(num_rois) + 0.5)
            ax_imneuropil.set_yticklabels(yticklabels)

        if state["show_cross_correlations"]:
            lags, cross_correlations = compute_cross_correlations(roi_activity.T, max_lag=50)
            for iroi, cluster_id in enumerate(cluster):
                color = "black" if iroi == roi_idx else colors[iroi]
                linewidth = 0.8 if iroi == roi_idx else 0.5
                zorder = 10 if iroi == roi_idx else 1
                ax_cross_correlations.plot(
                    lags, cross_correlations[roi_idx, iroi], color=color, linewidth=linewidth, zorder=zorder
                )
            ax_cross_correlations.set_xlabel("Lag (s)")
            ax_cross_correlations.set_ylabel("Cross-correlation")

        # Plot the ROI Contours
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for iroi, cluster_id in enumerate(cluster):
            color = "black" if iroi == roi_idx else colors[iroi]
            linewidth = 1.2 if iroi == roi_idx else 1
            zorder = 10 if iroi == roi_idx else 1
            ax_hulls.plot(
                contours[iroi][:, 0, 0], contours[iroi][:, 0, 1], color=color, linewidth=linewidth, zorder=zorder
            )
            min_x = min(min_x, np.min(contours[iroi][:, 0, 0]))
            max_x = max(max_x, np.max(contours[iroi][:, 0, 0]))
            min_y = min(min_y, np.min(contours[iroi][:, 0, 1]))
            max_y = max(max_y, np.max(contours[iroi][:, 0, 1]))

        center_x = (min_x + max_x) / 2
        range_x = max_x - min_x
        center_y = (min_y + max_y) / 2
        range_y = max_y - min_y
        ax_hulls.set_xlim(center_x - range_x, center_x + range_x)
        ax_hulls.set_ylim(center_y - range_y, center_y + range_y)
        ax_hulls.set_aspect("equal")
        ax_hulls.set_xticks([])
        ax_hulls.set_yticks([])
        ax_hulls.set_title("ROI Contours")

        # Plot the masks
        ax_masks.imshow(masks[roi_idx], cmap="gray_r", aspect="equal", origin="lower", extent=(0, self.Lx, 0, self.Ly))
        ax_masks.set_xlim(center_x - range_x, center_x + range_x)
        ax_masks.set_ylim(center_y - range_y, center_y + range_y)
        ax_masks.set_aspect("equal")
        ax_masks.set_xticks([])
        ax_masks.set_yticks([])
        ax_masks.set_title(mask_title)

        # Create custom colorbar
        ax_cbar.imshow(color_array, aspect="auto", origin="lower")
        ax_cbar.set_yticks([])
        ax_cbar.set_xticks(xticks)
        ax_cbar.set_xticklabels(xtick_labels)

        return fig
