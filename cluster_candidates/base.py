from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from scipy.spatial.distance import squareform
from .support import pair_val_from_vec, dist_between_points, corrcoef


@dataclass
class SameCellParams:
    """Parameters for same-cell candidate analysis.

    Attributes
    ----------
    keep_planes : List[int] | None
        List of plane indices to analyze. When None, will use the keep_planes
        defined in the session parameters (default: None)
    npix_cutoff : Optional[int]
        Minimum number of pixels for ROI masks (default: None)
    pix_to_um : float
        Conversion factor from pixels to micrometers (default: 1.3)
    """

    keep_planes: List[int] | None = None
    npix_cutoff: Optional[int] = None
    pix_to_um: float = 1.3

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> "SameCellParams":
        """Create a SameCellParams instance from a dictionary."""
        return cls(**params_dict)


@dataclass
class SameCellClusterParameters:
    """Parameters for identifying same-cell clusters. These are used to filter ROI pairs
    that probably represent the same cell. It's written as a dataclass to standardize it
    for continued use elsewhere.

    Attributes
    ----------
    corr_cutoff : float
        Minimum correlation threshold for a pair to be considered a same-cell candidate
    distance_cutoff : float
        Maximum distance between ROI pairs in μm
    keep_planes : List[int]
        List of plane indices to include default to all planes [0, 1, 2, 3, 4] to make sure
        that the session objects won't filter out planes before clustering analysis!
    npix_cutoff : float
        Minimum number of pixels for ROI masks
    min_distance : float | None
        Minimum distance between ROI pairs in μm
    max_correlation : float | None
        Maximum correlation between ROI pairs
    """

    corr_cutoff: float = 0.4
    distance_cutoff: float = 20.0
    keep_planes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    npix_cutoff: float = 0.0
    min_distance: float | None = None
    max_correlation: float | None = None


class SameCellProcessor:
    """Processes session data to extract ROI correlations, distances, and metadata.

    This class handles the data preparation phase of same-cell candidate analysis.
    It extracts ROI activity, positions, and calculates pairwise correlations and distances.
    """

    def __init__(
        self,
        spks: np.ndarray,
        roi_plane_idx: np.ndarray,
        stat: list[dict],
        params=None,
        mode: str = "weightedmean",
    ):
        """Initialize the SameCellProcessor.

        Parameters
        ----------
        spks : np.ndarray
            Activity data (time x ROIs)
        roi_plane_idx : np.ndarray
            Plane indices of ROIs (ROIs x 1)
        stat : list[dict]
            Suite2P stat output for each ROI
        params : SameCellParams or dict, optional
            Parameters for the analysis. If dict, will be converted to SameCellParams.
        """
        # Core attributes
        self.spks = spks
        self.roi_plane_idx = roi_plane_idx
        self.params = SameCellParams() if params is None else params

        # Convert params dict to SameCellParams if needed
        if isinstance(self.params, dict):
            self.params = SameCellParams.from_dict(self.params)

        self.num_rois = self.spks.shape[1]

        # Load ROI stats and positions
        self.roi_npix = np.array([s["npix"] for s in stat]).astype(np.int32)
        ypix = [s["ypix"] for s in stat]
        xpix = [s["xpix"] for s in stat]
        yc = np.array([np.median(y) for y in ypix])
        xc = np.array([np.median(x) for x in xpix])
        self.roi_xy_pos = np.stack((xc, yc)).T

        # Calculate pairwise correlations and distances
        self.pairwise_correlations = squareform(corrcoef(spks.T), checks=False)
        self.pairwise_distances = dist_between_points(self.roi_xy_pos[:, 0], self.roi_xy_pos[:, 1])

        # Create pair indices
        self.idx_roi1, self.idx_roi2 = pair_val_from_vec(np.arange(self.num_rois))
        self.plane_pair1, self.plane_pair2 = pair_val_from_vec(self.roi_plane_idx)
        self.npix_pair1, self.npix_pair2 = pair_val_from_vec(self.roi_npix)
        self.xpos_pair1, self.xpos_pair2 = pair_val_from_vec(self.roi_xy_pos[:, 0])
        self.ypos_pair1, self.ypos_pair2 = pair_val_from_vec(self.roi_xy_pos[:, 1])

        # Calculate number of pairs
        self.num_pairs = len(self.pairwise_correlations)
        assert self.num_pairs == self.num_rois * (self.num_rois - 1) / 2, "Pair calculation error"

    @property
    def keep_planes(self) -> List[int]:
        """Get the keep planes from the session parameters."""
        return self.params.keep_planes

    def get_pair_filter(
        self,
        *,
        npix_cutoff: Optional[int] = None,
        keep_planes: Optional[List[int]] = None,
        corr_cutoff: Optional[float] = None,
        distance_cutoff: Optional[float] = None,
        extra_filter: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate boolean filter for ROI pairs based on multiple criteria.

        Parameters
        ----------
        npix_cutoff : int, optional
            Minimum number of pixels for ROI masks
        keep_planes : list, optional
            List of plane indices to include
        corr_cutoff : float, optional
            Minimum correlation threshold
        distance_cutoff : float, optional
            Maximum distance between ROI pairs in μm
        extra_filter : np.ndarray, optional
            Additional boolean filter to apply

        Returns
        -------
        np.ndarray
            Boolean array indicating which pairs pass all filters
        """
        if keep_planes is not None:
            assert set(keep_planes) <= set(
                self.keep_planes
            ), f"Requested planes not in loaded data. Available planes: {self.keep_planes}"

        # Start with all pairs
        pair_idx = np.full(self.num_pairs, True)

        # Apply filters
        if npix_cutoff is not None:
            pair_idx &= (self.npix_pair1 > npix_cutoff) & (self.npix_pair2 > npix_cutoff)

        if keep_planes is not None:
            pair_idx &= np.any(np.stack([self.plane_pair1 == pidx for pidx in keep_planes]), axis=0)
            pair_idx &= np.any(np.stack([self.plane_pair2 == pidx for pidx in keep_planes]), axis=0)

        if corr_cutoff is not None:
            pair_idx &= self.pairwise_correlations > corr_cutoff

        if distance_cutoff is not None:
            pair_idx &= self.pairwise_distances < distance_cutoff

        if extra_filter is not None:
            pair_idx &= extra_filter

        return pair_idx

    def filter_pairs(self, pair_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Filter ROI pair data based on boolean index.

        Parameters
        ----------
        pair_idx : np.ndarray
            Boolean array indicating which pairs to keep

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing filtered versions of all pair measurements
        """
        return {
            "pairwise_correlations": self.pairwise_correlations[pair_idx],
            "pairwise_distances": self.pairwise_distances[pair_idx],
            "plane_pair1": self.plane_pair1[pair_idx],
            "plane_pair2": self.plane_pair2[pair_idx],
            "npix_pair1": self.npix_pair1[pair_idx],
            "npix_pair2": self.npix_pair2[pair_idx],
            "xpos_pair1": self.xpos_pair1[pair_idx],
            "xpos_pair2": self.xpos_pair2[pair_idx],
            "ypos_pair1": self.ypos_pair1[pair_idx],
            "ypos_pair2": self.ypos_pair2[pair_idx],
            "idx_roi1": self.idx_roi1[pair_idx],
            "idx_roi2": self.idx_roi2[pair_idx],
        }


def get_connected_groups(adjacency_matrix: np.ndarray, filter_islands: bool = True) -> List[List[int]]:
    """Find connected components in an undirected graph.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Square adjacency matrix where non-zero values indicate connections

    Returns
    -------
    List[List[int]]
        List of connected components, where each component is a list of node indices
    """
    # Validate input
    assert (
        adjacency_matrix.ndim == 2 and adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
    ), "Input must be a square matrix"
    assert np.all(adjacency_matrix == adjacency_matrix.T), "Input must be symmetric (undirected graph)"

    n = adjacency_matrix.shape[0]

    # Convert to set representation for efficient operations
    graph = []
    for i in range(n):
        connected = np.where(adjacency_matrix[i] > 0)[0]
        graph.append(set(connected))

    # Find all connected components
    visited = set()
    components = []

    for node in range(n):
        if node not in visited:
            # Start a new component
            component = set([node])
            frontier = set(graph[node])

            # Expand component until no new nodes are added
            while frontier:
                component.update(frontier)
                new_frontier = set()
                for f_node in frontier:
                    new_frontier.update(graph[f_node])
                frontier = new_frontier - component

            components.append(sorted(list(component)))
            visited.update(component)

    if filter_islands:
        components = [c for c in components if len(c) > 1]

    return components


def get_best_roi(scp: SameCellProcessor, cluster: List[int]):
    """This picks which ROI to keep in a cluster."""
    # Get sum of significant activity for each ROI (good measure of SNR)
    # We only consider SNR in this session only because it's not tracked
    activity = scp.spks[:, cluster]

    # OR WHATEVER YOU WANT TO USE TO PICK THE BEST ROI
    roi_snr = np.sum(activity, axis=0)
    idx_choice = np.argmax(roi_snr)

    return idx_choice
