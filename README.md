# Same-Cell Candidate Analysis Pipeline

This document describes the methodology used to identify and analyze potential same-cell candidates across different imaging planes. The pipeline processes Region of Interest (ROI) data from multiple imaging planes to identify clusters of ROIs that likely belong to the same neuron.

## Overview

The analysis pipeline consists of three main stages:
1. Data preparation and preprocessing
2. Pair-wise ROI analysis
3. Cluster identification and best ROI selection

There's also a number of visualization tools that help with parameter tuning and analysis.

## Installation
First, clone the repository
```bash
git clone https://github.com/landoskape/cluster-candidates
```

Then, install the package using pip
```bash
pip install -e .
```

The package uses torch to speed up the correlation coefficient calculation, so it's recommended to install torch with GPU support.
To install torch, follow the instructions [here](https://pytorch.org/get-started/locally/) after installing the dependencies.

## Data Preparation
1. Core data inputs:
   - `spks`: Activity data matrix (time x ROIs)
   - `roi_plane_idx`: Plane indices for each ROI
   - `stat`: Suite2P stat output containing ROI metadata

2. Optional parameters (`SameCellParams`):
   - `keep_planes`: List of plane indices to analyze
   - `npix_cutoff`: Minimum number of pixels for ROI masks
   - `pix_to_um`: Conversion factor from pixels to micrometers (default: 1.3)

## Pair-wise ROI Analysis

For each pair of ROIs, the pipeline computes:

1. Activity correlation:
   - Pearson correlation coefficient between ROI time series
   - Computed efficiently using GPU-accelerated operations when available (`_torch_corrcoef`)

2. Spatial metrics:
   - Euclidean distance \(d\) between ROI centers:
     \[ d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} \]
   where \((x_1, y_1)\) and \((x_2, y_2)\) are the coordinates of the ROI centers in μm

## Cluster Parameters

ROIs are grouped into clusters based on the following configurable parameters (`SameCellClusterParameters`):

1. Activity correlation thresholds:
   - Minimum correlation coefficient (`corr_cutoff`, default: 0.4)
   - Optional maximum correlation threshold (`max_correlation`)

2. Spatial constraints:
   - Maximum distance (`distance_cutoff`, default: 20 μm)
   - Optional minimum distance (`min_distance`)

3. Plane selection:
   - Default includes planes 0-4 (`keep_planes = [0, 1, 2, 3, 4]`)
   - Configurable to analyze specific subset of planes
   - We include all planes for consistency and because plane 0 is the one that has the highest hit rate for tracked cells - which means that if plane 0 ROI is good quality and part of a cluster, we'll probably pick that one to serve as the clusters representative. 

4. Additional filters:
   - Minimum ROI size (`npix_cutoff`, default: 0)
   - Optional good label filtering (`good_labels`)

## Cluster Identification

The pipeline uses these parameters to:

1. Create pair filters based on:
   - Correlation thresholds
   - Distance constraints
   - Plane selection
   - ROI size requirements
   - Additional custom filters

2. Construct an adjacency matrix where ROI pairs meeting all criteria are considered connected

3. Identify clusters using connected components analysis:
   - ROIs can be in the same cluster even if not directly connected
   - Connected through transitive relationships via other ROIs
   - Option to filter out single-ROI "islands"

## Best ROI Selection Algorithm

The best ROI selection process has been simplified to focus on activity-based metrics:

### **Selection Process**
1. For each cluster, calculate the sum of significant activity (SNR) for each ROI
2. Select the ROI with the highest SNR value

This simplified approach focuses on the quality of the signal in the current session. The selection can be customized by modifying the `get_best_roi` function to include additional selection criteria if needed. (I use a variety of other criteria including type of mask, plane, whether the ROI is tracked across sessions, etc.).

## Visualization and Analysis Tools

The pipeline includes several visualization tools for quality control and parameter optimization:

1. Correlation Analysis:
   - Correlation vs. distance scatter plots
   - Plane-pair histograms
   - Interactive parameter adjustment

2. Cluster Explorer:
   - Interactive visualization of ROI clusters
   - Activity trace comparison
   - Spatial relationship visualization
   - ROI mask inspection
   - Neuropil signal analysis
   - Color coding by plane or ROI index
   - Integration with ROI classification results

3. Distribution Analysis:
   - Cluster size distribution
   - Distance distribution between correlated ROIs
   - ROI removal analysis for different strategies


## Usage

See the notebook at `cluster_candidates.ipynb` for code examples. 

The pipeline is typically used as follows:

1. Prepare your input data:
   ```python
   # Example data format
   spks = np.array(...)  # time x ROIs activity matrix
   roi_plane_idx = np.array(...)  # ROI plane indices
   stat = [...]  # Suite2P stat output
   ```

2. Create a SameCellProcessor with the data:
   ```python
   processor = SameCellProcessor(
       spks=spks,
       roi_plane_idx=roi_plane_idx,
       stat=stat,
       params=SameCellParams(keep_planes=[0, 1, 2, 3, 4]) # for example... but not needed
   )
   ```

3. Use the processor to analyze clusters:
   ```python
   # Get pair filter based on criteria
   pair_filter = processor.get_pair_filter(
       corr_cutoff=0.4,
       distance_cutoff=20.0,
       keep_planes=[0, 1, 2, 3, 4]
   )
   
   # Get filtered pair data
   filtered_data = processor.filter_pairs(pair_filter)
   ```

4. Visualize results using the provided tools:
   ```python
   viewer = ClusterExplorer(processor, stat, ops, neuropil=None)
   ```

## License
This project is licensed under the GNU GPLv3 License. See the LICENSE file for details.

## Contact
For questions or feedback, please raise an issue on the GitHub repository.
