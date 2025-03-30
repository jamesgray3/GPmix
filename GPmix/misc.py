
__all__ = ['estimate_nclusters', 'silhouette_score', 'davies_bouldin_score', 'dpa', 'fast_dpa']

from sklearn.mixture import GaussianMixture
from skfda.preprocessing.dim_reduction import FPCA
# from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import davies_bouldin_score as dbs
import matplotlib.pyplot as plt
import numpy as np

from numba import jit

def silhouette_score(fd, y, **kwargs):
    return ss(fd.data_matrix.squeeze(), y, **kwargs)

def davies_bouldin_score(fd, y):
    return dbs(fd.data_matrix.squeeze(), y)

def gmms_fit_plot_(weights, means, stdev, ax = None, **kwargs):
    '''Plot Gaussian mixture density curves'''
    d_pdf = lambda x, mu, sigma: np.exp(-0.5 * (((x - mu) / sigma)) ** 2) / (sigma * np.sqrt(2 * np.pi))
    for i in range(len(weights)):
        x = np.linspace(means[i] - 3 * stdev[i], means[i] + 3 * stdev[i], 50)
        if ax:
            ax.plot(x, weights[i] * d_pdf(x, means[i], stdev[i]), linewidth = 3, **kwargs)
        else:
            plt.plot(x, weights[i] * d_pdf(x, means[i], stdev[i]), linewidth = 3, **kwargs)

def match_labels(cluster_labels, true_class_labels, cluster_class_labels_perm):
    '''Permute cluster labels to match specified labels'''
    label_match_dict = {}
    for key, value in zip(cluster_class_labels_perm, true_class_labels):
        label_match_dict[key] = value

    matched_labels = np.zeros_like(cluster_labels)
    for i in cluster_class_labels_perm:
        matched_labels[np.argwhere(cluster_labels == i)] = label_match_dict[i]
    return matched_labels

def estimate_nclusters(fdata, ncluster_grid = None):
    '''
    Estimate the number of clusters in a functional dataset.

    Parameters
    ----------
    fdata : Dataset
        The functional dataset for which the number of clusters is to be estimated.
    ncluster_grid : array-like, optional
        A list or array specifying the grid within which the number of clusters is searched. 
        By default, ncluster_grid is set to [2, 3, ..., 14].

    Returns
    -------
    n_clusters : int
        The estimated number of clusters in the functional dataset.
    '''

    if ncluster_grid is None:
        ncluster_grid = range(2,15)

    fpca_ = FPCA(n_components = 1)
    scores = fpca_.fit_transform(fdata)

    bic_ = []
    aic_ = []
    for n_comp in range(2, 15):
        model = GaussianMixture(n_components=n_comp, n_init= 20)
        model.fit(scores)
        bic_.append(model.bic(scores))
        aic_.append(model.aic(scores))
    
    return min([ncluster_grid[np.argmin(aic_)], ncluster_grid[np.argmin(bic_)]])

@jit(nopython=True)
def _expand_window_from_path(low_res_path, grid_size, radius):
    """Expand the window from the low resolution path"""
    path = low_res_path * 2.0
    truth_table = np.zeros((grid_size, grid_size), dtype=np.bool_)
    
    def point_to_line_segment_distance(px, py, x1, y1, x2, y2):
        line_vec_x, line_vec_y = x2 - x1, y2 - y1
        point_vec_x, point_vec_y = px - x1, py - y1
        
        line_len_squared = line_vec_x ** 2 + line_vec_y ** 2
        if line_len_squared == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        
        t = max(0, min(1, (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_len_squared))
        
        closest_x, closest_y = x1 + t * line_vec_x, y1 + t * line_vec_y
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
    
    for x in range(grid_size):
        for y in range(grid_size):
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                
                dist = point_to_line_segment_distance(x, y, x1, y1, x2, y2)
                
                if dist <= radius:
                    truth_table[x, y] = True
                    break
    
    return truth_table

@jit(nopython=True)
def _cost_function(a, b, n, p, q):
    """Cost function for dynamic programming algorithm. Calculates inner product of basis function and de-warped sample."""
    m = b.shape[0]
    (k, l), (i, _) = p, q

    x = np.arange(k, min(i + 1, m), 1, dtype=np.int64)
    y = (x - k) * m + l

    idx = np.round(y * m / n).astype(np.int64)
    idx = np.minimum(idx, m - 1)

    return np.dot(a[x], b[idx])

@jit(nopython=True)
def _preceding_neighbours(x, y, search_range=3):
    """Return selected set of neighbours for a given cell"""
    return (
        [(x - 1, y - 1)]
        + [(x - 1, y - dy) for dy in range(2, search_range + 1)]
        + [(x - dx, y - 1) for dx in range(2, search_range + 1)]
    )

@jit(nopython=True)
def _reconstruct_path(path_matrix):
    """Reconstruct the optimal path from the path matrix"""
    path_len = path_matrix.shape[0]
    path = [(path_len - 1, path_len - 1)]
    
    while path[-1] != (0,0):
        x, y = path[-1]
        x, y = int(x), int(y)
        nx, ny = path_matrix[x, y]
        path.append((nx, ny))
    
    path.reverse()
    return np.array(path)

@jit(nopython=True)
def dpa(f, g, n=20, window=None):
    """Dynamic programming algorithm for computing the optimal de=warping path by maximisation of cost function"""
    if window is None:
        window = np.ones((n, n), dtype=np.bool_)

    H = np.zeros((n, n), dtype=np.float64)
    P = np.zeros((n, n, 2), dtype=np.int64)
    
    for i in range(1, n):
        for j in range(1, n):
            if not window[i, j]:
                continue

            neighbours = [
                (k, l) for k, l in _preceding_neighbours(i, j)
                if k >= 0 and l >= 0 and window[k,l]
            ]
            Hc = np.array([
                H[k, l] + _cost_function(f, g, n, (k, l), (i, j))
                for k, l in neighbours
                if k >= 0 and l >= 0
            ])

            if len(Hc) == 0:
                H[i, j] = float('-inf')
            else:
                H[i, j] = np.max(Hc)
                P[i, j] = neighbours[np.argmax(Hc)]
    
    return H[-1, -1], _reconstruct_path(P)

def fast_dpa(f, g, n, radius):
    """Adaptation of "FastDTW" algorithm to speed up de-warping process"""
    if n < radius * 2 + 1:
        return dpa(f, g, n=n)

    _, low_res_path = fast_dpa(f, g, n // 2, radius)
    window = _expand_window_from_path(low_res_path, n, radius)
    
    return dpa(f, g, n=n, window=window)
