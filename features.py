import numpy as np
from scipy import ndimage
import functools
from typing import Callable, Union, Any
from skimage import measure
from scipy.fft import fft
from skimage.measure import regionprops
from scipy.spatial import cKDTree

def catch_error(default_value: Union[float, tuple, list] = np.nan):
    """
    Decorator that catches exceptions and returns NaN or specified default value.

    Args:
        default_value: Value to return on error (default: np.nan)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error if needed
                print(f"Error in {func.__name__}: {e}")
                return default_value

        return wrapper

    return decorator


def get_centroid(binary_mask):
    return ndimage.center_of_mass(binary_mask)

@catch_error()
def min_distance_to_object(binary_mask, point):
    """
    Calculate minimal distance from a point to a binary blob.

    Args:
        binary_mask: 2D binary array where blob pixels are True/1
        point: tuple (y, x) coordinates of the point

    Returns:
        float: Minimal distance in pixels
    """
    # Create distance transform (distance to nearest blob pixel)
    distance_map = ndimage.distance_transform_edt(~binary_mask)

    # Get distance at the specific point - convert to integers for indexing
    y, x = int(round(point[0])), int(round(point[1]))

    return distance_map[y, x]

@catch_error()
def edge_distance_to_nucleus(mask_of_interest, nucleus_mask):
    """
    Calculate the distance of a masks edge to the nucleus mask
    """
    mask_of_interest_distance_map = ndimage.distance_transform_edt(~mask_of_interest)
    return min(mask_of_interest_distance_map[nucleus_mask > 0])


@catch_error()
def centroid_distance_to_nucleus(mask_of_interest, nucleus_mask ):
    """ 
    Calculate the distance of a masks centroid to the nucleus mask
    """
    centroid = get_centroid(mask_of_interest)
    return min_distance_to_object(nucleus_mask, centroid)


def touch_area(mask1, mask2, number_dilations):
    """
    Calculate the area of contact between two binary masks after dilation.
    Args:
        mask1: First binary mask (2D array)
        mask2: Second binary mask (2D array)
        number_dilations: Number of dilations to apply to both masks
    Returns:
        int: Area of contact in pixels
    """
    dilated_mask1 = ndimage.binary_dilation(mask1, iterations=number_dilations)
    dilated_mask2 = ndimage.binary_dilation(mask2, iterations=number_dilations)
    return np.sum(np.logical_and(dilated_mask1, dilated_mask2))

def spherical_volume(mask):
    """
    Calculate the volume of a sphere given its surface area.
    :return:
    """
    area = np.sum(mask)
    return (4 / 3) * np.pi * (np.sqrt(area / np.pi)) ** 3


@catch_error()
def get_aspect_ratio(binary_mask):
    """
    Calculate aspect ratio of a binary blob.

    Args:
        binary_mask: 2D binary array where blob pixels are True/1

    Returns:
        float: Aspect ratio (major_axis_length / minor_axis_length)
    """
    # Label the binary mask (required for regionprops)
    labeled_mask = binary_mask.astype(int)

    # Get region properties
    props = regionprops(labeled_mask)

    if len(props) == 0:
        return np.nan

    # Get the first (and should be only) region
    region = props[0]

    # Calculate aspect ratio
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length

    if minor_axis == 0:
        return np.nan

    aspect_ratio = major_axis / minor_axis
    return aspect_ratio


# Updated ellipsoid_volume function
@catch_error(default_value=[])
def ellipsoid_volume(mask):
    """
    Calculate the volume of an ellipsoid given its surface area and aspect ratio.
    returns: a tuple of (prolate_volume, oblate_volume) where:
        - prolate_volume assumes the major axis is the longest dimension (like a rugby ball)
        - oblate_volume assumes the minor axis is the longest dimension (like a lentil)
    """

    area = np.sum(mask)
    aspect_ratio = get_aspect_ratio(mask)


    # Calculate ellipsoid volumes
    base_radius = np.sqrt(area / (np.pi * aspect_ratio))
    return (4 / 3) * np.pi * base_radius ** 3 * aspect_ratio, (4 / 3) * np.pi * base_radius ** 3 / aspect_ratio

def nuclear_periphery(granule_label_image, nucleus_label_image, number_dilations):
    """
    Calculate the nuclear periphery for each granule in the granule label image.

    Args:
        granule_label_image: 2D array where each unique value represents a different granule.
        nucleus_label_image: 2D binary array representing the nucleus.
        number_dilations: Number of dilations to apply to the nucleus.

    Returns:
        list: A list of dictionaries with granule index and nuclear periphery area.
    """
    data = []
    nucleus_periphery = np.logical_xor(
        ndimage.binary_dilation(nucleus_label_image == 1, iterations=number_dilations),
        ndimage.binary_erosion(nucleus_label_image == 1, iterations=number_dilations)
    )
    for granule_idx in np.unique(granule_label_image):
        if granule_idx == 0:
            continue
        granule_mask = granule_label_image == granule_idx
        granule_dilated = ndimage.binary_dilation(granule_mask, iterations=number_dilations)


        data.append({
            'GranuleIndex': granule_idx,
            'IsTouchingNuclearPeriphery': np.sum(np.logical_and(granule_dilated, nucleus_periphery)) > 0,
        })
    return data

def basic_granule_features(granule_label_image):
    data = []
    granule_props = measure.regionprops(granule_label_image)
    if not granule_props:
        return data
    granule_number = len(granule_props) - 1  # Exclude background label (0)
    for granule_lbl in granule_props:
        data.append({
            "GranuleIndex": granule_lbl.label,
            "Area": granule_lbl.area,
            "Perimeter": granule_lbl.perimeter,
            "CentroidY": granule_lbl.centroid[0],
            "CentroidX": granule_lbl.centroid[1],
            "MajorAxisLength": granule_lbl.major_axis_length,
            "MinorAxisLength": granule_lbl.minor_axis_length,
            "Eccentricity": granule_lbl.eccentricity,
            "Orientation": granule_lbl.orientation,
            "GranuleNumberPerCell": granule_number,
        })


    return data

@catch_error(default_value=np.nan)
def morans_i(image):
    """
    Calculate Moran’s I for a 2D image array, ignoring zero pixels outside the mask.
    Returns:
        I (float): Moran’s I statistic (-1 to +1)
   """
    image = np.array(image)
    rows, cols = image.shape

    # Identify non-zero pixels (masked region)
    mask = image != 0
    y_coords, x_coords = np.where(mask)  # Coordinates of non-zero pixels
    n = len(mask[mask > 0].flatten())  # Number of non-zero pixels

    if n == 0:
        return 0.0  # Avoid division by zero

    # Compute mean and variance using ONLY non-zero pixels
    non_zero_pixels = image[mask]
    mu = np.mean(non_zero_pixels)
    denominator = np.sum((non_zero_pixels - mu) ** 2)

    if denominator == 0:
        return 0.0

    # Define neighbor directions (8-connected)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    numerator = 0.0
    W = 0.0  # Total valid neighbor pairs

    # Iterate only over non-zero pixels
    for y, x in zip(y_coords, x_coords):
        zi = image[y, x] - mu

        # Check all 8 neighbors
        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            # Check if neighbor is within bounds AND non-zero
            if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx]:
                zj = image[ny, nx] - mu
                numerator += zi * zj
                W += 1

    if W == 0:
        return 0.0

    # Moran’s I formula (adjusted for non-zero pixels)
    i = (n / W) * (numerator / denominator)
    return i



def intensity_granule_features(granule_label_image, intensity_image):
    """
    Calculate intensity features for each granule in the granule label image.

    Args:
        granule_label_image: 2D array where each unique value represents a different granule.
        intensity_image: 2D array of intensity values corresponding to the granules.

    Returns:
        list: A list of dictionaries with granule index and intensity features.
    """
    data = []
    granule_int_regionprops = measure.regionprops(granule_label_image, intensity_image=intensity_image)

    if not granule_int_regionprops:
        return data

    for granule_lbl in granule_int_regionprops:
        data.append({
            "GranuleIndex": granule_lbl.label,
            "MeanIntensity": granule_lbl.mean_intensity,
            "MinIntensity": granule_lbl.min_intensity,
            "MaxIntensity": granule_lbl.max_intensity,
            "StdIntensity": granule_lbl.intensity_std,
            "WeightedCentroidY": granule_lbl.weighted_centroid[0],
            "WeightedCentroidX": granule_lbl.weighted_centroid[1],
            # "GranuleImage": granule_lbl.image_intensity,
            "MoransI": morans_i(granule_lbl.image_intensity),
            "GranuleSolidity": granule_lbl.solidity,
        })


    return data


def polar_to_fourier_series(binary_mask, n_harmonics=25):
    """
    Generate Fourier series from polar coordinate data (angles and distances).

        :param binary_mask: binary mask of a single blob (granule)
        :param n_harmonics: Number of Fourier harmonics to keep
    """
    # Sort by angle to ensure proper ordering
    contours = measure.find_contours(binary_mask.astype(float), 0.5)

    if len(contours) == 0:
        return np.nan

    # Use the longest contour (main boundary)
    boundary = contours[0]

    # Get centroid
    centroid = ndimage.center_of_mass(binary_mask)
    cy, cx = centroid

    # Extract boundary coordinates
    y_coords = boundary[:, 0]
    x_coords = boundary[:, 1]

    # Calculate distances from centroid to each boundary point
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    # Calculate angles from centroid to each boundary point
    angles = np.arctan2(y_coords - cy, x_coords - cx)

    # Convert angles to degrees
    angles = np.degrees(angles)
    angles = np.mod(angles, 360)  # Normalize angles to [0, 360)

    sorted_indices = np.argsort(angles)
    angles_sorted = angles[sorted_indices]
    distances_sorted = distances[sorted_indices]

    # Interpolate to uniform angular spacing
    uniform_angles = np.linspace(0, 360, len(distances_sorted), endpoint=False)
    uniform_distances = np.interp(uniform_angles, angles_sorted, distances_sorted)

    # Apply FFT
    fourier_coeffs = fft(uniform_distances)
    out_dict = {}

    max_harmonics = min(n_harmonics, len(fourier_coeffs))
    for i in range(max_harmonics):
        # Store both magnitude and phase instead of just real part
        out_dict[f'FourierMagnitudeH{i}'] = np.abs(fourier_coeffs[i])
        out_dict[f'FourierPhaseH{i}'] = np.angle(fourier_coeffs[i])

        # Or store the full complex coefficient
        out_dict[f'FourierCoeffH{i}'] = fourier_coeffs[i]

    return out_dict


def granule_fourier_series(granule_label_image, n_harmonics=50):
    """

    :param n_harmonics:
    :param granule_label_image:
    :return:
    """

    data = []
    # Find the unique labels in the granule segmentation
    unique_labels = np.unique(granule_label_image)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        # Create a binary mask for the current granule
        binary_mask = ()

        # Get Fourier series for the current granule
        fourier_data = polar_to_fourier_series(granule_label_image == label, n_harmonics=n_harmonics)

        # Add label to the output dictionary
        fourier_data['GranuleIndex'] = label

        data.append(fourier_data)
    return data

@catch_error(default_value=np.nan)
def hexagonality(label_image):
    """Compute hexagonality of a segmentation label image.

    For each centroid, the 6 nearest neighbours are found and the consecutive
    angular gaps (which sum to 360°) are compared to the ideal 60° spacing of
    a perfect hexagonal lattice (interior angle 120°).  The per-cell score is
    1 − mean(|gap − 60°|) / 60°, clipped to [0, 1].  The global score is the
    mean across all cells.

    Parameters
    ----------
    label_image : np.ndarray
        2-D integer label image; each unique value > 0 is one segment.

    Returns
    -------
    float
        Hexagonality in [0, 1].  Returns np.nan when fewer than 7 labels are
        present (not enough for one cell with 6 neighbours).
    """
    label_image = np.asarray(label_image, dtype=int)
    props = measure.regionprops(label_image)

    if len(props) < 7:
        return np.nan

    centroids = np.array([p.centroid for p in props])  # (N, 2)  row, col

    tree = cKDTree(centroids)
    # query k=7: index 0 is the point itself, indices 1-6 are the 6 nearest neighbours
    distances, indices = tree.query(centroids, k=7)

    # Estimate lattice spacing as the median nearest-neighbour distance
    nn_dist = distances[:, 1]                          # distance to closest neighbour
    lattice_spacing = np.median(nn_dist)

    scores = []
    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
        # Skip boundary cells: require all 6 neighbours within 1.5× lattice spacing
        if dists[6] > 1.5 * lattice_spacing:
            continue

        neighbours = centroids[nbrs[1:]]              # (6, 2)
        dy = neighbours[:, 0] - centroids[i, 0]
        dx = neighbours[:, 1] - centroids[i, 1]
        angles = np.sort(np.degrees(np.arctan2(dy, dx)) % 360)  # sorted in [0, 360)

        # Consecutive angular gaps including the wrap-around gap
        gaps = np.diff(angles, append=angles[0] + 360)          # sum == 360°

        mean_dev = np.mean(np.abs(gaps - 60.0))
        scores.append(max(0.0, 1.0 - mean_dev / 60.0))

    if not scores:
        return np.nan

    return float(np.mean(scores))


