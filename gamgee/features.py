import numpy as np
from scipy import ndimage
import functools
from typing import Callable, Union, Any
from skimage import measure
from scipy.fft import fft
from skimage.measure import regionprops

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
    min_distance = distance_map[y, x]

    return min_distance

def distance_to_nucleus(granule_label_img, nucleus_label_img ):
    distance_data = []
    for granule_idx in np.unique(granule_label_img):
        if granule_idx == 0:  # Skip background
            continue

        # Get coordinates of the granule
        # granule_coords = np.argwhere(granule_label_img == granule_idx)

        granule_coords = get_centroid(granule_label_img == granule_idx)
        # Calculate distance to nucleus for each pixel in the granule

        y, x = granule_coords
        distance = min_distance_to_object(nucleus_label_img > 0, (y, x))
        distance_data.append({
            'GranuleIndex': granule_idx,
            'CentroidY': y,
            'CentroidX': x,
            'DistanceToNucleus': distance
        })
    return distance_data


def touch_area(granule_label_image, nucleus_label_image, number_dilations):
        data = []
        nucleus_dilated = ndimage.binary_dilation(nucleus_label_image == 1, iterations=number_dilations)
        for granule_idx in np.unique(granule_label_image):
            if granule_idx == 0:
                continue
            granule_mask = granule_label_image == granule_idx
            granule_dilated = ndimage.binary_dilation(granule_mask, iterations=number_dilations)
            land = np.logical_and(granule_dilated, nucleus_dilated)

            data.append({
                'GranuleIndex': granule_idx,
                'NuclearTouchArea': np.sum(land),
                'NucleusIsTouching': True if np.sum(land) > 0 else False
            })
        return data

def spherical_volume(granule_label_image):
    """
    Calculate the volume of a sphere given its surface area.
    :param granule_label_image:
    :return:
    """
    data = []
    for granule_idx in np.unique(granule_label_image):
        if granule_idx == 0:
            continue
        granule_mask = granule_label_image == granule_idx
        area = np.sum(granule_mask)
        (4 / 3) * np.pi * (np.sqrt(area / np.pi)) ** 3
        data.append({
            'GranuleIndex': granule_idx,
            'SphericalVolume': (4 / 3) * np.pi * (np.sqrt(area / np.pi)) ** 3
        })
    return data






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
def ellipsoid_volume(granule_label_image):
    """
    Calculate the volume of an ellipsoid given its surface area and aspect ratio.
    """
    data = []

    for granule_idx in np.unique(granule_label_image):
        if granule_idx == 0:
            continue

        granule_mask = granule_label_image == granule_idx
        area = np.sum(granule_mask)

        # Get aspect ratio
        aspect_ratio = get_aspect_ratio(granule_mask)

        if np.isnan(aspect_ratio):
            continue

        # Calculate ellipsoid volumes
        base_radius = np.sqrt(area / (np.pi * aspect_ratio))

        data.append({
            'GranuleIndex': granule_idx,
            'EllipsoidVolumeMajor': (4 / 3) * np.pi * base_radius ** 3 * aspect_ratio,
            'EllipsoidVolumeMinor': (4 / 3) * np.pi * base_radius ** 3 / aspect_ratio
        })

    return data

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


def polar_to_fourier_series(binary_mask, n_harmonics=50):
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