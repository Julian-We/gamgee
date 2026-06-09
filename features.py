import numpy as np
from scipy import ndimage
import functools
from typing import Callable, Union, Any
from skimage import measure
from scipy.fft import fft
from skimage.measure import regionprops, manders_coloc_coeff, find_contours
from scipy.spatial import cKDTree
from scipy import stats


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
def min_distance_to_object(binary_mask, point, signed=False):
    """
    Calculate minimal distance from a point to a binary blob, optionally signed.

    Args:
        binary_mask: 2D binary array where blob pixels are True/1
        point: tuple (y, x) coordinates of the point

    """

    y, x = int(round(point[0])), int(round(point[1]))

    if signed:
        outward_distance_map = ndimage.distance_transform_edt(~(binary_mask > 0))
        inward_distance_map = ndimage.distance_transform_edt(binary_mask > 0)
        signed_distance_map = outward_distance_map.copy()
        signed_distance_map[binary_mask > 0] = -inward_distance_map[binary_mask > 0]
        return signed_distance_map[y, x]

    return ndimage.distance_transform_edt(~(binary_mask > 0))[y, x]


@catch_error()
def centroid_distance_to_nucleus(mask_of_interest, nucleus_mask, signed=True):
    """
    Calculate the distance of a masks centroid to the nucleus mask, optionally signed.

    """
    centroid = get_centroid(mask_of_interest)
    return min_distance_to_object(nucleus_mask, centroid, signed=signed)


@catch_error()
def edge_distance_to_nucleus(mask_of_interest, nucleus_mask, signed=False):
    if signed:
        outward_distance_map = ndimage.distance_transform_edt(~(nucleus_mask > 0))
        inward_distance_map = ndimage.distance_transform_edt(nucleus_mask > 0)
        signed_distance_map = outward_distance_map.copy()
        signed_distance_map[nucleus_mask > 0] = -inward_distance_map[nucleus_mask > 0]
        return signed_distance_map[mask_of_interest > 0].min()

    return ndimage.distance_transform_edt(~(nucleus_mask > 0))[
        mask_of_interest > 0
    ].min()


def nuclear_distance_features(granule_label_image, nucleus_mask, cell_mask):
    region_props = measure.regionprops(granule_label_image)

    for granule_lbl in region_props:
        granule_mask = granule_label_image == granule_lbl.label
        relative_distance = relative_nuclear_distance(
            get_centroid(granule_mask), nucleus_mask, cell_mask
        )
        relative_distance = (
            relative_distance if isinstance(relative_distance, dict) else {}
        )

        yield {
            "GranuleIndex": granule_lbl.label,
            "EdgeDistanceToNucleus": edge_distance_to_nucleus(
                granule_mask, nucleus_mask
            ),
            "EdgeDistanceToNucleusSigned": edge_distance_to_nucleus(
                granule_mask, nucleus_mask, signed=True
            ),
            "CentroidDistanceToNucleus": centroid_distance_to_nucleus(
                granule_mask, nucleus_mask
            ),
            "CentroidDistanceToNucleusSigned": centroid_distance_to_nucleus(
                granule_mask, nucleus_mask, signed=True
            ),
            "TouchAreaNucleus": touch_area(
                granule_mask, nucleus_mask, number_dilations=1
            ),
            "RelativeNuclearDistance": relative_distance.get(
                "SignedRelativeDistance", np.nan
            ),
            "FranctionAlongRayToCellBoundary": relative_distance.get(
                "NucleusBoundaryFraction", np.nan
            ),
        }


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


def iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.
    Args:
        mask1: First binary mask (2D array)
        mask2: Second binary mask (2D array)
    Returns:
        float: IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


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
    major_axis = region.axis_major_length
    minor_axis = region.axis_minor_length

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
    return (4 / 3) * np.pi * base_radius**3 * aspect_ratio, (
        4 / 3
    ) * np.pi * base_radius**3 / aspect_ratio


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
        ndimage.binary_erosion(nucleus_label_image == 1, iterations=number_dilations),
    )
    for granule_idx in np.unique(granule_label_image):
        if granule_idx == 0:
            continue
        granule_mask = granule_label_image == granule_idx
        granule_dilated = ndimage.binary_dilation(
            granule_mask, iterations=number_dilations
        )

        data.append(
            {
                "GranuleIndex": granule_idx,
                "IsTouchingNuclearPeriphery": np.sum(
                    np.logical_and(granule_dilated, nucleus_periphery)
                )
                > 0,
            }
        )
    return data


def basic_granule_features(granule_label_image):
    data = []
    granule_props = measure.regionprops(granule_label_image)
    if not granule_props:
        return data
    granule_number = len(granule_props) - 1  # Exclude background label (0)
    for granule_lbl in granule_props:
        data.append(
            {
                "GranuleIndex": granule_lbl.label,
                "Area": granule_lbl.area,
                "Perimeter": granule_lbl.perimeter,
                "CentroidY": granule_lbl.centroid[0],
                "CentroidX": granule_lbl.centroid[1],
                "MajorAxisLength": granule_lbl.axis_major_length,
                "MinorAxisLength": granule_lbl.axis_minor_length,
                "Eccentricity": granule_lbl.eccentricity,
                "Orientation": granule_lbl.orientation,
                "GranuleNumberPerCell": granule_number,
            }
        )

    return data


@catch_error(default_value=np.nan)
def morans_i(image, mask):
    """
    Calculate Moran’s I for a 2D image array, ignoring zero pixels outside the mask.
    Returns:
        I (float): Moran’s I statistic (-1 to +1)
    """
    image = np.array(image)
    rows, cols = image.shape

    # Identify non-zero pixels (masked region)
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
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

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
    granule_int_regionprops = measure.regionprops(
        granule_label_image, intensity_image=intensity_image
    )

    if not granule_int_regionprops:
        return data

    for granule_lbl in granule_int_regionprops:
        cropped_intensity_values = granule_lbl.image_intensity[
            granule_lbl.image > 0
        ].flatten()
        data.append(
            {
                "GranuleIndex": granule_lbl.label,
                "MeanIntensity": granule_lbl.mean_intensity,
                "MinIntensity": granule_lbl.min_intensity,
                "MaxIntensity": granule_lbl.max_intensity,
                "StdIntensity": granule_lbl.intensity_std,
                "WeightedCentroidY": granule_lbl.weighted_centroid[0],
                "WeightedCentroidX": granule_lbl.weighted_centroid[1],
                # "GranuleImage": granule_lbl.image_intensity,
                "MoransI": morans_i(granule_lbl.image_intensity, granule_lbl.image),
                "GranuleSolidity": granule_lbl.solidity,
                "GranuleSkewness": stats.skew(cropped_intensity_values),
                "GranuleKurtosis": stats.kurtosis(cropped_intensity_values),
                "GranuleEntropy": stats.entropy(cropped_intensity_values + 1e-10),
                "GranuleCV": np.std(cropped_intensity_values)
                / (np.mean(cropped_intensity_values) + 1e-10),
            }
        )

    return data


def advanced_granule_features(granule_label_image):
    """
    Calculate advanced features for each granule in the granule label image, such as solidity and convexity and the fourfourier series of the granule boundary.
    """
    data = []
    region_props = measure.regionprops(granule_label_image)

    fourier_series_data = granule_fourier_series(granule_label_image, n_harmonics=25)

    for granule_lbl in region_props:
        granule_data = {
            "GranuleIndex": granule_lbl.label,
            "GranuleSolidity": granule_lbl.solidity,
            "GranuleConvexity": granule_lbl.convex_area / granule_lbl.area
            if granule_lbl.area > 0
            else np.nan,
            "SphericalVolume": spherical_volume(granule_lbl.image),
            "EllipsoidVolumeProlate": ellipsoid_volume(granule_lbl.image)[0],
            "EllipsoidVolumeOblate": ellipsoid_volume(granule_lbl.image)[1],
        }

        # Add Fourier series data for this granule
        fourier_data = next(
            (
                item
                for item in fourier_series_data
                if item["GranuleIndex"] == granule_lbl.label
            ),
            None,
        )
        if fourier_data:
            granule_data.update(fourier_data)

        data.append(granule_data)
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
        out_dict[f"FourierMagnitudeH{i}"] = np.abs(fourier_coeffs[i])
        out_dict[f"FourierPhaseH{i}"] = np.angle(fourier_coeffs[i])

        # Or store the full complex coefficient
        out_dict[f"FourierCoeffH{i}"] = fourier_coeffs[i]

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
        fourier_data = polar_to_fourier_series(
            granule_label_image == label, n_harmonics=n_harmonics
        )

        # Add label to the output dictionary
        fourier_data["GranuleIndex"] = label

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
    nn_dist = distances[:, 1]  # distance to closest neighbour
    lattice_spacing = np.median(nn_dist)

    scores = []
    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
        # Skip boundary cells: require all 6 neighbours within 1.5× lattice spacing
        if dists[6] > 1.5 * lattice_spacing:
            continue

        neighbours = centroids[nbrs[1:]]  # (6, 2)
        dy = neighbours[:, 0] - centroids[i, 0]
        dx = neighbours[:, 1] - centroids[i, 1]
        angles = np.sort(np.degrees(np.arctan2(dy, dx)) % 360)  # sorted in [0, 360)

        # Consecutive angular gaps including the wrap-around gap
        gaps = np.diff(angles, append=angles[0] + 360)  # sum == 360°

        mean_dev = np.mean(np.abs(gaps - 60.0))
        scores.append(max(0.0, 1.0 - mean_dev / 60.0))

    if not scores:
        return np.nan

    return float(np.mean(scores))


def threshold_parameters(image_of_interest, confining_segmentation, percentile):

    try:
        binary_image = image_of_interest > np.percentile(
            image_of_interest[confining_segmentation > 0], percentile
        )
        binary_image = ndimage.binary_opening(binary_image)

        return {
            f"percentile{percentile}_area": np.sum(binary_image),
            f"percentile{percentile}_mean_intensity": np.mean(
                image_of_interest[binary_image]
            ),
            f"percentile{percentile}_sum_intensity": np.min(
                image_of_interest[binary_image]
            ),
        }
    except Exception as e:
        return {
            f"percentile{percentile}_area": np.nan,
            f"percentile{percentile}_mean_intensity": np.nan,
            f"percentile{percentile}_sum_intensity": np.nan,
        }


@catch_error()
def manders_across_percentiles(imgA, imgB, step=0.1, mask=None):
    percentile_data = []
    for percentile in np.arange(0, 100 + step, step):
        if mask is None:
            threshold_A = np.percentile(imgA, percentile)
            threshold_B = np.percentile(imgB, percentile)
        else:
            threshold_A = np.percentile(imgA[mask > 0], percentile)
            threshold_B = np.percentile(imgB[mask > 0], percentile)

        binary_A = imgA > threshold_A
        binary_B = imgB > threshold_B

        m1 = manders_coloc_coeff(imgA, binary_B, mask=mask)
        m2 = manders_coloc_coeff(imgB, binary_A, mask=mask)

        percentile_data.append(
            {
                "Percentile": percentile,
                "M1": m1,
                "M2": m2,
            }
        )
    return percentile_data


@catch_error()
def manders(img, segmentation, mask=None):
    return manders_coloc_coeff(img, segmentation > 0, mask=mask)


@catch_error()
def pearsons(imgA, imgB, mask=None):
    if mask is not None:
        imgA = imgA[mask > 0]
        imgB = imgB[mask > 0]

    if len(imgA) == 0 or len(imgB) == 0:
        return np.nan

    return stats.pearsonr(imgA.flatten(), imgB.flatten())[0]


@catch_error()
def relative_nuclear_distance(
    point,
    nucleus_mask: np.ndarray,
    cell_mask: np.ndarray,
) -> dict:
    """
    Compute where a point lies along the axis from the nucleus centroid outward
    to the cell boundary, normalised so that:

        -1  → nucleus centroid
         0  → nucleus / cytoplasm boundary  (ray exits nucleus)
        +1  → cell boundary  (ray exits cell)

    Points inside the nucleus return values in [-1, 0].
    Points in the cytoplasm return values in [0, 1].

    Parameters
    ----------
    point : (row, col) in the same pixel coordinate space as the masks.
    nucleus_mask : 2-D boolean / uint array; non-zero pixels belong to the nucleus.
    cell_mask    : 2-D boolean / uint array; non-zero pixels belong to the cell
                   (must contain the nucleus region).

    Returns
    -------
    dict with keys:
        "SignedRelativeDistance"
            Piecewise-linear value in [-1, +1] as described above.
            Returns np.nan if the point lies outside the cell mask.

        "NucleusBoundaryFraction"
            d_nuc / (d_nuc + d_cyto):  the fraction of the total ray length
            (centroid → cell boundary) that is occupied by the nucleus segment.
            Equivalent to your original NucleusToCellBoundaryPercent.

        "NucleusBoundaryIntersection"
            (row, col) float – where the ray crosses the nucleus boundary.

        "CellBoundaryIntersection"
            (row, col) float – where the ray crosses the cell boundary.
    """
    point = np.asarray(point, dtype=float)

    # ------------------------------------------------------------------
    # 1.  Nucleus centroid
    # ------------------------------------------------------------------
    nuc_props = regionprops((nucleus_mask > 0).astype(np.uint8))
    if not nuc_props:
        raise ValueError("nucleus_mask contains no labelled region.")
    centroid = np.array(nuc_props[0].centroid, dtype=float)  # (row, col)

    # ------------------------------------------------------------------
    # 2.  Ray direction: centroid → point
    #     If the point IS the centroid we cannot define a direction; fall back
    #     to a zero-length result of exactly -1.
    # ------------------------------------------------------------------
    direction = point - centroid
    dist_to_point = np.linalg.norm(direction)

    if dist_to_point < 1e-9:
        # Point is at the centroid → deepest interior value
        return {
            "SignedRelativeDistance": -1.0,
            "NucleusBoundaryFraction": np.nan,
            "NucleusBoundaryIntersection": tuple(centroid),
            "CellBoundaryIntersection": tuple(centroid),
        }

    unit = direction / dist_to_point

    # ------------------------------------------------------------------
    # 3.  Helper: find the first intersection of the ray with a binary mask's
    #     boundary contour, marching outward from `start` in `direction`.
    # ------------------------------------------------------------------
    def _ray_mask_intersection(
        mask: np.ndarray, start: np.ndarray, unit_vec: np.ndarray
    ) -> np.ndarray:
        """
        Walk along the ray in sub-pixel steps and return the (row, col) of the
        first pixel that transitions from inside → outside (or outside → inside)
        the mask.  Falls back to contour-segment intersection for sub-pixel accuracy.
        """
        h, w = mask.shape

        # Coarse march to find approximate crossing distance
        step = 0.5  # pixels
        max_steps = int(np.hypot(h, w) / step) + 10
        prev_inside = _sample_mask(mask, start, h, w)

        crossing_dist = None
        for i in range(1, max_steps):
            pos = start + unit_vec * (i * step)
            inside = _sample_mask(mask, pos, h, w)
            if inside != prev_inside:
                crossing_dist = (i - 0.5) * step
                break
            prev_inside = inside

        if crossing_dist is None:
            # Ray never crosses – return a far-away fallback
            return start + unit_vec * max_steps * step

        # Refine with binary search
        lo, hi = (crossing_dist - step), crossing_dist
        for _ in range(20):
            mid = (lo + hi) / 2
            if _sample_mask(mask, start + unit_vec * mid, h, w) == prev_inside:
                lo = mid
            else:
                hi = mid
        return start + unit_vec * ((lo + hi) / 2)

    def _sample_mask(mask, pos, h, w):
        r, c = int(round(pos[0])), int(round(pos[1]))
        r = max(0, min(h - 1, r))
        c = max(0, min(w - 1, c))
        return mask[r, c] > 0

    # ------------------------------------------------------------------
    # 4.  Find the two boundary crossings
    # ------------------------------------------------------------------
    nuc_boundary_pt = _ray_mask_intersection(nucleus_mask > 0, centroid, unit)
    cell_boundary_pt = _ray_mask_intersection(cell_mask > 0, centroid, unit)

    d_nuc = np.linalg.norm(nuc_boundary_pt - centroid)
    d_total = np.linalg.norm(cell_boundary_pt - centroid)
    d_cyto = d_total - d_nuc

    # Guard against degenerate geometry
    if d_nuc < 1e-9 or d_cyto < 1e-9:
        nuc_boundary_fraction = np.nan
    else:
        nuc_boundary_fraction = d_nuc / d_total

    # ------------------------------------------------------------------
    # 5.  Signed relative distance
    # ------------------------------------------------------------------
    point_inside_cell = _sample_mask(cell_mask > 0, point, *cell_mask.shape)
    if not point_inside_cell:
        signed = np.nan
    elif d_nuc < 1e-9:
        signed = np.nan
    else:
        point_inside_nucleus = _sample_mask(
            nucleus_mask > 0, point, *nucleus_mask.shape
        )
        if point_inside_nucleus:
            # Map [centroid … nucleus boundary] → [-1 … 0]
            signed = -1.0 + dist_to_point / d_nuc
            signed = float(np.clip(signed, -1.0, 0.0))
        else:
            # Map [nucleus boundary … cell boundary] → [0 … 1]
            if d_cyto < 1e-9:
                signed = 0.0
            else:
                signed = (dist_to_point - d_nuc) / d_cyto
                signed = float(np.clip(signed, 0.0, 1.0))

    return {
        "SignedRelativeDistance": signed,
        "NucleusBoundaryFraction": float(nuc_boundary_fraction)
        if not np.isnan(nuc_boundary_fraction)
        else np.nan,
        "NucleusBoundaryIntersection": tuple(nuc_boundary_pt),
        "CellBoundaryIntersection": tuple(cell_boundary_pt),
    }


def granule_to_cytoplasm_intensity_ratio(
    intensity_image, granule_mask, nucleus_mask, cell_mask
):
    """
    Calculate granule to cytoplasm intensity ratio by dividing the mean intensity of the granule
    """
    granule_mask = granule_mask > 0
    cell_mask = cell_mask > 0
    nucleus_mask = nucleus_mask > 0

    # Get cytoplamic mask by subtracting granules+nucleus from cell mask
    nuclues_and_granules_mask = np.logical_or(nucleus_mask, granule_mask)
    cytoplasm_mask = np.logical_and(cell_mask, ~nuclues_and_granules_mask)

    return np.mean(intensity_image[granule_mask]) / (
        np.mean(intensity_image[cytoplasm_mask]) + 1e-15
    )
