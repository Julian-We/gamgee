import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

def merge_segmentations(label_img, threshold=0.25, connectivity=1, min_shared=1):
    """
    Merge labels in `label_img` when two objects touch and share >= `threshold`
    fraction of the smaller object's boundary.

    Parameters
    - label_img: 2D or 3D integer numpy array with 0 as background.
    - threshold: fraction (0..1) of the smaller boundary that must be shared to fuse.
    - connectivity: connectivity for morphological ops (1 or the full connectivity).
    - min_shared: minimum number of touching boundary pixels/voxels to consider a merge.

    Returns
    - new_labels: relabeled numpy array with fused objects.
    - merges: list of tuples (kept_label, absorbed_label).
    """
    img = label_img.copy()
    shape = img.shape
    structure = ndi.generate_binary_structure(img.ndim, connectivity)
    changed = True
    merges = []

    while changed:
        changed = False
        labels = np.unique(img)
        labels = labels[labels != 0]

        # Precompute masks, eroded masks, boundaries and boundary sizes
        masks = {}
        boundaries = {}
        bcounts = {}
        for L in labels:
            m = img == L
            if not m.any():
                continue
            eroded = ndi.binary_erosion(m, structure=structure, border_value=0)
            boundary = m & (~eroded)
            bc = int(boundary.sum())
            masks[L] = m
            boundaries[L] = boundary
            bcounts[L] = bc if bc > 0 else int(m.sum())  # fallback

        # Check pairs for touching
        checked_pairs = set()
        for a in labels:
            if a not in boundaries:
                continue
            for b in labels:
                if b <= a or b not in boundaries:
                    continue
                pair = (a, b)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Dilate boundary of a and intersect with boundary of b to get contact length
                dil_a = ndi.binary_dilation(boundaries[a], structure=structure)
                contact = dil_a & boundaries[b]
                shared = int(contact.sum())
                if shared < min_shared:
                    # try the other way (in case of anisotropic shapes)
                    dil_b = ndi.binary_dilation(boundaries[b], structure=structure)
                    shared = int((dil_b & boundaries[a]).sum())

                if shared >= min_shared:
                    smaller = min(bcounts[a], bcounts[b])
                    if smaller <= 0:
                        continue
                    frac = shared / float(smaller)
                    if frac >= threshold:
                        # fuse b into a (keep a)
                        img[img == b] = a
                        merges.append((a, b))
                        changed = True
                        # after a merge, break to restart pairs because boundaries changed
                        break
            if changed:
                break

    # Optionally relabel to compact integers (scipy label)
    unique = np.unique(img)
    unique = unique[unique != 0]
    new_map = {old: i + 1 for i, old in enumerate(unique)}
    new_img = np.zeros_like(img, dtype=np.int32)
    for old, new in new_map.items():
        new_img[img == old] = new

    # update merges to mapped indices
    mapped_merges = []
    for kept, absorbed in merges:
        if kept in new_map and absorbed in new_map:
            mapped_merges.append((new_map[kept], new_map[absorbed]))
        elif kept in new_map and absorbed not in new_map:
            mapped_merges.append((new_map[kept], None))
        else:
            mapped_merges.append((None, None))

    return new_img, mapped_merges

# Example usage (commented):
# new_labels, merges = merge_touching_segments(label_array, threshold=0.25)


def clean_cell_segmentations(cell_segmentation):
    """
    Clean cell segmentations by merging touching objects and selecting the one
    closest to the image center.
    Args:
        cell_segmentation: Binary numpy array of cell segmentation.

    Returns:

    """
    segmentation = cell_segmentation.copy()
    lbl = label(segmentation)

    if len(np.unique(lbl)) == 2:
        return segmentation
    elif len(np.unique(lbl)) < 2:
        raise ValueError("No objects found in the segmentation.")
    else:
        merged_segmentations = merge_segmentations(segmentation)

        # Get center of the image
        y_dim, x_dim = segmentation.shape
        center_y, center_x = y_dim // 2, x_dim // 2
        min_distance = None
        best_region = None
        # Select the object closest to the center
        for region in regionprops(merged_segmentations):
            cy, cx = region.centroid
            distance = np.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_region = region
        if best_region is not None:
            cleaned_segmentation = np.zeros_like(segmentation)
            cleaned_segmentation[merged_segmentations == best_region.label] = 1
            return cleaned_segmentation
        else:
            raise ValueError("No suitable object found in the segmentation.")


def delete_outside_objects(segmentation_of_interest, confining_segmentation):
    soi = segmentation_of_interest.copy()
    confining = confining_segmentation.copy()

    soi_labels = label(soi)
    confining_labels = label(confining)

    for region in regionprops(soi_labels):
        minr, minc, maxr, maxc = region.bbox
        # Check if any part of the region is within the confining segmentation
        if np.sum(confining_labels[minr:maxr, minc:maxc][soi_labels[minr:maxr, minc:maxc] == region.label]) == 0:
            # If not, remove the region
            soi_labels[soi_labels == region.label] = 0

    return soi_labels.astype(np.uint8)