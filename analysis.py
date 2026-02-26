import sys
from features import edge_distance_to_nucleus, spherical_volume
import numpy as np
sys.path.append("/Users/icb_remote/Documents/JW/py/packages/")
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import datetime
from skimage.measure import regionprops, label
from gamgee.instance.segmentations import delete_outside_objects, merge_segmentations
from gamgee import features
from multiprocessing import Pool


def distance(centroid_l, center_l):
    return abs(np.sqrt((centroid_l[0]- center_l[0])**2 + (centroid_l[1] - center_l[1])**2))

def catch_error(func):
    """
    Catches errors and puts the into the logbook. If the object (self) in the function self.lock == True then just self.log() that the function could not be performed. If an error occurs, log that error and set self.lock to True
    """
    def wrapper(self, *args, **kwargs):
        if self.lock:
            self.log(f"Sample is locked. Cannot perform function {func.__name__}.")
            return
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.log(f"Error in function {func.__name__}: {str(e)}")
            print(f"Error in function {func.__name__}: {str(e)}")
            self.lock = True
    return wrapper

class Sample:
    def __init__(self, path: str | Path,  **kwargs):
        # Generate for image a 
        self.lock = False
        self.uid = None
        self.images = {}
        self.denoised_images = {}
        self.segmentations = {}
        self.markers = []
        self.compartments = {}

        self.logs = {}
        self.root_dir = Path(path) if not isinstance(path, Path) else path
        self._unupickle()
        self.clean_up_segmentations()

        self.per_granule_results = []
        self.detailed_results = []

    def log(self, message: str):
        self.logs[datetime.datetime.now().isoformat()] = message

    @catch_error
    def clean_up_segmentations(self):
        """
        Starting with the nucleus segmentation, one cell is selected (the most center one if multiple nuclei are detected). 
        Then the cell segmentation is chosen that contains the centroid of the nucleus
        For each other segmentation delete the segmentation outside the cell segmentation
        """
        nucleus_markers = [mrk for mrk, cmp in self.compartments.items() if "nucleus" in cmp.lower()]
        if nucleus_markers != []:
            # Get number of segmentations
            marker = nucleus_markers[0]
            self.segmentations[marker], change_map = merge_segmentations(self.segmentations[marker])
            if change_map:
                self.log(f"Segmentations for marker {marker} were merged. {len(change_map)} segmentations were merged into {len(np.unique(self.segmentations[marker])) - 1} segmentations.")
            number_of_nuc_segmentations = np.unique(self.segmentations[marker]).shape[0] - 1

            if number_of_nuc_segmentations > 1:
                self.log(f"Multiple nuclei detected for marker {marker}. Only the most center one is kept.")

                # Get the most center segmentation
                center = np.array(self.images[marker].shape) / 2
                centroids = {}
                min_centroid = (np.inf, np.inf)
                for i in np.unique(self.segmentations[marker]):
                    if i == 0:
                        continue
                    mask = self.segmentations[marker] == i
                    if np.sum(mask) == 0:
                        continue
                    centroid = np.array(np.nonzero(mask)).mean(axis=1)
                    
                    centroids[tuple(centroid)] = i

                    if distance(centroid, center) < distance(min_centroid, center):
                        min_centroid = centroid
                self.segmentations[marker][self.segmentations[marker] != centroids[tuple(min_centroid)]] = 0
            else:
                self.log("Only one nucleus segmentation found")
                self.segmentations[marker] = self.segmentations[marker].astype('uint8')
                min_centroid = np.array(np.nonzero(self.segmentations[marker])).mean(axis=1)
        else: 
            raise ValueError(f"Nucleus marker not found in <{self.root_dir.name}>")


        cell_markers = [mrk for mrk, cmp in self.compartments.items() if "cell" in cmp.lower()]
        if cell_markers != []:
            for marker in cell_markers:
                if marker in self.segmentations:
                    # Get the segmentation that contains the centroid of the nucleus
                    self.segmentations[marker], _ = merge_segmentations(self.segmentations[marker])
                else:
                    raise ValueError(f"Cell segmentation for marker {marker} not found in <{self.root_dir.name}>")
                 
                # Find segmentation that contains min_centroid and delete all other cell segmentations
                cell_segmentation = None
                for i in np.unique(self.segmentations[marker]):
                    mask = self.segmentations[marker] == i
                    if np.sum(mask) == 0:
                        continue
                    if mask[int(min_centroid[0]), int(min_centroid[1])]:
                        cell_segmentation = i
                        break
                if not cell_segmentation:
                    raise ValueError(f"No cell segmentation contains the nucleus centroid for marker {marker} in <{self.root_dir.name}>")

                self.segmentations[marker] = delete_outside_objects(self.segmentations[marker], self.segmentations[marker] == cell_segmentation)
        
        other_markers = [mrk for mrk in self.markers if mrk not in cell_markers and mrk not in nucleus_markers]
        for marker in other_markers:
            if marker in self.segmentations:
                self.segmentations[marker] = delete_outside_objects(self.segmentations[marker], self.segmentations[cell_markers[0]] > 0)
            else:
                raise ValueError(f"Segmentation for marker {marker} not found in <{self.root_dir.name}>")


    def _unupickle(self):
        # Search for pickle files in root directory
        
        # Check of there is a output folder
        possible_out_names = ["out", "output", "export", "exp"]

        contains_output_folder = False

        for name in possible_out_names:
            if (self.root_dir / name).exists():
                contains_output_folder = True
                break

        pickle_file_canditates = []
        if not contains_output_folder:
           # Search for pickle file containering "images" in root dir
            for file in self.root_dir.iterdir():
                if file.suffix == ".pkl" and "image" in file.stem.lower():
                    pickle_file_canditates.append(file)
        else:
            for out_name in possible_out_names:
                if (self.root_dir / out_name).exists():
                    # Search for pickle file containering "images" in output folder
                    for file in (self.root_dir / out_name).iterdir():
                        if file.suffix == ".pkl" and "image" in file.stem.lower():
                            pickle_file_canditates.append(file)

        if len(pickle_file_canditates)  < 1:
            raise FileNotFoundError("No pickle file containing 'image' found in the root directory or output folder.")
        elif len(pickle_file_canditates) > 1:
            print(f"Multiple pickle files containing 'image' found. First on in the list ist chosen <{pickle_file_canditates[0].name}>")
            pickle_file_path = pickle_file_canditates[0]
        else:
            pickle_file_path = pickle_file_canditates[0]

        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
            self.uid = data.pop("uid", None)
            if not self.uid:
                raise ValueError(f"No unique identifier passed in pickle file <{pickle_file_path.name}>")
            for marker, makrer_dict in data.items():
                self.markers.append(marker)
                self.images[marker] = makrer_dict.get("raw_image", None)
                self.denoised_images[marker] = makrer_dict.get("denoised_image", None)
                self.segmentations[marker] = makrer_dict.get("segmentation", None)
                self.compartments[marker] = makrer_dict.get("compartment", None)
            self.log(f"Data loaded from pickle file <{pickle_file_path.name}>")

    def plot_images_and_segmentations(self, ax):
        if self.lock:
            self.log("Sample is locked. Cannot plot images and segmentations.")
            return 
        nls_markers = [mrk for mrk in self.markers if "nls" in mrk]
        other_markers = [mrk for mrk in self.markers if mrk not in nls_markers]
        total_channels = len(other_markers) + 1
        if len(ax) != total_channels:
            self.log(f"Number of subplots {len(ax)} does not match number of channels {total_channels} for sample <{self.root_dir.name}>")
        i = 0
        for i, marker in enumerate(other_markers):
            ax[i].imshow(self.images[marker], cmap="gray")
            ax[i].set_title(f"{marker} - {self.compartments[marker]}")
            if self.segmentations[marker] is not None:
                ax[i].contour(self.segmentations[marker], colors='r', linewidths=0.5)
        ax[i+1].imshow(self.images[nls_markers[0]], cmap="gray")
        ax[i+1].set_title(f"{nls_markers[0]} - {self.compartments[nls_markers[0]]}")
        for marker in nls_markers:
            if self.segmentations[marker] is not None:
                ax[i+1].contour(self.segmentations[marker], colors='r', linewidths=0.5)
    
    def extract_features(self):
        """
        Extract features for each granule. For each granule, the following features are extracted:
        """

        if self.lock:
            self.log("Sample is locked. Cannot extract features.")
            return
        
    
        # Get the name of the nucleus marker
        nucleus_marker = next((mrk for mrk, cmp in self.compartments.items() if "nucleus" in cmp.lower()), None)
        nucleus_mask = self.segmentations[nucleus_marker] > 0

        for marker in self.markers:
            lbl = self.segmentations[marker]
            region_props = regionprops(lbl, intensity_image=self.images[marker])
            
            # Marker wide values
            number_of_segmentations = len(region_props)
            hexagonality = features.hexagonality(lbl)

            for region in region_props:
                marker_result_dict = {
                    "marker_name": marker,
                    "compartment": self.compartments[marker],
                    "hexagonality": hexagonality
                }
                
                masked_intensity_img = self.images[marker][lbl == region.label]
                masked_segmentation = lbl[lbl == region.label]
                marker_result_dict.update({
                    "area" : region.area,
                    "centroid_x" : region_props.centroid[0],
                    "centroid_y" : region_props.centroid[1],
                    "weighted_centroid_x" : region_props.weighted_centroid[0],
                    "weighted_centroid_y" : region_props.weighted_centroid[1],
                    "aspect_ratio" : region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0,
                    "eccentricity" : region.eccentricity,
                    "orientation" : region.orientation,
                    "euler_number" : region.euler_number,
                    "extent" : region.extent,
                    "intensity_max" : region.intensity_max,
                    "intensity_mean" : region.intensity_mean,
                    "intensity_min" : region.intensity_min,
                    "intensity_median": region.intensity_median,
                    "intensity_std" : region.intensity_std,
                    "perimeter" : region.perimeter,
                    "spherical_volume": features.spherical_volume(masked_segmentation),
                    "ellipsoid_volume_minor": features.ellipsoid_volume(masked_segmentation)[1],
                    "ellipsoid_volume_major": features.ellipsoid_volume(masked_segmentation)[0],
                    "segmentation_count": number_of_segmentations,
                    "morans_i": features.morans_i(masked_intensity_img)
                })

                marker_result_dict.update(features.polat_to_fourier_series(masked_segmentation))

                if "granule" in self.compartments[marker].lower():
                    marker_result_dict.update({
                        "touch_area_nucleus": features.touch_area(masked_segmentation, nucleus_mask),
                        "centroid_distance_to_nucleus": features.centroid_distance_to_nucleus(masked_segmentation, nucleus_mask),
                        "edge_distance_to_nucleus": features.edge_distance_to_nucleus(masked_segmentation, nucleus_mask)
                    })

                self.detailed_results.append(marker_result_dict)
