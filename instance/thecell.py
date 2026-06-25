import sys
import copy
from .instance import Marker
import gamgee.features as features
import tifffile as tiff
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import scipy.ndimage as ndi
from skimage import measure


class TheCell:
    def __init__(
        self,
        path: str | Path,
        full_auto=False,
        conditions=[],
        model_handler=None,
        model_handler_id: str | None = None,
        granuleA="dnd1",
        granuleB="gra",
    ):
        self.logs = {}
        path = Path(path) if isinstance(path, str) else path
        self.path = path
        full_name = path.name
        uid, file_name = full_name.split("__", 1)
        file_name_splits = file_name.split("_")
        # Find date structure in the splits (YYYYMMDD or YYYY-MM-DD)

        date_str: str = ""
        cell_condition: str = ""
        dev_stage: str = "24hpf"  # Defaulting to 24hpf
        for split in file_name_splits:
            if re.match(r"\d{8}", split) or re.match(r"\d{4}-\d{2}-\d{2}", split):
                date = (
                    datetime.strptime(split, "%Y%m%d")
                    if len(split) == 8
                    else datetime.strptime(split, "%Y-%m-%d")
                )
                # Date to iso
                date_str = date.isoformat()
                continue
            for condition in conditions:
                if condition.lower() in split.lower():
                    cell_condition = condition

            if "hpf" in split.lower():
                dev_stage = split
        if cell_condition == "":
            for condition in conditions:
                if condition.lower() in file_name.lower():
                    cell_condition = condition

        self.uid = uid
        self.acquisition_date = date_str
        self.condition = cell_condition
        self.stage = dev_stage
        self.name = file_name
        self.features: dict = {}

        self.granuleA = granuleA
        self.granuleB = granuleB

        self.log(
            f"Initialized cell with UID: {self.uid}, acquisition date: {self.acquisition_date}, condition: {self.condition}"
        )

        self.model_handler_id = model_handler_id
        self.markers = self._populate_markers(model_handler)

    def attach_model_handler(self, model_handler, model_handler_id: str | None = None):
        if model_handler_id is not None:
            self.model_handler_id = model_handler_id
        for marker in self.markers.values():
            if hasattr(marker, "sam_model"):
                marker.set_sam_model(model_handler)

    def __getstate__(self):
        state = self.__dict__.copy()
        markers = state.get("markers")
        if markers:
            stripped_markers = {}
            for name, marker in markers.items():
                marker_copy = copy.copy(marker)
                # for k, v in marker_copy.__dict__.items():
                #     print(f"{k}\t {v}")
                #     print("\n")
                if hasattr(marker_copy, "sam_model"):
                    marker_copy.sam_model = None
                stripped_markers[name] = marker_copy
            state["markers"] = stripped_markers
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "model_handler_id"):
            self.model_handler_id = None

    def log(self, message: str):
        self.logs[datetime.now().isoformat()] = f"THECELL: {message}"

    def print_logs(self):
        for timestamp, message in self.logs.items():
            print(f"{timestamp}: {message}")

    def _populate_markers(self, model_handler):
        images_dir = self.path / "images"
        return {
            self.granuleA: Marker(
                image_path=images_dir / f"{self.granuleA}.tif",
                parent_name=self.name,
                parent_id=self.uid,
                model_handler=model_handler,
                compartment="granules",
            ),
            self.granuleB: Marker(
                image_path=images_dir / f"{self.granuleB}.tif",
                parent_name=self.name,
                parent_id=self.uid,
                model_handler=model_handler,
                compartment="granules",
            ),
            "nucleus": Marker(
                image_path=images_dir / "nls.tif",
                parent_name=self.name,
                parent_id=self.uid,
                model_handler=model_handler,
                compartment="nucleus",
            ),
            "cell": Marker(
                image_path=images_dir / "nls.tif",
                parent_name=self.name,
                parent_id=self.uid,
                model_handler=model_handler,
                compartment="cell",
            ),
        }

    def write_segmentations(self):
        """
        Write segmenatations from TheCell to disk. Funtion mean to be used befor screening and refining segmentations
        """
        mask_dir = self.path / "masks"
        mask_dir.mkdir(exist_ok=True)

        for marker_name, marker in self.markers.items():
            output_path = mask_dir / f"{marker_name}.tif"
            tiff.imwrite(output_path, marker.segmentation.astype(np.uint16))

    def read_segmentations(self):
        """
        Read segmentations from disk to TheCell. Function meant to be used after screening and refining segmentations
        """
        mask_dir = self.path / "masks"

        for marker_name, marker in self.markers.items():
            segmentation_path = mask_dir / f"{marker_name}.tif"
            if segmentation_path.exists():
                # print("Got segmentation")
                marker.segmentation = measure.label(tiff.imread(segmentation_path))

    def plot_markers_on_axis(
        self,
        ax: np.ndarray,
        blind=True,
        granule_percentile=99,
        segmentation_cmap="nipy_spectral",
        granule_alpha=0.4,
    ):

        if ax.shape != (3,):
            return

        ax[0].imshow(self.markers["nucleus"].raw_image, cmap="gray")
        ax[1].imshow(
            self.markers[self.granuleA].raw_image,
            cmap="gray",
            vmax=np.percentile(
                self.markers[self.granuleA].raw_image, granule_percentile
            ),
        )
        ax[2].imshow(
            self.markers[self.granuleB].raw_image,
            cmap="gray",
            vmax=np.percentile(
                self.markers[self.granuleB].raw_image, granule_percentile
            ),
        )

        ax[0].contour(
            self.markers["nucleus"].segmentation.astype(bool),
            colors="#f0be4a",
            linewidths=0.5,
        )
        ax[0].contour(
            self.markers["cell"].segmentation.astype(bool),
            colors="#3f83bf",
            linewidths=0.5,
        )

        ax[1].imshow(
            self.markers[self.granuleA].segmentation,
            cmap=segmentation_cmap,
            alpha=(self.markers[self.granuleA].segmentation > 0).astype(float)
            * granule_alpha,
        )
        ax[1].contour(
            self.markers[self.granuleA].segmentation.astype(bool),
            linewidths=0.5,
        )

        ax[2].imshow(
            self.markers[self.granuleB].segmentation,
            cmap=segmentation_cmap,
            alpha=(self.markers[self.granuleB].segmentation > 0).astype(float)
            * granule_alpha,
        )
        ax[2].contour(
            self.markers[self.granuleB].segmentation.astype(bool),
            linewidths=0.5,
        )

        ax[0].text(
            -0.1,
            0.5,
            f"{self.uid}-{self.condition if not blind else ''}",
            ha="center",
            va="center",
            rotation="vertical",
            transform=ax[0].transAxes,
        )

    def clean_segmentations(self, max_granule_size=650):
        # Fill holes in cell segmentation
        self.markers["cell"].segmentation = ndi.binary_fill_holes(
            self.markers["cell"].segmentation > 0
        )

        # Delete segmentations outside of the cell segmentation
        for marker_name in [self.granuleA, self.granuleB, "nucleus"]:
            self.markers[marker_name].segmentation[
                self.markers["cell"].segmentation == 0
            ] = 0

        self.markers["cell"].segmentation = measure.label(
            self.markers["cell"].segmentation
        )

        # Fill holes in nucleus segmentation
        self.markers["nucleus"].segmentation = ndi.binary_fill_holes(
            self.markers["nucleus"].segmentation > 0
        )
        self.markers["nucleus"].segmentation = measure.label(
            self.markers["nucleus"].segmentation
        )

        for granule_marker in [self.granuleA, self.granuleB]:
            regions = measure.regionprops(self.markers[granule_marker].segmentation)
            for region in regions:
                if region.area > max_granule_size:
                    self.markers[granule_marker].segmentation[
                        self.markers[granule_marker].segmentation == region.label
                    ] = 0

        self.log(
            "Cleaned segmentations by filling holes and removing segmentations outside of the cell"
        )

    def compute_features(self):
        data_collector = {
            "uid": self.uid,
            "condition": self.condition,
            "stage": self.stage,
        }

        # Pass marker features to data collector
        data_collector.update(
            {
                "granule_features": {
                    self.granuleA: self.markers[self.granuleA].get_features(),
                    self.granuleB: self.markers[self.granuleB].get_features(),
                }
            }
        )

        # Get basic cell and nucleus features
        data_collector.update(
            {
                "cell": {
                    "Area": np.sum(self.markers["cell"].segmentation > 0),
                    "SphericalVolume": features.spherical_volume(
                        self.markers["cell"].segmentation > 0
                    ),
                },
                "nucleus": {
                    "Area": np.sum(self.markers["nucleus"].segmentation > 0),
                    "SphericalVolume": features.spherical_volume(
                        self.markers["nucleus"].segmentation > 0
                    ),
                },
            }
        )

        # Get nuclear distance features
        for marker_name in [self.granuleA, self.granuleB]:
            data_collector["granule_features"][marker_name][
                "NuclearDistanceFeatures"
            ] = list(
                features.nuclear_distance_features(
                    self.markers[marker_name].segmentation,
                    self.markers["nucleus"].segmentation,
                    self.markers["cell"].segmentation,
                )
            )

        # Get dnd1 and gra co-localization features
        coloc_features = {
            "MandersPercentile": features.manders_across_percentiles(
                self.markers[self.granuleA].raw_image,
                self.markers[self.granuleB].raw_image,
                mask=self.markers["cell"].segmentation > 0,
            ),
            "MandersSegmentationsM1": features.manders(
                self.markers[self.granuleA].raw_image,
                self.markers[self.granuleB].segmentation > 0,
                mask=self.markers["cell"].segmentation > 0,
            ),
            "MandersSegmentationsM2": features.manders(
                self.markers[self.granuleB].raw_image,
                self.markers[self.granuleA].segmentation > 0,
                mask=self.markers["cell"].segmentation > 0,
            ),
            "Pearsons": features.pearsons(
                self.markers[self.granuleA].raw_image,
                self.markers[self.granuleB].raw_image,
                mask=self.markers["cell"].segmentation > 0,
            ),
            "IoU": features.iou(
                self.markers[self.granuleA].segmentation > 0,
                self.markers[self.granuleB].segmentation > 0,
            ),
        }
        data_collector["granule_features"]["colocalization"] = coloc_features

        granule_to_cytoplasm = {
            f"{self.granuleA}_to_cytoplasm": features.granule_to_cytoplasm_intensity_ratio(
                self.markers[self.granuleA].raw_image,
                self.markers[self.granuleA].segmentation,
                self.markers["nucleus"].segmentation,
                self.markers["cell"].segmentation,
            ),
            f"{self.granuleB}_to_cytoplasm": features.granule_to_cytoplasm_intensity_ratio(
                self.markers[self.granuleB].raw_image,
                self.markers[self.granuleB].segmentation,
                self.markers["nucleus"].segmentation,
                self.markers["cell"].segmentation,
            ),
        }

        data_collector["granule_features"]["granule_to_cytoplasm"] = (
            granule_to_cytoplasm
        )

        self.features = data_collector

    def cell_segmentation_exists(self, min_cell_area=100):
        """
        Returns False if the cell segmentation is empty or below a certain area threshold, True otherwise
        """
        area_cell_segmentation = np.sum(self.markers["cell"].segmentation > 0)
        if area_cell_segmentation < min_cell_area:  # Threshold for minimum cell area
            return False
        else:
            return True

    def get_granule_features(self):
        # Check if features is an empty dict
        if not self.features:
            raise ValueError(
                "Features have not been computed yet. Please run compute_features() first."
            )

        big_data = []
        for marker_name in [self.granuleA, self.granuleB]:
            marker_features = self.features["granule_features"][marker_name]
            unique_granule_indices = np.unique(self.markers[marker_name].segmentation)
            marker_feature_dict = {
                "uid": self.uid,
                "condition": self.condition,
                "stage": self.stage,
                "marker": marker_name,
            }

            for granule_idx in unique_granule_indices:
                if granule_idx == 0:
                    continue
                granule_feature_dict = marker_feature_dict.copy()
                for feature_category_name, feature_list in marker_features.items():
                    if isinstance(feature_list, str):
                        continue
                    for granule_data_dict in feature_list:
                        if granule_data_dict["GranuleIndex"] == granule_idx:
                            granule_feature_dict["GranuleIndex"] = granule_idx
                            granule_feature_dict["FeatureCategory"] = (
                                feature_category_name
                            )
                            for (
                                feature_name,
                                feature_value,
                            ) in granule_data_dict.items():
                                if feature_name != "GranuleIndex":
                                    granule_feature_dict[feature_name] = feature_value
                big_data.append(granule_feature_dict)
        return big_data

    def get_cell_features(self):

        data = {
            "uid": self.uid,
            "condition": self.condition,
            "stage": self.stage,
            "markers": f"{self.granuleA}, {self.granuleB}",
        }

        coloc = self.features["granule_features"]["colocalization"].copy()
        # Remove MandersPercentile from coloc dict
        coloc.pop("MandersPercentile", None)
        data.update(coloc)

        data.update(self.features["granule_features"]["granule_to_cytoplasm"])

        for key in ["cell", "nucleus"]:
            data.update(
                {
                    f"{key.capitalize()}{feature_name}": feature_value
                    for feature_name, feature_value in self.features[key].items()
                }
            )
        return data
