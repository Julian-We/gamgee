import re
import threading
import uuid
from pathlib import Path
import numpy as np
from skimage import measure, restoration
from skimage.io import imread
from skimage.util import img_as_float32, img_as_ubyte
from skimage.exposure import rescale_intensity
from .modelhandler import ModelHandler
from gamgee.utils.utils import normalize
import gamgee.features as features_module

class Marker:
    def __init__(self, name: str, parent_name: str, parent_id: str, parent_root: Path, model_handler=None,
                 **kwargs):
        """Initialize a marker instance.
        Args:
            name (str): Name of the marker.
        """

        # Define IO attributes
        self.parent_name = parent_name
        self.parent_id = parent_id
        self.parent_root = parent_root if parent_root is not isinstance(parent_root, Path) else Path(parent_root)
        self.name = name
        self.uid = uuid.uuid4().hex
        self.root = self.parent_root / self.name

        self.logs = {
            "Name": self.name,
            "Parent Name": self.parent_name,
            "Parent ID": self.parent_id,
            "Preprocessing": []
        }

        # Define metadata attributes
        if not kwargs.get('compartment', False):
            self.compartment = self.set_compartment()
        else:
            self.compartment = kwargs.get('compartment')
        self.sam_model = None
        if model_handler is None:
            model_handler = ModelHandler()
        self.set_sam_model(model_handler)


        # Define model attributes
        self.denoising_model_name = kwargs.get('denoising_model_name', None)
        if self.denoising_model_name is None:
            if self.compartment == 'granules':
                self.denoising_model_name = '20250812_JW_granule_25'
            elif self.compartment == 'cell':
                self.denoising_model_name = '250721_PGC_nls_JS_noise25'
            elif self.compartment == 'nucleus':
                self.denoising_model_name = '250721_PGC_nls_JS_noise25'
            else:
                self.denoising_model_name = None
        self.logs["Denoising Model"] = self.denoising_model_name

        # Define data attributes
        self.raw_image = None
        self._denoised_image = None  # Private attribute to store the actual value
        self.segmentation = None

        self.identify_image()
        self.preprocess()

        # Generate freatue dictionary. Keys should be feature names, values the corresponding values or a list of values
        self.features = None



    def set_compartment(self):
        granule_compartment_keywords = ['granule', 'granules',
                                        'dnd', 'gra', 'ddx', 'nos', 'piwi', 'tdrd', 'tdrda',
                                        'dead end', 'nanos', 'vasa', 'granulito', 'hyper', 'hypergerm']
        cell_compartment_keywords = ['cell', 'membrane', 'membranes', 'cyto', 'cytoplasm', 'cytoplasmatic']
        nucleus_compartment_keywords = ['nucleus', 'nuclei', 'nls', 'nuclear', 'nucleic', 'dna', 'chromatin', 'dapi']
        if re.sub(r'\d+', '', self.name).lower() in granule_compartment_keywords:
            self.logs["Compartment"] = 'granules'
            return 'granules'
        elif re.sub(r'\d+', '', self.name).lower() in cell_compartment_keywords:
            self.logs["Compartment"] = 'cell'
            return 'cell'
        elif re.sub(r'\d+', '', self.name).lower() in nucleus_compartment_keywords:
            self.logs["Compartment"] = 'nucleus'
            return 'nucleus'
        else:
            return 'unknown'


    def set_sam_model(self, model_handler: ModelHandler):
        if self.compartment == 'granules':
            self.sam_model = model_handler.granules
        elif self.compartment == 'cell':
            if 'membrane' in self.name.lower():
                self.sam_model = model_handler.cell_membrane
            elif 'cyto' in self.name.lower() or 'cell' in self.name.lower():
                self.sam_model = model_handler.cell_nls
            else:
                self.sam_model = model_handler.large
        elif self.compartment == 'nucleus':
            self.sam_model = model_handler.nucleus_nls
        else:
            self.sam_model = model_handler.large
        self.logs["SAM Model"] = self.sam_model.friendly_name


    def identify_image(self):
        for file in self.root.iterdir():
            if file.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                self.raw_image = normalize(imread(file))
                self.logs["Image File"] = str(file)
                self.logs["Image Shape"] = self.raw_image.shape
                return 'Image loaded successfully.'
        raise ValueError(f"No image file found in directory {self.root}.")



    def preprocess(self, **kwargs):
        """Preprocess the raw image.
        Args:
            kwargs: Additional parameters for preprocessing.
        """
        if self.raw_image is None:
            raise ValueError("Raw image is not loaded.")
        else:
            img = self.raw_image.copy()

        if img.ndim != 2:
            self.logs["Preprocessing"].append(f"Image is not 2D (shape:{img.shape})– performing MIP.")
            self.raw_image = np.max(img, axis=0)

    def get_adaptive_denoising_params(self):
        image = self.raw_image
        """Estimate mild denoising parameters based on image characteristics."""
        # Calculate image noise level using robust MAD estimator
        gray_diff = np.diff(image.astype(np.float32), axis=1)
        noise_std = np.median(np.abs(gray_diff - np.median(gray_diff))) / 0.6745

        # Calculate image dynamic range
        percentile_99 = np.percentile(image, 99)
        percentile_1 = np.percentile(image, 1)
        dynamic_range = percentile_99 - percentile_1

        # Adaptive weight calculation (much lower = less blurry)
        # Reduced base weight for minimal denoising
        base_weight = 0.005  # Reduced from 0.05 for less aggressive denoising
        weight = base_weight * (noise_std / dynamic_range) * 500  # Reduced multiplier from 1000
        weight = np.clip(weight, 0.005, 0.08)  # Lower range to preserve detail

        # Fewer iterations for less aggressive denoising
        iterations = max(25, min(100, int(50 * (noise_std / dynamic_range) * 5)))  # Reduced iterations

        return weight, iterations

    def tv_denoising(self):
        if self.denoising_model_name is None:
            self.logs["Preprocessing"].append("No denoising model specified. Using adaptive TV chambolle denoising.")

            # Get adaptive parameters for mild denoising
            weight, iterations = self.get_adaptive_denoising_params()

            self.denoised_image = restoration.denoise_tv_chambolle(
                self.raw_image,
                weight=weight,
                max_num_iter=5
            )

            self.logs["Denoising Parameters"] = {
                "weight": weight, "iterations": iterations, "method": "adaptive_tv_chambolle"
            }



    @property
    def denoised_image(self):
        return self._denoised_image

    @denoised_image.setter
    def denoised_image(self, value):
        self._denoised_image = value
        if value is not None:
            # Automatically run segmentation when denoised_image is set
            try:
                self.segment()
            except Exception as e:
                print(f"Error during automatic segmentation: {e}")
                self.logs["Auto-segmentation Error"] = str(e)

    def segment(self, tv_denoise_if_needed=True):
        if self.denoised_image is None and tv_denoise_if_needed:
            self.tv_denoising()
        if self.denoised_image is None:
            raise ValueError("Denoised image is not available for segmentation.")
        if self.sam_model is None:
            raise ValueError("SAM model is not set for segmentation.")

        # Get the model name to find the appropriate lock
        model_name = None
        for name, model in self.sam_model.__dict__.items():
            if hasattr(model, 'model_type'):
                model_name = name
                break

        # Use the model lock to ensure thread-safe segmentation
        if hasattr(self.sam_model, '_model_locks'):
            lock = self.sam_model._model_locks.get(model_name, threading.Lock())
        else:
            # Fallback if model handler doesn't have locks
            lock = threading.Lock()

        with lock:
            self.segmentation = self.sam_model.segment(self.denoised_image)

        self.segmentation = self.sam_model.segment(self.denoised_image)
        if self.compartment.lower() == "cell":
            self.logs["Segmentation Compartment"] = self.compartment
            # If there is more than one cell, keep only the one that is closest to the center
            if np.max(self.segmentation) > 1:
                regions = measure.regionprops(self.segmentation)
                image_center = np.array(self.denoised_image.shape) / 2
                min_distance = float('inf')
                selected_label = 0
                for region in regions:
                    centroid = np.array(region.centroid)
                    distance = np.linalg.norm(centroid - image_center)
                    if distance < min_distance:
                        min_distance = distance
                        selected_label = region.label
                self.segmentation = (self.segmentation == selected_label).astype(np.int32)
                self.logs["Segmentation Note"] = f"Multiple cells detected. Kept only the cell closest to the center (label {selected_label})."
        self.logs["Segmentation Info"] = self.sam_model.get_model_info()
        self.logs["Segmentation Shape"] = self.segmentation.shape

        self.features = self.get_features()

    def get_features(self, **kwargs):
        if self.segmentation is None:
            raise ValueError("Segmentation is not available for feature extraction.")
        return {
            "Marker Name": self.name,
            "Compartment": self.compartment,
            "BasicMorphology": features_module.basic_granule_features(self.segmentation),
            "IntensityFeatures": features_module.intensity_granule_features(self.raw_image, self.segmentation)
        }