# from .segmenter import SegmentationModel
# from .denoising_interface import care_denoising, denoise_with_care
# from .utils.denoising import encode_memmap_info, get_memmap_info
# from .utils.utils import imread, normalize
# import numpy as np
# from skimage import measure, restoration
# import pickle
# import threading
# from pathlib import Path
# import re
# import uuid
# import tempfile
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading
#
# # Get the directory where this file is located
# _CURRENT_DIR = Path(__file__).parent
# _MODELS_DIR = _CURRENT_DIR / "models" / "msam"
#
# class ModelHandler:
#     def __init__(self, **kwargs):
#         self.segmentation_models = {
#         "base": SegmentationModel(None,
#                                   model_type='vit_b_lm',
#                                   friendly_name='µSAM Base model',
#                                   upsampling_factor=1),
#         "large": SegmentationModel(None,
#                                    model_type='vit_l_lm',
#                                    friendly_name='µSAM Large model',
#                                    upsampling_factor=1),
#         "cell_membrane": SegmentationModel(None,
#                                            model_type='vit_l_lm',
#                                            friendly_name='µSAM Cell (Membrane) model',
#                                            upsampling_factor=1,
#                                            cell_compartment='cell'),
#         "nucleus_nls": SegmentationModel(_MODELS_DIR / 'nls_nucleus' if (_MODELS_DIR / 'nls_nucleus').exists() else None,
#                                          model_type='vit_b_lm',
#                                          friendly_name='µSAM Nucleus (NLS) model',
#                                          cell_compartment='nucleus',
#                                          upsampling_factor=1),
#         "cell_nls": SegmentationModel(_MODELS_DIR / 'nls_cell' if (_MODELS_DIR / 'nls_cell').exists() else None,
#                                       model_type='vit_b_lm',
#                                       friendly_name='µSAM Cell (NLS) model',
#                                       cell_compartment='cell',
#                                       upsampling_factor=1),
#         "granules": SegmentationModel(path=_MODELS_DIR / 'granules' if (_MODELS_DIR / 'granules').exists() else None,
#                                       model_type='vit_l_lm',
#                                       friendly_name='µSAM Granules model',
#                                       cell_compartment='granules',
#                                       upsampling_factor=3),
#         }
#
#         self._model_locks = {model_name: threading.Lock() for model_name in self.segmentation_models.keys()}
#     def __getattr__(self, name):
#         """Allow direct access to models as attributes.
#
#         Example: mh.granules instead of mh.segmentation_models.get('granules')
#         """
#         if name in self.segmentation_models:
#             return self.segmentation_models[name]
#         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
#
#     def add_model(self, model_name: str,
#                   model_path: str,
#                   model_type: str = 'vit_b_lm',
#                   friendly_name: str = None,
#                   cell_compartment: str = None):
#         """Add a new segmentation model to the handler.
#
#         Args:
#             :param model_name: model_name (str): Name of the model.
#             :param model_path:  model_path (str): Path to the model file.
#             :param model_type: model_type (str): Type of the model, default is 'vit_b_lm'.
#             :param friendly_name: friendly_name (str): Friendly name for the model, default is None.
#             :param cell_compartment: Compartment of the cell that the model is trained on, e.g. 'cell', 'nucleus', 'granule'. Default is None.
#         """
#         if friendly_name is None:
#             friendly_name = model_name
#         if model_name in self.segmentation_models.keys():
#             raise ValueError(f"Model {model_name} already exists in the handler. Please use a different name.")
#         self.segmentation_models[model_name] = SegmentationModel(Path(model_path),
#                                                                  model_type=model_type,
#                                                                  friendly_name=friendly_name,
#                                                                  cell_compartment=cell_compartment)
#         self._model_locks[model_name] = threading.Lock()
#     def get_available_models(self):
#         """Get a list of available segmentation models.
#
#         Returns:
#             list: List of model names.
#         """
#         return list(self.segmentation_models.keys())
#
#     def get_model_lock(self, model_name: str):
#         """Get the lock for a specific model."""
#         return self._model_locks.get(model_name, threading.Lock())
#
#     def get_model_by_compartment(self):
#         """Get a dictionary of models grouped by cell compartment.
#
#         Returns:
#             dict: Dictionary with compartment names as keys and lists of model names as values.
#         """
#         compartment_dict = {}
#         for model_name, model in self.segmentation_models.items():
#             compartment = model.cell_compartment if hasattr(model, 'cell_compartment') else 'unknown'
#             if compartment not in compartment_dict:
#                 compartment_dict[compartment] = []
#             compartment_dict[compartment].append(model_name)
#         return compartment_dict
#
#
# class TheCell:
#     def __init__(self, root_path: str, model_handler: ModelHandler, name=None, blacklist=None, **kwargs):
#         """Initialize a cell instance.
#         Args:
#             name (str): Name of the cell.
#             blacklist (list): List of folder names to ignore when scanning for markers.
#         """
#         self.root = Path(root_path)
#         self.output_root = self.root / 'output'
#         self.output_root.mkdir(exist_ok=True, parents=True)
#         self.name = name if name is not None else self.root.name
#         self.cell_id = uuid.uuid4().hex
#
#         # Default blacklist for common non-marker folders
#         if blacklist is None:
#             blacklist = ['.git', '__pycache__', '.DS_Store', 'models', 'utils', 'temp', 'cache', 'logs',
#                          'export', 'results', 'denoised', 'segmentations', 'masks', 'output', 'xprt', 'raw']
#         self.blacklist = blacklist
#
#         self.logs = {
#             "Name": self.name,
#             "Cell ID": self.cell_id,
#         }
#
#         self.markers = {}
#         self.care_denoising_models = {}
#         self.model_handler = model_handler
#
#         # Scan for marker folders and create Marker objects
#         self._scan_and_create_markers()
#
#     def _scan_and_create_markers(self):
#         """Scan root directory for folders containing single image files and create Marker objects."""
#         if not self.root.exists():
#             raise ValueError(f"Root path {self.root} does not exist.")
#
#         valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif']
#
#         for folder in self.root.iterdir():
#             if not folder.is_dir():
#                 continue
#
#             # Skip blacklisted folders
#             if folder.name in self.blacklist:
#                 continue
#
#             # Find image files in the folder
#             image_files = [f for f in folder.iterdir()
#                           if f.is_file() and f.suffix.lower() in valid_extensions]
#
#             # Only create marker if exactly one image file is found
#             if len(image_files) == 1:
#                 try:
#                     marker = Marker(
#                         name=folder.name,
#                         parent_name=self.name,
#                         parent_id=self.cell_id,
#                         parent_root=self.root,
#                         model_handler=self.model_handler
#                     )
#                     self.markers[folder.name] = marker
#                     self.logs[f"Marker_{folder.name}"] = "Created successfully"
#                 except Exception as e:
#                     self.logs[f"Marker_{folder.name}_Error"] = str(e)
#             elif len(image_files) == 0:
#                 self.logs[f"Folder_{folder.name}"] = "No image files found"
#             else:
#                 self.logs[f"Folder_{folder.name}"] = f"Multiple image files found ({len(image_files)})"
#
#     def denoise(self, use_tv_denoising=False, max_workers=None):
#         """Denoise all markers in parallel.
#
#         Args:
#             use_tv_denoising (bool): If True, use TV denoising instead of CARE models
#             max_workers (int): Maximum number of parallel workers. If None, uses number of CPU cores
#         """
#         if not self.markers:
#             raise ValueError("No markers found to process.")
#
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             future_to_marker = {}
#             for marker in self.markers.values():
#                 if use_tv_denoising or marker.denoising_model_name is None:
#                     future = executor.submit(self._denoise_marker_tv, marker)
#                 else:
#                     future = executor.submit(self._denoise_marker_care, marker)
#                 future_to_marker[future] = marker.name
#
#             for future in as_completed(future_to_marker):
#                 marker_name = future_to_marker[future]
#                 try:
#                     future.result()
#                     self.logs[f"Denoising_{marker_name}"] = "Denoising completed successfully"
#                 except Exception as e:
#                     self.logs[f"Denoising_{marker_name}_Error"] = str(e)
#
#
#
#
#     def _denoise_marker_care(self, marker):
#         """Denoise a single marker using CARE model."""
#         if marker.denoising_model_name is None:
#             # If no CARE model specified, fall back to TV denoising
#             self._denoise_marker_tv(marker)
#         else:
#             # Use CARE denoising for single marker
#             denoise_with_care([marker], model_name=marker.denoising_model_name)
#
#     def _denoise_marker_tv(self, marker):
#         """Denoise a single marker using TV denoising."""
#         marker.tv_denoising()
#
#     def plot_markers_and_segmentations(self):
#         import matplotlib.pyplot as plt
#         num_markers = len(self.markers)
#         if num_markers == 0:
#             print("No markers to plot.")
#             return
#
#         num_markers = len(self.markers)
#         fig, axes = plt.subplots(num_markers, 3, figsize=(15, 5 * num_markers))
#         for i, (marker_name, marker) in enumerate(self.markers.items()):
#             axes[i, 0].imshow(marker.raw_image, cmap='gray')
#             axes[i, 0].set_title(f"{marker.name} - Raw Image")
#             axes[i, 0].axis('off')
#
#             if marker.denoised_image is not None:
#                 axes[i, 1].imshow(marker.denoised_image, cmap='gray')
#                 axes[i, 1].set_title(f"{marker.name} - Denoised Image")
#             else:
#                 axes[i, 1].text(0.5, 0.5, 'No Denoised Image', horizontalalignment='center', verticalalignment='center')
#             axes[i, 1].axis('off')
#
#             if marker.segmentation is not None:
#                 axes[i, 2].imshow(marker.segmentation, cmap='nipy_spectral')
#                 axes[i, 2].set_title(f"{marker.name} - Segmentation")
#             else:
#                 axes[i, 2].text(0.5, 0.5, 'No Segmentation', horizontalalignment='center', verticalalignment='center')
#             axes[i, 2].axis('off')
#
#         # save as png
#         plt.tight_layout()
#         plt.savefig(self.output_root / f"{self.name}_markers_segmentations.png", dpi=300)
#
#
#
#     # def denoise_and_segment(self, use_tv_denoising=False):
#     #     """Denoise and segment all markers, grouping by denoising model for efficiency."""
#     #     if not self.markers:
#     #         raise ValueError("No markers found to process.")
#     #
#     #     # Group markers by denoising model
#     #     model_groups = {}
#     #     no_model_markers = []
#     #
#     #     for marker_name, marker in self.markers.items():
#     #         if marker.denoising_model_name is None or use_tv_denoising:
#     #             no_model_markers.append(marker)
#     #         else:
#     #             model_name = marker.denoising_model_name
#     #             if model_name not in model_groups:
#     #                 model_groups[model_name] = []
#     #             model_groups[model_name].append(marker)
#     #
#     #     # Process markers without CARE denoising models (use TV denoising or raw image)
#     #     for marker in no_model_markers:
#     #         if use_tv_denoising:
#     #             marker.tv_denoising()
#     #             self.logs[f"{marker.name}_denoising"] = "TV Chambolle denoising applied"
#     #         else:
#     #             marker.denoised_image = marker.raw_image.copy()
#     #             self.logs[f"{marker.name}_denoising"] = "No denoising applied (using raw image)"
#     #
#     #     # Process markers with CARE denoising models
#     #     print(f"Starting CARE denoising for {model_groups} model groups.")
#     #     for model_name, markers_list in model_groups.items():
#     #         try:
#     #             denoise_with_care(markers_list, model_name=model_name)
#     #             self.logs[f"CARE_denoising_{model_name}"] = f"Applied to {len(markers_list)} markers"
#     #         except Exception as e:
#     #             # Fallback to TV denoising if CARE fails
#     #             self.logs[f"CARE_denoising_{model_name}_Error"] = str(e)
#     #             for marker in markers_list:
#     #                 marker.tv_denoising()
#     #                 self.logs[f"{marker.name}_denoising_fallback"] = "TV denoising applied (CARE failed)"
#     #
#     #     self.logs["Denoising_Complete"] = f"Processed {len(self.markers)} markers"
#
#
#
#
#
#
# class Marker:
#     def __init__(self, name: str, parent_name: str, parent_id: str, parent_root: Path, model_handler=None,
#                  **kwargs):
#         """Initialize a marker instance.
#         Args:
#             name (str): Name of the marker.
#         """
#
#         # Define IO attributes
#         self.parent_name = parent_name
#         self.parent_id = parent_id
#         self.parent_root = parent_root if parent_root is not isinstance(parent_root, Path) else Path(parent_root)
#         self.name = name
#         self.uid = uuid.uuid4().hex
#         self.root = self.parent_root / self.name
#
#         self.logs = {
#             "Name": self.name,
#             "Parent Name": self.parent_name,
#             "Parent ID": self.parent_id,
#             "Preprocessing": []
#         }
#
#         # Define metadata attributes
#         if not kwargs.get('compartment', False):
#             self.compartment = self.set_compartment()
#         else:
#             self.compartment = kwargs.get('compartment')
#         self.sam_model = None
#         if model_handler is None:
#             model_handler = ModelHandler()
#         self.set_sam_model(model_handler)
#
#
#         # Define model attributes
#         self.denoising_model_name = kwargs.get('denoising_model_name', None)
#         if self.denoising_model_name is None:
#             if self.compartment == 'granules':
#                 self.denoising_model_name = '20250812_JW_granule_25'
#             elif self.compartment == 'cell':
#                 self.denoising_model_name = '250721_PGC_nls_JS_noise25'
#             elif self.compartment == 'nucleus':
#                 self.denoising_model_name = '250721_PGC_nls_JS_noise25'
#             else:
#                 self.denoising_model_name = None
#         self.logs["Denoising Model"] = self.denoising_model_name
#
#         # Define data attributes
#         self.raw_image = None
#         self._denoised_image = None  # Private attribute to store the actual value
#         self.segmentation = None
#
#         self.identify_image()
#
#         self.preprocess()
#
#         # Generate freatue dictionary. Keys should be feature names, values the corresponding values or a list of values
#         self.features = []
#
#
#
#     def set_compartment(self):
#         granule_compartment_keywords = ['granule', 'granules',
#                                         'dnd', 'gra', 'ddx', 'nos', 'piwi', 'tdrd', 'tdrda',
#                                         'dead end', 'nanos', 'vasa', 'granulito', 'hyper', 'hypergerm']
#         cell_compartment_keywords = ['cell', 'membrane', 'membranes', 'cyto', 'cytoplasm', 'cytoplasmatic']
#         nucleus_compartment_keywords = ['nucleus', 'nuclei', 'nls', 'nuclear', 'nucleic', 'dna', 'chromatin', 'dapi']
#         if re.sub(r'\d+', '', self.name).lower() in granule_compartment_keywords:
#             self.logs["Compartment"] = 'granules'
#             return 'granules'
#         elif re.sub(r'\d+', '', self.name).lower() in cell_compartment_keywords:
#             self.logs["Compartment"] = 'cell'
#             return 'cell'
#         elif re.sub(r'\d+', '', self.name).lower() in nucleus_compartment_keywords:
#             self.logs["Compartment"] = 'nucleus'
#             return 'nucleus'
#         else:
#             return 'unknown'
#
#
#     def set_sam_model(self, model_handler: ModelHandler):
#         if self.compartment == 'granules':
#             self.sam_model = model_handler.granules
#         elif self.compartment == 'cell':
#             if 'membrane' in self.name.lower():
#                 self.sam_model = model_handler.cell_membrane
#             elif 'cyto' in self.name.lower() or 'cell' in self.name.lower():
#                 self.sam_model = model_handler.cell_nls
#             else:
#                 self.sam_model = model_handler.large
#         elif self.compartment == 'nucleus':
#             self.sam_model = model_handler.nucleus_nls
#         else:
#             self.sam_model = model_handler.large
#         self.logs["SAM Model"] = self.sam_model.friendly_name
#
#
#     def identify_image(self):
#         for file in self.root.iterdir():
#             if file.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif']:
#                 self.raw_image = normalize(imread(file))
#                 self.logs["Image File"] = str(file)
#                 self.logs["Image Shape"] = self.raw_image.shape
#                 return 'Image loaded successfully.'
#         raise ValueError(f"No image file found in directory {self.root}.")
#
#
#
#     def preprocess(self, **kwargs):
#         """Preprocess the raw image.
#         Args:
#             kwargs: Additional parameters for preprocessing.
#         """
#         if self.raw_image is None:
#             raise ValueError("Raw image is not loaded.")
#         else:
#             img = self.raw_image.copy()
#
#         if img.ndim != 2:
#             self.logs["Preprocessing"].append(f"Image is not 2D (shape:{img.shape})– performing MIP.")
#             self.raw_image = np.max(img, axis=0)
#
#     def get_adaptive_denoising_params(self):
#         image = self.raw_image
#         """Estimate mild denoising parameters based on image characteristics."""
#         # Calculate image noise level using robust MAD estimator
#         gray_diff = np.diff(image.astype(np.float32), axis=1)
#         noise_std = np.median(np.abs(gray_diff - np.median(gray_diff))) / 0.6745
#
#         # Calculate image dynamic range
#         percentile_99 = np.percentile(image, 99)
#         percentile_1 = np.percentile(image, 1)
#         dynamic_range = percentile_99 - percentile_1
#
#         # Adaptive weight calculation (much lower = less blurry)
#         # Reduced base weight for minimal denoising
#         base_weight = 0.005  # Reduced from 0.05 for less aggressive denoising
#         weight = base_weight * (noise_std / dynamic_range) * 500  # Reduced multiplier from 1000
#         weight = np.clip(weight, 0.005, 0.08)  # Lower range to preserve detail
#
#         # Fewer iterations for less aggressive denoising
#         iterations = max(25, min(100, int(50 * (noise_std / dynamic_range) * 5)))  # Reduced iterations
#
#         return weight, iterations
#
#     def tv_denoising(self):
#         if self.denoising_model_name is None:
#             self.logs["Preprocessing"].append("No denoising model specified. Using adaptive TV chambolle denoising.")
#
#             # Get adaptive parameters for mild denoising
#             weight, iterations = self.get_adaptive_denoising_params()
#
#             self.denoised_image = restoration.denoise_tv_chambolle(
#                 self.raw_image,
#                 weight=weight,
#                 max_num_iter=5
#             )
#
#             self.logs["Denoising Parameters"] = {
#                 "weight": weight, "iterations": iterations, "method": "adaptive_tv_chambolle"
#             }
#
#
#
#     @property
#     def denoised_image(self):
#         return self._denoised_image
#
#     @denoised_image.setter
#     def denoised_image(self, value):
#         self._denoised_image = value
#         if value is not None:
#             # Automatically run segmentation when denoised_image is set
#             try:
#                 self.segment()
#             except Exception as e:
#                 print(f"Error during automatic segmentation: {e}")
#                 self.logs["Auto-segmentation Error"] = str(e)
#
#     def segment(self, tv_denoise_if_needed=True):
#         if self.denoised_image is None and tv_denoise_if_needed:
#             self.tv_denoising()
#         if self.denoised_image is None:
#             raise ValueError("Denoised image is not available for segmentation.")
#         if self.sam_model is None:
#             raise ValueError("SAM model is not set for segmentation.")
#
#         # Get the model name to find the appropriate lock
#         model_name = None
#         for name, model in self.sam_model.__dict__.items():
#             if hasattr(model, 'model_type'):
#                 model_name = name
#                 break
#
#         # Use the model lock to ensure thread-safe segmentation
#         if hasattr(self.sam_model, '_model_locks'):
#             lock = self.sam_model._model_locks.get(model_name, threading.Lock())
#         else:
#             # Fallback if model handler doesn't have locks
#             lock = threading.Lock()
#
#         with lock:
#             self.segmentation = self.sam_model.segment(self.denoised_image)
#
#         self.segmentation = self.sam_model.segment(self.denoised_image)
#         if self.compartment.lower() == "cell":
#             self.logs["Segmentation Compartment"] = self.compartment
#             # If there is more than one cell, keep only the one that is closest to the center
#             if np.max(self.segmentation) > 1:
#                 regions = measure.regionprops(self.segmentation)
#                 image_center = np.array(self.denoised_image.shape) / 2
#                 min_distance = float('inf')
#                 selected_label = 0
#                 for region in regions:
#                     centroid = np.array(region.centroid)
#                     distance = np.linalg.norm(centroid - image_center)
#                     if distance < min_distance:
#                         min_distance = distance
#                         selected_label = region.label
#                 self.segmentation = (self.segmentation == selected_label).astype(np.int32)
#                 self.logs["Segmentation Note"] = f"Multiple cells detected. Kept only the cell closest to the center (label {selected_label})."
#         self.logs["Segmentation Info"] = self.sam_model.get_model_info()
#         self.logs["Segmentation Shape"] = self.segmentation.shape
#
#     def get_features(self, **kwargs):
#         if self.segmentation is None:
#             raise ValueError("Segmentation is not available for feature extraction.")
#         features = measure.regionprops_table(self.segmentation, intensity_image=self.denoised_image,
#                                              properties=['label', 'area', 'centroid', 'mean_intensity', 'max_intensity',
#                                                          'min_intensity', 'eccentricity', 'solidity', 'extent'],
#                                              **kwargs)
#         return features