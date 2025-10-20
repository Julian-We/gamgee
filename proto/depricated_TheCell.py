# class TheCell:
#     def __init__(self, name: str, root_path: str, **kwargs):
#         """Initialize a cell instance.
#         Args:
#             name (str): Name of the cell.
#         """
#         self.name = name
#         self.cell_id = uuid.uuid4().hex
#         self.logs = {
#             "Name": self.name,
#             "Cell ID": self.cell_id,
#         }
#
#
#         self.root = Path(root_path)
#         # Paralell dictionaries that have a tag (key) as a label and correspond to a path/raw_image/model/segmentation
#         self.paths = {}
#         self.raw_images = {}
#         self.images_for_segmentation = {}
#         self.segmentations = {}
#         self.segmentation_models = {}
#         self.compartments = {}
#
#         # Checkpoint for loader function
#         self.loader_checkpoint = False
#
#     @staticmethod
#     def imread(path) -> np.ndarray:
#         """Read an image from a given path.
#         Args:
#             path (Path): Path to the image file.
#
#         Returns:
#             np.ndarray: Image data as a numpy array.
#         """
#         if not isinstance(path, Path):
#             path = Path(path)
#         if not path.is_file():
#             raise ValueError("Path is not a file.")
#
#         if path.suffix.lower() == '.tif' or path.suffix.lower() == '.tiff':
#             return tiff.imread(path)
#         elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
#             with Image.open(path) as img:
#                 return np.array(img)
#         else:
#             raise ValueError(f"Unsupported file format: {path.suffix}")
#
#
#
#     def loader(self, loader_dict:dict = None, model_handler:ModelHandler = None, **kwargs):
#         """
#         Load data for the cell instance.
#
#         Args:
#             loader_dict (dict): Dictionary containing data subfolders and the models that are used. Format is suposed to be key: tag, value: [path_to_image_file, model_descriptor, image_restauration_dict].
#             Image restauration dict is optional and can contain the following keys:
#                 - rolling_ball: bool, if True, rolling ball algorithm is applied
#                 - rolling_ball_radius: int, radius for the rolling ball algorithm
#                 - non_local_means: bool, if True, non-local means denoising is applied
#                 - non_local_means_params: dict, parameters for non-local means denoising
#         :return:
#         """
#         if loader_dict is None:
#             # if verbose in kwargs print("No loader_dict provided. Using root path to load data.")
#             loader_dict = {}
#             if kwargs.get('verbose', True):
#                 print("No loader_dict provided. Using root path to load data.")
#
#             if not self.root.exists():
#                 raise ValueError(f"Root path {self.root} does not exist.")
#
#             self.loader_checkpoint = True
#             # If no loader_dict is provided, create a default one from the root directory
#
#             model_translator = {
#                 'granule': 'granules',
#                 'cell': 'cell_nls',
#                 'membrane': 'cell_membrane',
#                 'nucleus': 'nucleus_nls',
#             }
#
#
#             directories = [item for item in self.root.iterdir() if item.is_dir()]
#
#             for directory in directories:
#                 dir_name = directory.name
#                 # All image files in the directory (tif, tiff, png, jpg, jpeg. bmp, gif)
#                 image_files = list(directory.glob('*.tif')) + \
#                               list(directory.glob('*.tiff')) + \
#                               list(directory.glob('*.png')) + \
#                               list(directory.glob('*.jpg')) + \
#                               list(directory.glob('*.jpeg')) + \
#                               list(directory.glob('*.bmp')) + \
#                               list(directory.glob('*.gif'))
#                 if not image_files:
#                     raise ValueError(f"No image files found in directory {directory}.")
#                 if len(image_files) > 1:
#                     raise ValueError(f"Multiple image files found in directory {directory}. Please provide a single image file for segmentation. Movies are not supported yet.")
#
#                 name_in_translator = False
#                 model_name = None
#                 # print(dir_name.lower())
#                 for key in model_translator.keys():
#                     if key in dir_name.lower():
#                         if kwargs.get('verbose', True):
#                             print(f"Found model {key} in directory {dir_name}.")
#                         name_in_translator = True
#                         model_name = model_translator[key]
#                 if not name_in_translator:
#                     model_name = 'base'
#
#                 raw_image_path = image_files[0]
#
#                 if model_name is not None:
#                     self.paths[dir_name] = raw_image_path
#                     self.logs[dir_name] = {}
#                     self.raw_images[dir_name] = self.imread(raw_image_path)
#                     self.images_for_segmentation[dir_name] = self.raw_images[dir_name]
#                     self.segmentation_models[dir_name] = model_name
#                     self.compartments[dir_name] = model_name.cell_compartment if hasattr(model_name, 'cell_compartment') else None
#                 else:
#                     raise ValueError(f"Directory name {dir_name} does not match any known model names. Please provide a directory name that contains one of the following keywords: {', '.join(model_translator.keys())}.")
#
#                 loader_dict[dir_name] = [
#                     str(raw_image_path),
#                     model_name,
#                     kwargs.get('image_restauration', {})
#                 ]
#
#         else:
#             self.loader_checkpoint = True
#             if kwargs.get('verbose', True):
#                 print("Loader dictionary provided. Using it to load data.")
#
#         if not isinstance(loader_dict, dict):
#             raise ValueError("Loader dictionary must be a dictionary.")
#
#         if model_handler is None:
#             model_handler = ModelHandler()
#             if kwargs.get('verbose', True):
#                 print("No model handler provided. Using default model handler.")
#
#         self.populator(loader_dict, model_handler, **kwargs)
#
#     def populator(self, loader_dict:dict, model_handler:ModelHandler,
#                   care_denoise:bool = True,
#                   **kwargs):
#         if model_handler is None:
#             raise ValueError("Module handler cannot be None.")
#         for key, value in loader_dict.items():
#             self.paths[key] = Path(value[0])
#             self.raw_images[key] = self.imread(value[0])
#             if not self.raw_images[key].ndim == 2:
#                 raise ValueError(f"Image <{Path(value[0]).name}> is not a 2D image. Please provide a 2D image for segmentation.")
#
#             if len(value) > 1 and value[1] in model_handler.segmentation_models:
#                 self.segmentation_models[key] = model_handler.segmentation_models[value[1]]
#             else:
#                 raise ValueError(f"<{Path(value[0]).name}>Model {value[1]} not found in model handler.")
#
#
#
#         self.images_for_segmentation = care_denoising(self.raw_images) if care_denoise else self.raw_images

