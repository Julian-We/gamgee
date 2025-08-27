from .segmenter import SegmentationModel
from .denoising_interface import care_denoising
import numpy as np
from skimage import measure, restoration
import pickle
import tifffile as tiff
from PIL import Image
from pathlib import Path
import re
from scipy import ndimage

class ModelHandler:
    def __init__(self, **kwargs):
        self.segmentation_models = {
        "base": SegmentationModel(None,
                                  model_type='vit_b_lm',
                                  friendly_name='µSAM Base model',
                                  upsampling_factor=1),
        "large": SegmentationModel(None,
                                   model_type='vit_l_lm',
                                   friendly_name='µSAM Large model',
                                   upsampling_factor=1),
        "cell_membrane": SegmentationModel(None,
                                           model_type='vit_l_lm',
                                           friendly_name='µSAM Cell (Membrane) model',
                                           upsampling_factor=1,
                                           cell_compartment='cell'),
        "nucleus_nls": SegmentationModel('/Users/julian/Documents/General Science/Programming/py/general_analysis/20250325_cellpose-retrain/sam_nucleus_large_refined/models/checkpoints/sam_nucleus_refined_up2_33313133',
                                         model_type='vit_l_lm',
                                         friendly_name='µSAM Nucleus (NLS) model'),
        "cell_nls": SegmentationModel('/Users/julian/Documents/General Science/Programming/py/general_analysis/20250325_cellpose-retrain/sam_cell_large_refined/models/checkpoints/sam_cell_refined_up2_33315528',
                                      model_type='vit_l_lm',
                                      friendly_name='µSAM Cell (NLS) model'),
        "granules": SegmentationModel(path='/Users/julian/Documents/General Science/Programming/py/general_analysis/20250325_cellpose-retrain/sam_granule_large_refined/models/checkpoints/sam_granules_refined_up3_33314058',
                                      model_type='vit_l_lm',
                                      friendly_name='µSAM Granules model'),
        }

    def __getattr__(self, name):
        """Allow direct access to models as attributes.

        Example: mh.granules instead of mh.segmentation_models.get('granules')
        """
        if name in self.segmentation_models:
            return self.segmentation_models[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def add_model(self, model_name: str,
                  model_path: str,
                  model_type: str = 'vit_b_lm',
                  friendly_name: str = None,
                  cell_compartment: str = None):
        """Add a new segmentation model to the handler.

        Args:
            :param model_name: model_name (str): Name of the model.
            :param model_path:  model_path (str): Path to the model file.
            :param model_type: model_type (str): Type of the model, default is 'vit_b_lm'.
            :param friendly_name: friendly_name (str): Friendly name for the model, default is None.
            :param cell_compartment: Compartment of the cell that the model is trained on, e.g. 'cell', 'nucleus', 'granule'. Default is None.
        """
        if friendly_name is None:
            friendly_name = model_name
        if model_name in self.segmentation_models.keys():
            raise ValueError(f"Model {model_name} already exists in the handler. Please use a different name.")
        self.segmentation_models[model_name] = SegmentationModel(model_path,
                                                                 model_type=model_type,
                                                                 friendly_name=friendly_name,
                                                                 cell_compartment=cell_compartment)

    def get_available_models(self):
        """Get a list of available segmentation models.

        Returns:
            list: List of model names.
        """
        return list(self.segmentation_models.keys())

class TheCell:
    def __init__(self, name: str, root_path: str, **kwargs):
        """Initialize a cell instance.
        Args:
            name (str): Name of the cell.
        """
        self.name = name
        self.cell_id = np.random.randint(1, 100000000)
        self.logs = {
            "Name": self.name,
            "Cell ID": self.cell_id,
        }


        self.root = Path(root_path)
        # Paralell dictionaries that have a tag (key) as a label and correspond to a path/raw_image/model/segmentation
        self.paths = {}
        self.raw_images = {}
        self.images_for_segmentation = {}
        self.segmentations = {}
        self.segmentation_models = {}
        self.compartments = {}

        # Checkpoint for loader function
        self.loader_checkpoint = False

    @staticmethod
    def imread(path) -> np.ndarray:
        """Read an image from a given path.
        Args:
            path (Path): Path to the image file.

        Returns:
            np.ndarray: Image data as a numpy array.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_file():
            raise ValueError("Path is not a file.")

        if path.suffix.lower() == '.tif' or path.suffix.lower() == '.tiff':
            return tiff.imread(path)
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            with Image.open(path) as img:
                return np.array(img)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")



    def loader(self, loader_dict:dict = None, model_handler:ModelHandler = None, **kwargs):
        """
        Load data for the cell instance.

        Args:
            loader_dict (dict): Dictionary containing data subfolders and the models that are used. Format is suposed to be key: tag, value: [path_to_image_file, model_descriptor, image_restauration_dict].
            Image restauration dict is optional and can contain the following keys:
                - rolling_ball: bool, if True, rolling ball algorithm is applied
                - rolling_ball_radius: int, radius for the rolling ball algorithm
                - non_local_means: bool, if True, non-local means denoising is applied
                - non_local_means_params: dict, parameters for non-local means denoising
        :return:
        """
        if loader_dict is None:
            # if verbose in kwargs print("No loader_dict provided. Using root path to load data.")
            loader_dict = {}
            if kwargs.get('verbose', True):
                print("No loader_dict provided. Using root path to load data.")

            if not self.root.exists():
                raise ValueError(f"Root path {self.root} does not exist.")

            self.loader_checkpoint = True
            # If no loader_dict is provided, create a default one from the root directory

            model_translator = {
                'granule': 'granules',
                'cell': 'cell_nls',
                'membrane': 'cell_membrane',
                'nucleus': 'nucleus_nls',
            }


            directories = [item for item in self.root.iterdir() if item.is_dir()]

            for directory in directories:
                dir_name = directory.name
                # All image files in the directory (tif, tiff, png, jpg, jpeg. bmp, gif)
                image_files = list(directory.glob('*.tif')) + \
                              list(directory.glob('*.tiff')) + \
                              list(directory.glob('*.png')) + \
                              list(directory.glob('*.jpg')) + \
                              list(directory.glob('*.jpeg')) + \
                              list(directory.glob('*.bmp')) + \
                              list(directory.glob('*.gif'))
                if not image_files:
                    raise ValueError(f"No image files found in directory {directory}.")
                if len(image_files) > 1:
                    raise ValueError(f"Multiple image files found in directory {directory}. Please provide a single image file for segmentation. Movies are not supported yet.")

                name_in_translator = False
                model_name = None
                # print(dir_name.lower())
                for key in model_translator.keys():
                    if key in dir_name.lower():
                        if kwargs.get('verbose', True):
                            print(f"Found model {key} in directory {dir_name}.")
                        name_in_translator = True
                        model_name = model_translator[key]
                if not name_in_translator:
                    model_name = 'base'

                raw_image_path = image_files[0]

                if model_name is not None:
                    self.paths[dir_name] = raw_image_path
                    self.logs[dir_name] = {}
                    self.raw_images[dir_name] = self.imread(raw_image_path)
                    self.images_for_segmentation[dir_name] = self.raw_images[dir_name]
                    self.segmentation_models[dir_name] = model_name
                    self.compartments[dir_name] = model_name.cell_compartment if hasattr(model_name, 'cell_compartment') else None
                else:
                    raise ValueError(f"Directory name {dir_name} does not match any known model names. Please provide a directory name that contains one of the following keywords: {', '.join(model_translator.keys())}.")

                loader_dict[dir_name] = [
                    str(raw_image_path),
                    model_name,
                    kwargs.get('image_restauration', {})
                ]

        else:
            self.loader_checkpoint = True
            if kwargs.get('verbose', True):
                print("Loader dictionary provided. Using it to load data.")

        if not isinstance(loader_dict, dict):
            raise ValueError("Loader dictionary must be a dictionary.")

        if model_handler is None:
            model_handler = ModelHandler()
            if kwargs.get('verbose', True):
                print("No model handler provided. Using default model handler.")

        self.populator(loader_dict, model_handler, **kwargs)

    def populator(self, loader_dict:dict, model_handler:ModelHandler,
                  care_denoise:bool = True,
                  **kwargs):
        if model_handler is None:
            raise ValueError("Module handler cannot be None.")
        for key, value in loader_dict.items():
            self.paths[key] = Path(value[0])
            self.raw_images[key] = self.imread(value[0])
            if not self.raw_images[key].ndim == 2:
                raise ValueError(f"Image <{Path(value[0]).name}> is not a 2D image. Please provide a 2D image for segmentation.")

            if len(value) > 1 and value[1] in model_handler.segmentation_models:
                self.segmentation_models[key] = model_handler.segmentation_models[value[1]]
            else:
                raise ValueError(f"<{Path(value[0]).name}>Model {value[1]} not found in model handler.")



        self.images_for_segmentation = care_denoising(self.raw_images) if care_denoise else self.raw_images

    @staticmethod
    def image_restauration(
                    image: type(np.ndarray),
                    rolling_ball: bool = False,
                    rolling_ball_radius: int = 60,
                    non_local_means: bool = False,
                    non_local_means_params: dict = None,
                    ):
        """
        Perform segmentation on the image associated with the given tag.
        :param image:
        :param rolling_ball: bool, if True, rolling ball algorithm is applied
        :param rolling_ball_radius: int, radius for the rolling ball algorithm
        :param non_local_means: bool, if True, non-local means denoising is applied
        :param non_local_means_params: dict, parameters for non-local means denoising
        :return: np.ndarray: Filtered image.
        """

        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array.")

        if rolling_ball:
            if not isinstance(rolling_ball_radius, int) or rolling_ball_radius <= 0:
                raise ValueError("Rolling ball radius must be a positive integer.")
            image = restoration.rolling_ball(image, radius=rolling_ball_radius)

        if non_local_means:
            if non_local_means_params is None:
                non_local_means_params = {}
            if not isinstance(non_local_means_params, dict):
                raise ValueError("Non-local means parameters must be a dictionary.")
            image = restoration.denoise_nl_means(image, **non_local_means_params)

        return image

    def segmentation(self, **kwargs):
        """
        Generate segmentation for the cell instance.
        :param kwargs: Additional parameters for segmentation.
        :return: Dictionary with tag as key and segmentation result as value.
        """
        if not self.loader_checkpoint:
            raise ValueError("Loader checkpoint is not set. Please run the loader method first.")

        for tag, image in self.images_for_segmentation.items():
            try:
                model = self.segmentation_models.get(tag)
                if model is None:
                    raise ValueError(f"No segmentation model found for tag {tag}.")
                if isinstance(model, SegmentationModel):
                    self.segmentations[tag] = model.segment(image, **kwargs)
                self.logs[tag].update({"Model": model.friendly_name,"Image Shape": image.shape,"Segmentation Shape": self.segmentations[tag].shape,"Segmentation Info": model.get_model_info()})
            except Exception as e:
                print(f"Error during segmentation for tag {tag}: {e}")
                self.logs[tag].update({"Error": str(e)})

    def main(self, loader_dict:dict = None, model_handler:ModelHandler = None, **kwargs):
        """
        Main method to run the cell instance.
        This method will call the
        """
        #TODO: Implement main method to run the cell instance. After segmentation and before feature extraction, check
        # if the segmentation throw out segmentations that are not in the cell boundary