from pathlib import Path
from gamgee.segmenter import SegmentationModel
import threading


# Get the directory where this file is located
_CURRENT_DIR = Path(__file__).parent
_MODELS_DIR = _CURRENT_DIR.parent / "models" / "msam"

class ModelHandler:
    def __init__(self, **kwargs):
        self.segmentation_models = {}
        self.preconfigureations = {
        "base": dict(path=None,
                                  model_type='vit_b_lm',
                                  friendly_name='µSAM Base model',
                                  upsampling_factor=1),
        "large": dict(path=None,
                                   model_type='vit_l_lm',
                                   friendly_name='µSAM Large model',
                                   upsampling_factor=1),
        "cell_membrane": dict(path=None,
                                           model_type='vit_l_lm',
                                           friendly_name='µSAM Cell (Membrane) model',
                                           upsampling_factor=1,
                                           cell_compartment='cell'),
        "nucleus_nls":      dict(path=_MODELS_DIR / 'nls_nucleus' / 'sam_large_blobs_up1_35464802',
                                         model_type='vit_l_lm',
                                         friendly_name='µSAM Nucleus (NLS) model',
                                         cell_compartment='nucleus',
                                         upsampling_factor=1),
        "cell_nls": dict(path=_MODELS_DIR / 'nls_cell' / 'sam_large_blobs_up1_35464802',
                                      model_type='vit_l_lm',
                                      friendly_name='µSAM Cell (NLS) model',
                                      cell_compartment='cell',
                                      upsampling_factor=1),
        "granules": dict(path=_MODELS_DIR / 'granules' / 'sam_granules_refined_up3_35416497',
                                      model_type='vit_l_lm',
                                      friendly_name='µSAM Granules model',
                                      cell_compartment='granules',
                                      upsampling_factor=3),
        }

        self._model_locks = {model_name: threading.Lock() for model_name in self.segmentation_models.keys()}
    def __getattr__(self, name):
        """Allow direct access to models as attributes.

        Example: mh.granules instead of mh.segmentation_models.get('granules')
        """
        if name in self.segmentation_models or name in self.preconfigureations:
            if name not in self.segmentation_models:
                self._model_locks[name] = threading.Lock()
                config = self.preconfigureations[name]
                self.segmentation_models[name] = SegmentationModel(Path(config['path']),
                                                                   model_type=config.get('model_type', 'vit_b_lm'),
                                                                   friendly_name=config.get('friendly_name', name),
                                                                   cell_compartment=config.get('cell_compartment', None),
                                                                   upsampling_factor=config.get('upsampling_factor', None))
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
        self.segmentation_models[model_name] = SegmentationModel(Path(model_path),
                                                                 model_type=model_type,
                                                                 friendly_name=friendly_name,
                                                                 cell_compartment=cell_compartment)
        self._model_locks[model_name] = threading.Lock()
    def get_available_models(self):
        """Get a list of available segmentation models.

        Returns:
            list: List of model names.
        """
        return list(self.segmentation_models.keys())

    def get_model_lock(self, model_name: str):
        """Get the lock for a specific model."""
        return self._model_locks.get(model_name, threading.Lock())

    def get_model_by_compartment(self):
        """Get a dictionary of models grouped by cell compartment.

        Returns:
            dict: Dictionary with compartment names as keys and lists of model names as values.
        """
        compartment_dict = {}
        for model_name, model in self.segmentation_models.items():
            compartment = model.cell_compartment if hasattr(model, 'cell_compartment') else 'unknown'
            if compartment not in compartment_dict:
                compartment_dict[compartment] = []
            compartment_dict[compartment].append(model_name)
        return compartment_dict