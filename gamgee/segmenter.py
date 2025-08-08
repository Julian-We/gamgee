import os
import re
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from .utils import upsampling
from skimage.transform import resize
from skimage import restoration
import numpy as np
import matplotlib.pyplot as plt



class SegmentationModel:
    def __init__(self, path, model_type="vit_b_lm", upsampling_factor=None, friendly_name=None, cell_compartment=None):
        if path is not None and os.path.exists(path):
            self.checkpoints_root = path if not path.endswith(os.path.sep) else path[:-1]
            self.checkpoints = os.path.join(self.checkpoints_root, 'best.pt') if path is not None else None
        else:
            self.checkpoints_root = None
            self.checkpoints = None
        self.checkpoints_name = os.path.basename(self.checkpoints_root) if self.checkpoints_root else model_type
        self.model_type = model_type
        if upsampling_factor is None:
            # Find the upsampling factor from the checkpoint name "up5" in a checkoints.split('_')
            s = self.checkpoints_name
            print(s)
            match = re.search(r'up(\d+)', s)
            if match:
                self.upsampling_factor = int(match.group(1))
            else:
                raise ValueError("No upsampling factor found in the checkpoint name.")
        else:
            self.upsampling_factor = upsampling_factor
        self.model_type = model_type
        if path is None:
            print("No checkpoint path provided, using default model.")
            self.predictor, self.segmenter = get_predictor_and_segmenter(
                model_type=self.model_type
            )
        else:
            self.predictor, self.segmenter = get_predictor_and_segmenter(
                checkpoint=self.checkpoints,
                model_type=self.model_type
            )

        self.friendly_name = friendly_name if friendly_name else self.checkpoints_name
        self.cell_compartment = cell_compartment.lower() if cell_compartment else None

    def segment(self, image, foreground_smoothing=2.0):
        """
        Perform automatic instance segmentation on the input image.

        :param image: Path to the input image.
        :param foreground_smoothing: Smoothing factor for the foreground.
        :return: Segmentation mask.
        """

        if self.upsampling_factor == 1:
            # Return segmentation directly
            return automatic_instance_segmentation(
                predictor=self.predictor,
                segmenter=self.segmenter,
                input_path=image,
                foreground_smoothing=foreground_smoothing
            )
        elif self.upsampling_factor > 1:
            # Upsample the image, perform segmentation and downsample the result
            segmentation = automatic_instance_segmentation(
                predictor=self.predictor,
                segmenter=self.segmenter,
                input_path=upsampling(image, self.upsampling_factor),
                foreground_smoothing=foreground_smoothing
            )

            return resize(
                segmentation,
                output_shape=image.shape,
                order=0,  # Nearest neighbor interpolation
                preserve_range=True,
                anti_aliasing=False
            ).astype(segmentation.dtype)
        else:
            raise ValueError(f"{self.upsampling_factor} is not a valid upsampling factor.")

    def get_model_info(self):
        """
        Get information about the segmentation model.

        :return: Dictionary containing model information.
        """
        return {
            "checkpoints_root": self.checkpoints_root,
            "checkpoints": self.checkpoints,
            "checkpoints_name": self.checkpoints_name,
            "friendly_name": self.friendly_name,
            "model_type": self.model_type,
            "upsampling_factor": self.upsampling_factor
        }


class SegmentationInstance:
    def __init__(self, image, segmentation_model: SegmentationModel,
                 instant_prediction=True,
                 use_background_substration=False,
                 rolling_ball_radius=60):
        """ Initialize the SegmentationInstance with an image and a segmentation model.
        :param image: Input image as a 2D numpy array.
        :param segmentation_model: An instance of SegmentationModel.
        :param instant_prediction: If True, perform segmentation immediately.
        :param use_background_substration: If True, apply background subtraction using a rolling ball algorithm.
        """

        self.image = image
        if image.ndim != 2:
            raise ValueError("Image must be a 2D array.")

        if not use_background_substration:
            bg = restoration.rolling_ball(image, radius=rolling_ball_radius)
            self.image_filtered = image - bg
        else:
            self.image_filtered = image

        if instant_prediction:
            self.segmentation = segmentation_model.segment(self.image_filtered, foreground_smoothing=2.0)
            if self.segmentation is None:
                raise ValueError("Segmentation failed, returned None.")
        else:
            self.segmentation = None

        self.segmentation_model_info = segmentation_model.get_model_info()

    def get_segmentation_comparison(self, segmentation_model_dict, fig_scaling=5.0, export_path=None):
        """
        Compare the segmentation with other models.

        :param segmentation_model_dict: Dictionary of segmentation models to compare against.
        :param export_path: Path to save the comparison figure. If None, the figure will not be saved.
        :param fig_scaling: Scaling factor for the figure size.
        :return: Dictionary of segmentation results.
        """
        comparison_results = {}
        for model_name, model in segmentation_model_dict.items():
            comparison_results[model_name] = model.segment(self.image, foreground_smoothing=2.0)

        fig, axs = plt.subplots(1, len(comparison_results) + 1,
                                figsize=(fig_scaling * len(comparison_results), len(comparison_results)))
        axs[0].imshow(self.image, cmap='gray')
        axs[0].set_title('Original Image')
        for i, (model_name, segmentation) in enumerate(comparison_results.items(), start=1):
            axs[i].imshow(segmentation, cmap='nipy_spectral')
            axs[i].set_title(model_name)
        for ax in axs:
            ax.axis('off')
        # plt.tight_layout()
        if export_path:
            plt.savefig(os.path.join(export_path, f'{hex(np.random.randint(0, 10000))}.png'), dpi=300)
            plt.close(fig)

        self.segmentation = comparison_results

    def segment(self, segmentation_model: SegmentationModel, foreground_smoothing=2.0):
        """
        Perform automatic instance segmentation on the input image using the provided segmentation model.

        :param segmentation_model: An instance of SegmentationModel.
        :param foreground_smoothing: Smoothing factor for the foreground.
        :return: Segmentation mask.
        """
        self.segmentation = segmentation_model.segment(self.image_filtered, foreground_smoothing=foreground_smoothing)
        if self.segmentation_model_info != segmentation_model.get_model_info():
            self.segmentation_model_info = segmentation_model.get_model_info()
        return self.segmentation
