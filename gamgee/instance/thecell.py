import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .marker import Marker
from gamgee.denoising_interface import denoise_with_care
from .modelhandler import ModelHandler

class TheCell:
    def __init__(self, root_path: str, model_handler: ModelHandler, name=None, blacklist=None, **kwargs):
        """Initialize a cell instance.
        Args:
            name (str): Name of the cell.
            blacklist (list): List of folder names to ignore when scanning for markers.
        """
        self.root = Path(root_path)
        self.output_root = self.root / 'output'
        self.output_root.mkdir(exist_ok=True, parents=True)
        self.name = name if name is not None else self.root.name
        self.cell_id = uuid.uuid4().hex

        # Default blacklist for common non-marker folders
        if blacklist is None:
            blacklist = ['.git', '__pycache__', '.DS_Store', 'models', 'utils', 'temp', 'cache', 'logs',
                         'export', 'results', 'denoised', 'segmentations', 'masks', 'output', 'xprt', 'raw']
        self.blacklist = blacklist

        self.logs = {
            "Name": self.name,
            "Cell ID": self.cell_id,
        }

        self.markers = {}
        self.care_denoising_models = {}
        self.markers_compartments = {

        }
        self.model_handler = model_handler

        # Scan for marker folders and create Marker objects
        self._scan_and_create_markers()

        self.denoise()
        self.plot_markers_and_segmentations()


    def _scan_and_create_markers(self):
        """Scan root directory for folders containing single image files and create Marker objects."""
        if not self.root.exists():
            raise ValueError(f"Root path {self.root} does not exist.")

        valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif']

        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue

            # Skip blacklisted folders
            if folder.name in self.blacklist:
                continue

            # Find image files in the folder
            image_files = [f for f in folder.iterdir()
                          if f.is_file() and f.suffix.lower() in valid_extensions]

            # Only create marker if exactly one image file is found
            if len(image_files) == 1:
                try:
                    marker = Marker(
                        name=folder.name,
                        parent_name=self.name,
                        parent_id=self.cell_id,
                        parent_root=self.root,
                        model_handler=self.model_handler
                    )
                    self.markers[folder.name] = marker
                    self.logs[f"Marker_{folder.name}"] = "Created successfully"
                except Exception as e:
                    self.logs[f"Marker_{folder.name}_Error"] = str(e)
            elif len(image_files) == 0:
                self.logs[f"Folder_{folder.name}"] = "No image files found"
            else:
                self.logs[f"Folder_{folder.name}"] = f"Multiple image files found ({len(image_files)})"

    def denoise(self, use_tv_denoising=False, max_workers=None):
        """Denoise all markers in parallel.

        Args:
            use_tv_denoising (bool): If True, use TV denoising instead of CARE models
            max_workers (int): Maximum number of parallel workers. If None, uses number of CPU cores
        """
        if not self.markers:
            raise ValueError("No markers found to process.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_marker = {}
            for marker in self.markers.values():
                if use_tv_denoising or marker.denoising_model_name is None:
                    future = executor.submit(self._denoise_marker_tv, marker)
                else:
                    future = executor.submit(self._denoise_marker_care, marker)
                future_to_marker[future] = marker.name

            for future in as_completed(future_to_marker):
                marker_name = future_to_marker[future]
                try:
                    future.result()
                    self.logs[f"Denoising_{marker_name}"] = "Denoising completed successfully"
                except Exception as e:
                    self.logs[f"Denoising_{marker_name}_Error"] = str(e)




    def _denoise_marker_care(self, marker):
        """Denoise a single marker using CARE model."""
        if marker.denoising_model_name is None:
            # If no CARE model specified, fall back to TV denoising
            self._denoise_marker_tv(marker)
        else:
            # Use CARE denoising for single marker
            denoise_with_care([marker], model_name=marker.denoising_model_name)

    def _denoise_marker_tv(self, marker):
        """Denoise a single marker using TV denoising."""
        marker.tv_denoising()

    def plot_markers_and_segmentations(self):
        import matplotlib.pyplot as plt
        num_markers = len(self.markers)
        if num_markers == 0:
            print("No markers to plot.")
            return

        num_markers = len(self.markers)
        fig, axes = plt.subplots(num_markers, 3, figsize=(15, 5 * num_markers))
        for i, (marker_name, marker) in enumerate(self.markers.items()):
            axes[i, 0].imshow(marker.raw_image, cmap='gray')
            axes[i, 0].set_title(f"{marker.name} - Raw Image")
            axes[i, 0].axis('off')

            if marker.denoised_image is not None:
                axes[i, 1].imshow(marker.denoised_image, cmap='gray')
                axes[i, 1].set_title(f"{marker.name} - Denoised Image")
            else:
                axes[i, 1].text(0.5, 0.5, 'No Denoised Image', horizontalalignment='center', verticalalignment='center')
            axes[i, 1].axis('off')

            if marker.segmentation is not None:
                axes[i, 2].imshow(marker.segmentation, cmap='nipy_spectral')
                axes[i, 2].set_title(f"{marker.name} - Segmentation")
            else:
                axes[i, 2].text(0.5, 0.5, 'No Segmentation', horizontalalignment='center', verticalalignment='center')
            axes[i, 2].axis('off')

        # save as png
        plt.tight_layout()
        plt.savefig(self.output_root / f"{self.name}_markers_segmentations.png", dpi=300)