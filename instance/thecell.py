import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .marker import Marker
from gamgee.denoising_interface import denoise_with_care
from .segmentations import clean_cell_segmentations, delete_outside_objects
from .modelhandler import ModelHandler
import tifffile as tiff
import pickle


def load_instance(instance_path: str):
    """Load a TheCell instance from a pickle file."""
    instance_path = Path(instance_path)
    if not instance_path.exists():
        raise ValueError(f"Instance path {instance_path} does not exist.")
    with open(instance_path, 'rb') as f:
        loaded_instance = pickle.load(f)
    return loaded_instance


class TheCell:
    def __init__(self, root_path: str|Path, model_handler: ModelHandler,
                 name=None, blacklist=None, refine_segmentations=True, uid=None, **kwargs):
        """Initialize a cell instance.
        Args:
            name (str): Name of the cell.
            blacklist (list): List of folder names to ignore when scanning for markers.
            root_path (str): Path to the root directory containing marker folders.
            model_handler (ModelHandler): Instance of ModelHandler to manage segmentation models.
            refine_segmentations (bool): Whether to refine segmentations after denoising.
        """
        self.root = Path(root_path)
        self.image_root = self.root
        self.output_root = self.root / 'output'
        self.output_root.mkdir(exist_ok=True, parents=True)
        self.name = name if name is not None else self.root.name
        self.cell_id = uuid.uuid4().hex if uid is None else uid
        self.all_features = {}

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
        # print(f"Found {len(self.markers)} markers for cell '{self.name}'.")
        self.denoise()
        # if refine_segmentations:
        #     self.refine_segmentations()
        self.plot_markers_and_segmentations()

        self.save_segmentations()

    def _scan_and_create_markers(self):
        """Scan root directory directly for image files. If none found, look for subfolders containing single image files."""
        if not self.root.exists():
            raise ValueError(f"Root path {self.root} does not exist.")

        # Check for "raw" or "MIP" subfolders
        potential_subfolders = ['raw', 'mip', 'mips', 'imgs']
        for subfolder in self.image_root.iterdir():
            if subfolder.is_dir() and subfolder.name.lower() in potential_subfolders:
                self.image_root = subfolder
                print(f"Found subfolder '{subfolder.name}' for images. Updating image root to {self.image_root}")
                break


        valid_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif']

        # Scan for images in root directory
        image_files = [f for f in self.image_root.iterdir()
                       if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith(".")]
        print(f"Found {len(image_files)} image files in root directory {self.image_root}")
        if image_files:
            for image_file in image_files:
                try:
                    marker_name = image_file.stem if not "nls" in image_file.stem.lower() else "nls_nucleus"
                    marker = Marker(
                        image_path=image_file,
                        parent_name=self.name,
                        parent_id=self.cell_id,
                        model_handler=self.model_handler
                    )
                    self.markers[marker_name] = marker
                    self.logs[f"Marker_{marker_name}"] = "Created successfully"
                    if "nls" in marker_name.lower():
                        print(image_file)
                        marker_name = f"{marker_name}_cell"
                        marker = Marker(
                            image_path=image_file,
                            parent_name=self.name,
                            parent_id=self.cell_id,
                            model_handler=self.model_handler,
                            compartment="cell")
                        self.markers[marker_name] = marker
                        self.markers_compartments[marker_name] = "cell"
                        self.logs[f"Marker_{marker_name}"] = "Created as cell compartment based on name"

                except Exception as e:
                    print(e)
                    self.logs[f"Marker_{marker_name}_Error"] = str(e)
        else:
            self.logs["Image.Root"] = "No image files found in root directory"
            print(f"Warning: No image files found in root directory {self.image_root}")

            # Scan for folders containing single image files
            for folder in self.image_root.iterdir():
                if folder.is_dir() and folder.name not in self.blacklist:
                    image_files = [f for f in folder.iterdir()
                                   if f.is_file() and f.suffix.lower() in valid_extensions]
                    if len(image_files) == 1:
                        try:
                            marker_name = folder.name
                            marker = Marker(
                                name=marker_name,
                                parent_name=self.name,
                                parent_id=self.cell_id,
                                parent_root=self.image_root,
                                model_handler=self.model_handler
                            )
                            self.markers[marker_name] = marker
                            self.logs[f"Marker_{marker_name}"] = "Created successfully"
                        except Exception as e:
                            print(e)
                            self.logs[f"Marker_{marker_name}_Error"] = str(e)
                    elif len(image_files) == 0:
                        self.logs[f"Folder_{folder.name}"] = "No image files found"
                        print(f"Warning: No image files found in folder {folder}")
                    else:
                        self.logs[f"Folder_{folder.name}"] = f"Multiple image files found ({len(image_files)})"
                        print(f"Warning: Multiple image files found in folder {folder}, skipping.")


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
                # try:
                future.result()
                self.logs[f"Denoising_{marker_name}"] = "Denoising completed successfully"
                # except Exception as e:
                #     self.logs[f"Denoising_{marker_name}_Error"] = str(e)

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

    def refine_segmentations(self):
        if "cell" in self.markers_compartments.values():
            nucleus_markers = []
            granule_markers = []
            for marker_name, marker_compartment in self.markers_compartments.items():
                if marker_compartment == "cell":
                    cell_marker = self.markers[marker_name]
                    break
                if marker_compartment == "nucleus":
                    nucleus_markers.append(self.markers[marker_name])
                if marker_compartment == "granule":
                    granule_markers.append(self.markers[marker_name])

            # Fuse and/or select cell segmentation
            self.markers['cell'].segmentation = clean_cell_segmentations(self.markers['cell'].segmentation)

            # Kick out nucleus and granule objects outside cell
            for nucleus_marker in nucleus_markers:
                nucleus_marker.segmentation = delete_outside_objects(nucleus_marker.segmentation,
                                                                    self.markers['cell'].segmentation)
            for granule_marker in granule_markers:
                granule_marker.segmentation = delete_outside_objects(granule_marker.segmentation,
                                                                    self.markers['cell'].segmentation)



    def save_segmentations(self):
        """Save segmentation masks for all markers."""
        print(f"The markers are{self.markers.keys()}")
        for marker_name, marker in self.markers.items():
            seg_out_dir = self.root / 'segmentations' / f"{marker_name}.tif"
            seg_out_dir.parent.mkdir(parents=True, exist_ok=True)
            seg_path = seg_out_dir
            tiff.imwrite(seg_path, marker.segmentation.astype('uint16'))


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
        plt.close(fig)

    def save_instance(self):
        """Save the entire TheCell instance to a pickle file."""
        instance_path = self.output_root / f"{self.cell_id}_TheCell_instance.pkl"
        with open(instance_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"TheCell instance saved to {instance_path}")

    def pickle_images(self):
        """Per Marker save raw, denoised and segmentation images, in one pickle file per TheCell"""
        images_data = {}
        images_data["uid"] = self.cell_id
        for marker_name, marker in self.markers.items():
            images_data[marker_name] = {
                "raw_image": marker.raw_image,
                "denoised_image": marker.denoised_image,
                "segmentation": marker.segmentation,
                "compartment" : marker.compartment,
            }
        images_path = self.output_root / f"{self.cell_id}_TheCell_images.pkl"
        with open(images_path, 'wb') as f:
            pickle.dump(images_data, f)
        print(f"Marker images saved to {images_path}")

