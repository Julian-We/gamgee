from multiprocessing import shared_memory
from pathlib import Path
import numpy as np
import argparse
import subprocess
import sys
from pathlib import Path


def check_environment_exists(env_name):
    """Check if conda environment exists."""
    try:
        result = subprocess.run(['conda', 'env', 'list'],
                                capture_output=True, text=True, check=True)
        return env_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def get_yml_path():
    """Get path to csb_deep.yml from gamgee.environments."""
    try:
        import gamgee.environments
        print()  # Ensure the module is loaded
        # Get the path to the csb_deep.yml file
        env_module_path = Path(gamgee.__file__).parent / 'environments'
        yml_path = env_module_path / 'csb_deep.yml'
        return yml_path if yml_path.exists() else None
    except ImportError:
        return None


def create_environment(env_name, yml_path, use_mamba=False):
    """Create conda environment from yml file."""
    cmd_tool = 'mamba' if use_mamba else 'conda'
    cmd = [cmd_tool, 'env', 'create', '-f', str(yml_path), '-n', env_name]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def resolve_environment(env_name='csbdeep'):
    if check_environment_exists(env_name):
        pass
    else:
        yml_path = get_yml_path()
        if not yml_path:
            print("Could not find csb_deep.yml in gamgee.environments")
            sys.exit(1)

        print(f"Creating environment '{env_name}'...")

        # Try mamba first, then conda
        success = False
        for tool, use_mamba in [('mamba', True), ('conda', False)]:
            print(f"Trying {tool}...")
            if create_environment(env_name, yml_path, use_mamba):
                print(f"Successfully created environment '{env_name}' using {tool}")
                success = True
                break
            else:
                print(f"Failed to create environment using {tool}")

        if not success:
            print("Failed to create environment with both mamba and conda")
            sys.exit(1)



def get_shared_memory_info(identifier_str:str):
    """
    Get information about the shared memory segment.

    :param identifier_str: Identifier string for the shared memory segment.
    :return: Dictionary containing shared memory information.
    """
    shm_name, shape_raw, dtype = identifier_str.split('-')
    shape = tuple(map(int, shape_raw.split('_')))

    return {
        "name": shm_name,
        "shape": shape,
        "dtype": dtype
    }


def convert_to_uint16(image):
    """
    Convert image to uint16 type.

    Args:
        image: Input image array.

    Returns:
        Converted image as uint16.
    """
    return (image / image.max() * 65535).astype(np.uint16)

resolve_environment('csbdeep')

parser = argparse.ArgumentParser(description="CARE Denoising Script")
parser.add_argument('files', nargs='*', help='List of image identifier strings for shared memory segments')
parser.add_argument('--model', '-m', default='20250812_JW_granule_25', help='CARE model name to use')

args = parser.parse_args()

if not args.files:
    print("No image identifier strings provided. Exiting.")
    exit(1)

this_dir = Path(__file__).parent
model_base_dir = this_dir / 'models' / 'care'

# Create CARE model instance
care_model = CARE(config=None, name=args.model, basedir=model_base_dir)

for identifier_string in args.files:
    shm_info = get_shared_memory_info(identifier_string)
    shm = shared_memory.SharedMemory(name=shm_info['name'])
    print(
        f"Processing shared memory segment: {shm_info['name']} with shape {shm_info['shape']} and dtype {shm_info['dtype']}")

    # Create a numpy array from the shared memory segment
    image_shm = np.ndarray(shm_info['shape'], dtype=shm_info['dtype'], buffer=shm.buf)

    # Copy the image data to a new array for processing
    image_data = np.copy(image_shm)

    # Denoise the image using the CARE model
    denoised_image = care_model.predict(image_data, axes='YX')
    img_int = convert_to_uint16(denoised_image)

    # CREATE NEW SHARED MEMORY FOR RESULT
    result_shm = shared_memory.SharedMemory(create=True, size=img_int.nbytes)
    result_array = np.ndarray(img_int.shape, dtype=img_int.dtype, buffer=result_shm.buf)
    result_array[:] = img_int[:]

    # Format result identifier string in same format as input
    result_identifier = f"{result_shm.name}-{img_int.shape[0]}_{img_int.shape[1]}-{str(img_int.dtype)}"

    print(f"RESULT_IDENTIFIER:{result_identifier}")

    # Close connections but don't unlink - notebook will handle cleanup
    shm.close()
    # result_shm.close()

    print(f"Finished processing {identifier_string}")


