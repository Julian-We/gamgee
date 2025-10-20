import subprocess
import sys
from pathlib import Path

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


def encode_memmap_info(uid, filepath, shape, dtype, delimiter='^', shape_delimiter='_'):
    """
    Encode memmap information into an identifier string.

    :param filepath: Path to the memmap file.
    :param shape: Tuple containing the shape of the array.
    :param dtype: Data type of the array.
    :param delimiter: Character to separate filepath, shape, and dtype (default: '-').
    :param shape_delimiter: Character to separate shape dimensions (default: '_').
    :param uid: Unique identifier to ensure uniqueness of the memmap file.
    :return: Identifier string for the memmap file.
    """
    shape_str = shape_delimiter.join(map(str, shape))
    return f"{filepath}{delimiter}{shape_str}{delimiter}{dtype}{delimiter}{uid}"


def get_memmap_info(identifier_str: str, delimiter='^', shape_delimiter='_'):
    """
    Get information about the memmap file.

    :param identifier_str: Identifier string for the memmap file.
    :param delimiter: Character that separates filepath, shape, and dtype (default: '-').
    :param shape_delimiter: Character that separates shape dimensions (default: '_').
    :return: Dictionary containing memmap info.
    """
    # identifier_str: filepath-shape-dtype
    filepath, shape_raw, dtype, uid = identifier_str.split(delimiter)
    shape = tuple(map(int, shape_raw.split(shape_delimiter)))
    return {
        "filepath": filepath,
        "shape": shape,
        "dtype": dtype,
        "uid": uid
    }

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
    except Exception as e:
        print(e)
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

resolve_environment()
