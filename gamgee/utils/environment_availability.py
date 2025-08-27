import subprocess
from pathlib import Path
import sys


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
        env_module_path = Path(gamgee.__file__).parent.parent / 'environments'
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
