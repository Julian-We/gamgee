from .utils.denoising import encode_memmap_info, get_memmap_info
import tempfile
import numpy as np
import gamgee
from pathlib import Path
import subprocess


def generate_tempfile(marker):
    # Create temporary file for the image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
    temp_path = temp_file.name
    temp_file.close()

    # Create memory-mapped array and write data
    mmap_array = np.memmap(temp_path, dtype=marker.raw_image.dtype, mode='w+', shape=marker.raw_image.shape)
    mmap_array[:] = marker.raw_image[:]
    mmap_array.flush()

    # Encode memmap information into an identifier string
    identifier_string = encode_memmap_info(uid=marker.uid,
                                           filepath=temp_path,
                                           shape=marker.raw_image.shape,
                                           dtype=str(marker.raw_image.dtype))
    marker.mmap_identifier = identifier_string
    return identifier_string

def care_denoising(mmap_strings: list[str], model_name='20250812_JW_granule_25'):
    path_denoisepy = Path(gamgee.__file__).parent / 'caredenoising.py'

    mmap_strings_quoted = [f'"{mmapstring}"' for mmapstring in mmap_strings]
    cmd = f'conda run -n csbdeep python "{path_denoisepy}" {" ".join(mmap_strings_quoted)} --model {model_name}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        raise RuntimeError(f"CARE denoising failed: {result.stderr}")


    result_identifiers = []
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_IDENTIFIER:"):
            _, identifier = line.split(":", 1)
            result_identifiers.append(identifier.strip())

    if len(result_identifiers) != len(mmap_strings):
        raise ValueError("Number of result identifiers does not match number of input memmap strings.")
    else:
        return result_identifiers


def denoise_with_care(markers: list, model_name='20250812_JW_granule_25'):
    mmap_strings = [generate_tempfile(marker) for marker in markers]
    result_identifiers = care_denoising(mmap_strings, model_name=model_name)
    memmap_infos = list(map(get_memmap_info, result_identifiers))


    for marker, memmap_info in zip(markers, memmap_infos):
        if marker.uid != memmap_info['uid']:
            print(f"Warning: UID mismatch for marker {marker.uid} and result {memmap_info['uid']}. Searching for correct marker.")
            for res in memmap_infos:
                if res['uid'] == marker.uid:
                    memmap_info = res
                    break
        denoised_memmap = np.memmap(memmap_info['filepath'], dtype=memmap_info['dtype'], mode='r', shape=memmap_info['shape'])
        marker.denoised_image = np.array(denoised_memmap)
        marker.denoised_mmap_identifier = memmap_info['uid']
        # Clean up the temporary input memmap file
        try:
            Path(memmap_info['filepath']).unlink()
        except Exception as e:
            print(f"Warning: Could not delete temporary file {memmap_info['filepath']}: {e}")
        marker.logs['Preprocessing'].append(f"Denoised with CARE model {model_name}")
