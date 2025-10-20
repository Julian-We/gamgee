from pathlib import Path
import numpy as np
import argparse
import subprocess
from csbdeep.models import CARE
import sys
from pathlib import Path
import tempfile
from utils.denoising import encode_memmap_info, get_memmap_info



# def encode_memmap_info(uid, filepath, shape, dtype, delimiter='^', shape_delimiter='_'):
#     """
#     Encode memmap information into an identifier string.
#
#     :param filepath: Path to the memmap file.
#     :param shape: Tuple containing the shape of the array.
#     :param dtype: Data type of the array.
#     :param delimiter: Character to separate filepath, shape, and dtype (default: '-').
#     :param shape_delimiter: Character to separate shape dimensions (default: '_').
#     :return: Identifier string for the memmap file.
#     """
#     shape_str = shape_delimiter.join(map(str, shape))
#     return f"{filepath}{delimiter}{shape_str}{delimiter}{dtype}{delimiter}{uid}"
#
# def get_memmap_info(identifier_str: str, delimiter='^', shape_delimiter='_'):
#     """
#     Get information about the memmap file.
#
#     :param identifier_str: Identifier string for the memmap file.
#     :param delimiter: Character that separates filepath, shape, and dtype (default: '-').
#     :param shape_delimiter: Character that separates shape dimensions (default: '_').
#     :return: Dictionary containing memmap info.
#     """
#     # identifier_str: filepath-shape-dtype
#     filepath, shape_raw, dtype, uid = identifier_str.split(delimiter)
#     shape = tuple(map(int, shape_raw.split(shape_delimiter)))
#     return {
#         "filepath": filepath,
#         "shape": shape,
#         "dtype": dtype,
#         "uid": uid
#     }


def convert_to_uint16(image):
    """
    Normalize and convert image to uint16.
    """
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        img_inte = np.zeros_like(image, dtype=np.uint16)
    else:
        img_normalized = (image - img_min) / (img_max - img_min)
        img_inte = (img_normalized * 65535).astype(np.uint16)
    return img_inte


parser = argparse.ArgumentParser(description="CARE Denoising Script")
parser.add_argument('files', nargs='+', help='List of image identifier strings for memmap files (space-separated)')
parser.add_argument('--model', '-m', default='20250812_JW_granule_25', help='CARE model name to use')

args = parser.parse_args()

if not args.files:
    print("No image identifier strings provided. Exiting.")
    exit(1)

this_dir = Path(__file__).parent
model_base_dir = this_dir / 'models' / 'care'
print(f"files to process: {args.files}")
# Create CARE model instance
care_model = CARE(config=None, name=args.model, basedir=model_base_dir)

for identifier_string in args.files:

    memmap_info = get_memmap_info(identifier_string)

    print(
        f"Processing memmap file: {memmap_info['filepath']} with shape {memmap_info['shape']} and dtype {memmap_info['dtype']}")

    # Load image from memmap file
    image_memmap = np.memmap(memmap_info['filepath'], dtype=memmap_info['dtype'], mode='r', shape=memmap_info['shape'])
    image_data = np.array(image_memmap)

    # Denoise the image using the CARE model
    denoised_image = care_model.predict(image_data, axes='YX')
    img_int = convert_to_uint16(denoised_image)

    # Write result to a new memmap file using tempfile
    tmp_result = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
    result_memmap = np.memmap(tmp_result.name, dtype=img_int.dtype, mode='w+', shape=img_int.shape)
    result_memmap[:] = img_int[:]
    result_memmap.flush()

    # Format result identifier string
    result_identifier = encode_memmap_info(memmap_info['uid'], tmp_result.name, img_int.shape, str(img_int.dtype))

    print(f"RESULT_IDENTIFIER:{result_identifier}")

    print(f"Finished processing {identifier_string}")

    # Clean up the temporary input memmap file
    try:
        Path(memmap_info['filepath']).unlink()
    except Exception as e:
        print(f"Warning: Could not delete temporary file {memmap_info['filepath']}: {e}")