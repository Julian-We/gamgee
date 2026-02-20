import sqlite3 as sql
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import pickle
import sys
sys.path.append('..')
from gamgee.instance import ModelHandler, TheCell

#%%
root = Path('/Volumes/HELHEIM/analyzed_data/size_and_periphery/iPGC_24hpf_tdrd7a-modulation')


mh = ModelHandler()

for sample_dir in root.iterdir():
    if not sample_dir.is_dir() or sample_dir.name.startswith('.'):
        continue
    uid, condition, filename = sample_dir.name.split('__')
    print(f"Processing {uid} - {condition}")
    tc = TheCell(
        uid=uid,
        name=filename,
        root_path=sample_dir,
        model_handler=mh,
    )
    # tc.save_instance()
    tc.save_segmentations()
    break  # Process only the first sample for testing


print(tc.markers.keys())