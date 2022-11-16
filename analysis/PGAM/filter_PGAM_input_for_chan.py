""" 
This function takes an npz file containing spike counts and variables that 
has been prepared for many neurons / units and returns a smaller version 
with just spike counts for one channel.

This is an intermediate step in the PGAM development, as we don't yet know 
how to model counts from multiple different units (but hopefully one day 
we will)

 """

from pathlib import Path

import numpy as np

 # Settings
analysis = 'SRF'
ferret = 'F1901_Crumble'
channel = 'B09'     # A = left, B = right hemisphere

# Paths
data_path = Path.cwd() / 'data' / f"{ferret}_Squid"

file_name_options = {                       # File name depends on which analysis we're doing
    'Legacy': 'example_pgam_concatn.npz',
    'SRF': 'SRF_pgam_concatn.npz',
    'Temporal': 'NOT_IMPLEMENTED_YET'
}

input_file =  file_name_options[analysis]

# Load data
data = np.load(data_path / input_file, allow_pickle=True)

# Get column of index to keep
idx = np.where(data['neu_names'] == channel)[0][0]

spike_counts = data['counts'][:,idx].reshape(-1,1)

# Filter info
neu_info = data['neu_info'].all()
neu_key = f"neuron_{channel}"
neu_info = {neu_key:neu_info[channel]}

# Save for processing in Docker
output_name = input_file.replace('concatn',channel)

np.savez(
    data_path / output_name, 
    counts = spike_counts,
    variables = data['variables'],
    variable_names = data['variable_names'],
    neu_names = [neu_key],
    neu_info = neu_info, 
    trial_ids = data['trial_ids'])