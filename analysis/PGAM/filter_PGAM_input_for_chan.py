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
ferret = 'F1901_Crumble'
channel = 'A12'     # A = left, B = right hemisphere

# Paths
data_path = Path.cwd() / 'data' / f"{ferret}_Squid"

all_chan_file = data_path / 'example_pgam_concatn.npz'

# Load data
data = np.load(all_chan_file, allow_pickle=True)

# Get column of index to keep
idx = np.where(data['neu_names'] == channel)[0][0]

spike_counts = data['counts'][:,idx].reshape(-1,1)

# Filter info
neu_info = data['neu_info'].all()
neu_key = f"neuron_{channel}"
neu_info = {neu_key:neu_info[channel]}

# Save for processing in Docker
output_file = data_path / f'example_pgam_{channel}.npz'
np.savez(
    output_file, 
    counts = spike_counts,
    variables = data['variables'],
    variable_names = data['variable_names'],
    neu_names = [neu_key],
    neu_info = neu_info, 
    trial_ids = data['trial_ids'])