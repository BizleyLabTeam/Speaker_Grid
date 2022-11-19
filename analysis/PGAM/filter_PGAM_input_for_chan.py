""" 
This function takes an npz file containing spike counts and variables that 
has been prepared for many neurons / units and returns a smaller version 
with just spike counts for one channel.

This is an intermediate step in the PGAM development, as we don't yet know 
how to model counts from multiple different units (but hopefully one day 
we will)

 """

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

 # Settings
analysis = 'temporal'
ferret = 'F1901_Crumble'
primary_chan = 'B09'     # A = left, B = right hemisphere
secondary_chan = 'A22'      # Optional, set to None if 

file_name_options = {                       # File name depends on which analysis we're doing
    'Legacy': 'example_pgam_concatn.npz',
    'SRF': 'SRF_pgam_concatn.npz',
    'temporal': 'temporal_pgam_concatn.npz'
}


###############################
# configuration
def create_config(order:int, bin_width:float, primary_chan:str, secondary_chan:Optional[str]=None):
    """ 
    
    Args:
        Order: number of coefficient of the polynomials that make up splines
        bin_width: sample period in seconds
    
     """

    # Renaming neurons for consistency with source tutorials
    primary_neuron = f'neuron_{primary_chan}'

    # Keep for when instituting continuous head tracking
    angle_knots = np.linspace(-np.pi, np.pi, 7)
    angle_knots = [float(k) for k in angle_knots]

    cfg = {
        'events':           # click sounds
        {
            'lam':10,
            'penalty_type':'der',
            'der':2,
            'knots': np.nan,
            'order':order,
            'is_temporal_kernel': True,
            'is_cyclic': [False],
            'knots_num': 20,
            'kernel_length': 100,
            'kernel_direction': 0,
            'samp_period':bin_width 
        },
        primary_neuron:
        {
            'lam':10,
            'penalty_type':'der',
            'der':2,
            'knots': np.nan,
            'order':order,
            'is_temporal_kernel': True,
            'is_cyclic': [False],
            'knots_num': 8,
            'kernel_length': 201,
            'kernel_direction': 1,
            'samp_period':bin_width 
        }
    }

    # Optional extras for comparison between neurons
    if secondary_chan:
        secondary_neuron = f'neuron_{secondary_chan}'
        cfg[secondary_neuron] = {
            'lam':10,
            'penalty_type':'der',
            'der':2,
            'knots': np.nan,
            'order':order,
            'is_temporal_kernel': True,
            'is_cyclic': [False],
            'knots_num': 12,
            'kernel_length': 201,
            'kernel_direction': 0,
            'samp_period':bin_width 
        }

    return cfg

def save_config(file_path, analysis:str, primary_chan:str, secondary_chan:str, cov_dict:dict) -> None:
    """ Write configuration to YAML file """

    if secondary_chan:
        file_name = f'config_{analysis}_{primary_chan}_{secondary_chan}.yml'
    else:
        file_name = f'config_{analysis}_{primary_chan}.yml'

    with open(file_path / file_name, 'w') as outfile:
        yaml.dump(cov_dict, outfile, default_flow_style=False)


##############################################
def main():

    # Paths
    data_path = Path.cwd() / 'data' / f"{ferret}_Squid"

    input_file =  file_name_options[analysis]

    # Load data
    data = np.load(data_path / input_file, allow_pickle=True)

    # Get column of index to keep
    idx = np.where(data['neu_names'] == primary_chan)[0][0]
    spike_counts = data['counts'][:,idx].reshape(-1,1)

    # Filter info
    all_info = data['neu_info'].all()
    
    neu_key = f"neuron_{primary_chan}"
    neu_info = {neu_key:all_info[primary_chan]}

    output_name = input_file.replace('concatn',primary_chan)

    # Optional extras for comparing pairs of neurons
    if secondary_chan:

        # Add secondary channels to neuron list and metadata
        neu_key = f"neuron_{secondary_chan}"
        neu_info[neu_key] = all_info[secondary_chan]

        # Append spike counts
        idx = np.where(data['neu_names'] == secondary_chan)[0][0]
        spike_counts = np.hstack((spike_counts, data['counts'][:,idx].reshape(-1,1)))

        # Append secondary channel to file name
        output_name = output_name.replace('.npz',f"_{secondary_chan}.npz")

    # Save for processing in Docker
    np.savez(
        data_path / output_name, 
        counts = spike_counts,
        variables = data['variables'],
        variable_names = data['variable_names'],
        neu_names = list(neu_info.keys()),
        neu_info = neu_info, 
        trial_ids = data['trial_ids'])

    # Save config file
    order = 4   # the order of the spline is the number of coefficient of the polynomials
    bin_width = data['bin_width'].tolist()

    cov_dict = create_config(order, bin_width, primary_chan, secondary_chan)
    save_config(data_path, analysis, primary_chan, secondary_chan, cov_dict)
    
    


if __name__ == '__main__':
    main()
    