

from pathlib import Path


import yaml

# create and save the YAML with the fit list (1 fit only)
fit_dict = {
    'experiment_ID': ['exp_1'],
    'session_ID': ['session_A'],
    'neuron_num': [0],
    'path_to_input': ['/input/example_pgam_A12.npz'],
    'path_to_config': ['/config/config_example_data.yml'],
    'path_to_output': ['/output/']
} 

# save the yaml fit list
save_path = Path.cwd() / 'data/F1901_Crumble_Squid' / 'fit_A12_data.yml'

with open(save_path, 'w') as outfile:
    yaml.dump(fit_dict, outfile, default_flow_style=False)