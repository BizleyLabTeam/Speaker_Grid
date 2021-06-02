"""

# The purpose of this notebook is to get the head position and direction of the ferret for each stimulus presentation.
 
Input data comes from three sources:
(1) Stimulus metadata after temporal alignment to both video and neural recording systems
(2) LED tracking data
(3) Speaker positions in grid

See also:
    notebooks/combine_head_n_stimulus_loc.ipynb

Created:
    2021-06-01 by Stephen Town

"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from pathlib import Path

speaker_layout = '/home/stephen/Github/Speaker_Grid/metadata/Speaker_grid_layout_2021_05_21.csv'

# Load stimulus information and tracked LED positions
data_dir = Path('/home/stephen/Github/Speaker_Grid/data')
# data_dir = Path('/home/stephen/Github/Speaker_Grid/data/F1901_Crumble_Squid/2021-05-27_Squid_15-57')

stim_file = '2021-05-27T17-57-07_StimData_MCSVidAlign.csv'
tracking_file = '2021-05-27_CorrectedVid_17_56_08.csv'



def load_speaker_positions(file_path):
    
    spk = pd.read_csv(speaker_layout)
    spk = spk[['matlab_chan','2021_05_31_PixLoc_Corrected_x','2021_05_31_PixLoc_Corrected_y']]

    spk.dropna(inplace=True)
    spk.rename(columns={'matlab_chan':'Speaker','2021_05_31_PixLoc_Corrected_x':'speak_xpix','2021_05_31_PixLoc_Corrected_y':'speak_ypix'}, inplace=True)

    return spk

def wrap_to_pi(x):
    if x < -np.pi:
        x += (np.pi * 2)
    elif x > np.pi:
        x += -(np.pi * 2)
    return x

def merge_stim_and_tracking_data(stim, LEDs):
    
    # Merge on frame number
    join_data = pd.merge( stim, LEDs[['frame','blue_x','red_x','blue_y','red_y']], left_on='closest_frame', right_on='frame')

    # Remove tracking results that are outside area of interest (grid centre where we tested)
    join_data.loc[join_data['blue_x'] < 150, 'blue_x'] = np.nan
    join_data.loc[join_data['blue_x'] > 500, 'blue_x'] = np.nan
    join_data.loc[join_data['blue_y'] < 0, 'blue_y'] = np.nan
    join_data.loc[join_data['blue_y'] > 480, 'blue_y'] = np.nan

    join_data.loc[join_data['red_x'] < 150, 'red_x'] = np.nan
    join_data.loc[join_data['red_x'] > 500, 'red_x'] = np.nan
    join_data.loc[join_data['red_y'] < 0, 'red_x'] = np.nan
    join_data.loc[join_data['red_y'] > 480, 'red_x'] = np.nan

    # Calculate head position as center between red and blue LEDs
    join_data['head_x'] = (join_data['blue_x'] + join_data['red_x']) / 2
    join_data['head_y'] = (join_data['blue_y'] + join_data['red_y']) / 2

    # Compute head angle in the arena
    join_data['blue_zero_x'] = join_data['blue_x'] - join_data['head_x']
    join_data['blue_zero_y'] = join_data['blue_y'] - join_data['head_y']

    offset = -(np.pi/2)             # rotate cw by 90 degrees to get midline pointing forwards (anterior) on the head
    join_data['head_angle'] = np.arctan2(join_data['blue_zero_y'], join_data['blue_zero_x']) + offset
    join_data['head_angle'] = join_data['head_angle'].apply(wrap_to_pi)

    join_data.dropna(inplace=True)
    join_data.reset_index(inplace=True)

    return join_data


######################################################################################
def merge_headpose_and_speakerloc(join_data, spk):

    hs_data =  pd.merge(join_data, spk, on='Speaker')

    # Now we get the vector from head to stimulus (h2s = "head to stimulus" or "head to speaker")
    hs_data['h2s_x'] = hs_data['speak_xpix'] - hs_data['head_x']
    hs_data['h2s_y'] = hs_data['speak_ypix'] - hs_data['head_y']

    hs_data['h2s_distance'] = np.sqrt((hs_data['h2s_y'] ** 2) +  (hs_data['h2s_x']**2))

    hs_data['h2s_theta'] = np.arctan2(hs_data['h2s_y'], hs_data['h2s_x']) - hs_data['head_angle']
    hs_data['h2s_theta'] = hs_data['h2s_theta'].apply(wrap_to_pi)

    hs_data['h2s_x'] = hs_data['h2s_distance'] * np.cos(hs_data['h2s_theta'])  
    hs_data['h2s_y'] = hs_data['h2s_distance'] * np.sin(hs_data['h2s_theta'])  
    
    return hs_data

def find_tracking_file(file_path):

    return next( file_path.glob('*CorrectedVid*.csv'))
    

def main():

    # For each recording session
    for stim_file in data_dir.rglob('*StimData_MCSVidAlign*'):        
    
        LED_file = find_tracking_file(stim_file.parent)
        
        # Get head pose at each stimulus presentation
        stim = pd.read_csv( str(stim_file))
        LEDs = pd.read_csv( str(LED_file))

        join_data = merge_stim_and_tracking_data(stim, LEDs)

        # Get stimulus position relative to head
        spk = load_speaker_positions(speaker_layout)

        hs_data = merge_headpose_and_speakerloc(join_data, spk)

        # Save the data for use with neural analysis
        save_file = str(stim_file).replace('StimData_MCSVidAlign', 'Stim_LED_MCS')
        hs_data.to_csv( str(data_dir / save_file))



if __name__ == "__main__":
    main()