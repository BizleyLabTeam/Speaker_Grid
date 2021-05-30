"""

Plotting tools for drawing neural activity in the squid

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

repo_dir = Path('/home/stephen/Github/Speaker_Grid/data/')
ferret = 'F1901_Crumble_Squid'
rec_dir = '2021-05-27_Squid_15-57'

stim_file = '2021-05-27T17-57-07_StimulusData.csv'





def plot_raster(stim_data, spike_times):

    # Count spikes for each trial
    r_time, r_trial, r_color =  [], [], []
    trial_count = 0

    for _, pulse in stim_data.iterrows():

        trial_ev = spike_times[(spike_times > pulse['window_start']) & (spike_times < pulse['window_end'])]
        trial_ev += -pulse['MCS_Time']      # Reference stimulus time, not universal time

        trial_count += 1

        r_time.append(trial_ev)
        r_trial.append(np.full_like(trial_ev, trial_count))
        r_color.append(np.full_like(trial_ev, pulse['Speaker']))

    r_time = np.concatenate(r_time)
    r_trial = np.concatenate(r_trial)
    r_color = np.concatenate(r_color)

    # Bring spike counts together in dataframe
    raster_df = pd.DataFrame( r_time[:, np.newaxis], columns=['SpikeTime'])
    raster_df['Trial_Idx'] = r_trial[:, np.newaxis]
    raster_df['Speaker'] = r_color[:, np.newaxis]

    raster_df = raster_df.sort_values(by=['Speaker', 'Trial_Idx'])
    raster_df.reset_index(inplace=True, drop=True)  # Drop the original index 
    raster_df.reset_index(inplace=True)             # Ensure new index is available for plotting

    for idx, row in raster_df.iterrows():
        if idx == 0:
            new_trial = 1
        else:
            if row['Trial_Idx'] != previous_trial:
                new_trial += 1

        previous_trial = row['Trial_Idx']
        raster_df['Trial_Idx'].iloc[idx] = new_trial


    # Plot data
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.scatter( data=raster_df, x='SpikeTime', y='Trial_Idx', c='Speaker', s=0.01)
    ax.set_xlabel('Time post-stimulus (s)')
    ax.set_ylabel('Trial')
    ax.set_xlim((-0.1, 0.4))
    ax.set_ylim((0, new_trial))

    plt.show()


def main():

    # Load data
    stim_path = repo_dir / ferret / rec_dir / stim_file
    stim_data = pd.read_csv(stim_path)

    stim_data['window_start'] = stim_data['MCS_Time'] - 0.1
    stim_data['window_end'] = stim_data['MCS_Time'] + 0.4

    spike_path = repo_dir / ferret / rec_dir
    for spike_file in spike_path.glob('*SpikeTimes_2*.txt'):

        spike_times = np.loadtxt( spike_file, delimiter=',')

        plot_raster(stim_data, spike_times)


if __name__ == "__main__":
    main()


