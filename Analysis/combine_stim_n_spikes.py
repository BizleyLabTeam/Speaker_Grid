"""

The channel mapping is messed up, so let's keep everything in MCS channel numbers

Stephen Town - 2021-06-02
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import squidplot as sqp 

plt.style.use('seaborn')






def plot_array_psth():
    
    data_dir = Path('/home/stephen/Github/Speaker_Grid/data')    

    # For each recording session
    for stim_file in data_dir.rglob('*Stim_LED_MCS.csv'):        
            
        stim = pd.read_csv( stim_file)

        fig, axs = plt.subplots(nrows=8, ncols=4, sharex=True, figsize=(8,14))
        axs = np.ravel(axs)        
        psth_df = []

        spike_dir = stim_file.parent / 'spike_times'

        # Supposed left hand array
        for spike_file in spike_dir.glob('*SpikeTimes_C*.txt'):            

            mcs_chan = int(spike_file.stem[-2:]) 

            spikes = np.loadtxt( spike_file, delimiter=',')

            tvec, frate = sqp.plot_psth(stim, spikes, axs[mcs_chan-1])
            axs[mcs_chan-1].set_title(str(mcs_chan), fontsize=8)

            psth_df.append({
                'Chan': f"A{mcs_chan:02d}",
                'MeanRate': frate
            })            

        save_path = Path(stim_file.parent) / (stim_file.stem + '_LHS.png')
        plt.savefig( str(save_path))

        # Supposed right hand array
        fig, axs = plt.subplots(nrows=8, ncols=4, sharex=True, figsize=(8,14))
        axs = np.ravel(axs)   
        
        for spike_file in spike_dir.glob('*SpikeTimes_2*.txt'):            

            mcs_chan = int(spike_file.stem[-2:]) 

            spikes = np.loadtxt( spike_file, delimiter=',')

            tvec, frate = sqp.plot_psth(stim, spikes, axs[mcs_chan-1])
            axs[mcs_chan-1].set_title(str(mcs_chan), fontsize=8)

            psth_df.append({
                'Chan': f"B{mcs_chan:02d}",
                'MeanRate': frate
            })

        save_path = Path(stim_file.parent) / (stim_file.stem + '_RHS.png')
        plt.savefig( str(save_path))

        csv_out = str(stim_file).replace('Stim_LED_MCS','PSTH')
        psth_df = pd.DataFrame(psth_df)
        psth_df.to_csv(csv_out, index=False)



def main():

    plot_array_psth()
    return

    data_dir = Path('/home/stephen/Github/Speaker_Grid/data/F1810_Ursula_Squid')    

    # For each recording session
    for stim_file in data_dir.rglob('*Stim_LED_MCS.csv'):        
            
        stim = pd.read_csv( stim_file)

        spike_dir = stim_file.parent / 'spike_times'

        # Supposed left hand array
        for spike_file in spike_dir.glob('*SpikeTimes*.txt'):            
            
            if 'SpikeTimes_2' in spike_file.stem:
                mcs_chan = 'B' + spike_file.stem[-2:]
            else:
                mcs_chan = 'A' + spike_file.stem[-2:]

            spikes = np.loadtxt( spike_file, delimiter=',')

            def get_firing_rate_for_stim(x):
                n, _ = np.histogram(spikes-x, [0, 0.05])
                return n[0]

            stim[mcs_chan] = stim['MCS_Time'].apply(get_firing_rate_for_stim)
                        
        save_name = stim_file.name.replace('Stim_LED_MCS', 'StimSpikeCounts')
        save_path = Path(stim_file.parent) / save_name
        stim.to_csv( str(save_path), index=False)

if __name__ == '__main__':
    main()