""" Make sure there's actually some sound-driven activity to model """



from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

fontsize=6
plt.rc('font', size=fontsize) #controls default text size
plt.rc('axes', titlesize=fontsize+1) #fontsize of the title
plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fontsize) #fontsize of the y tick labels
plt.rc('legend', fontsize=fontsize) #fontsize of the legend



def get_quick_PSTH(counts, ev_idx, n_bins:int=100):
    """ Quick and dirty sampling of spike counts around time of events
    
    Forced time window: 50% before and 50% after event

    Returns:
        Mean spike count across events
    """

    offset = int(n_bins / 2)
    psth = np.zeros((len(ev_idx), n_bins))

    # For each stimulus, sample counts
    for i, ev in enumerate(ev_idx):
        psth[i,:] = counts[ev-offset:ev+offset]

    return psth.mean(axis=0)


def plot_all_PSTHs(events, counts, titles):

    # Create canvas
    fig, axs = plt.subplots(nrows=8, ncols=8, sharex=True)
    axs = np.ravel(axs)

    # Get samples on which an event was played
    ev_idx = np.where(events == 1)[0]

    # For each neuron (column of counts)
    for col, neu_name in enumerate(titles):
        
        psth = get_quick_PSTH(counts[:,col], ev_idx)
        
        axs[col].plot(psth, lw=0.7)
        axs[col].set_title(neu_name, pad=0)

    # plt.tight_layout()
    plt.show()
    
    


def main():

    # Settings
    ferret = 'F1901_Crumble'

    # Paths
    data_path = Path.cwd() / 'data' / f"{ferret}_Squid"

    all_chan_file = data_path / 'example_pgam_concatn.npz'

    data = np.load(all_chan_file, allow_pickle=True)

    # Extract events from variables array in saved file
    variable_names = data['variable_names']
    variables = data['variables']
    neu_names = data['neu_names']
    events = np.squeeze(variables[:, np.array(variable_names) == 'events'])

    # Show spike counts around event times (predict events should drive change in counts)
    plot_all_PSTHs(events, data['counts'], neu_names)




if __name__ == '__main__':
    main()
