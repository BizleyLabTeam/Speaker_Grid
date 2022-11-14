""" Plot head positions to see what range of pixel values we should specify knots for"""



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


def plot_positions(x, y):

    # Create canvas
    fig, ax = plt.subplots(nrows=1, ncols=1)
        
    ax.plot(psth, lw=0.7)
    ax.set_title(neu_name, pad=0)

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
    
    blue_x = np.squeeze(variables[:, np.array(variable_names) == 'Blue_X'])
    blue_y = np.squeeze(variables[:, np.array(variable_names) == 'Blue_Y'])

    # Show spike counts around event times (predict events should drive change in counts)
    plt.scatter(x=blue_x, y=blue_y, s=0.1)
    plt.show()




if __name__ == '__main__':
    main()
