""" Examines whether there is any spatial tuning in the response
so that we can figure out if the model has something to capture

 """

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


def get_spike_totals(counts, ev_idx, n_bins:int=10):
    """ Quick and dirty sampling of spike counts after clicks
    

    Returns:
        Spike count for each event
    """

    totals = np.zeros((len(ev_idx), n_bins))

    # For each stimulus, sample counts
    for i, ev in enumerate(ev_idx):
        totals[i,:] = counts[ev:ev+n_bins]

    return totals.sum(axis=1)


def plot_SRF(angles, totals, ax) -> None:

    angle_bins = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    bin_width = angle_bins[1] - angle_bins[0]
    
    n_bins = len(angle_bins)
    srf = np.zeros(n_bins)
    err = np.zeros(n_bins)

    # For each angle bin
    for i, abin in enumerate(angle_bins):
        srf[i] = np.mean( totals[(angles >= abin) & (angles < abin+bin_width)])
        err[i] = np.std( totals[(angles >= abin) & (angles < abin+bin_width)])

    # Plot 
    x = angle_bins + (bin_width/2)
    ax.plot(x, srf, lw=0.7)

    ax.set_ylim((0, srf.max()*1.5))
    ax.tick_params(pad=0)





def plot_all_SRFs(events, angles, counts, titles):

    # Create canvas
    fig, axs = plt.subplots(nrows=8, ncols=8, sharex=True)
    axs = np.ravel(axs)

    # Get samples on which an event was played
    ev_idx = np.where(events == 1)[0]
    angles = angles[~np.isnan(angles)]

    # For each neuron (column of counts)
    for col, neu_name in enumerate(titles):
        
        spike_totals = get_spike_totals(counts[:,col], ev_idx)
        
        plot_SRF(angles, spike_totals, axs[col])
        axs[col].set_title(neu_name, pad=0)


def preprocess(data):


    # Extract events from variables array in saved file
    variable_names = data['variable_names']
    variables = data['variables']

    events = np.squeeze(variables[:, np.array(variable_names) == 'events'])
    angles = np.squeeze(variables[:, np.array(variable_names) == 'h2s_theta'])

    return events, angles


def plot_spatial_tuning_for_all_chans(file_path):

    data = np.load(file_path, allow_pickle=True)
    neu_names = data['neu_names']

    events, angles = preprocess(data)
    plot_all_SRFs(events, angles, data['counts'], neu_names)

    plt.savefig(file_path.parent / f'{file_path.stem}_pgam_SRF_check.png')
    plt.close()


def main():
    pass

    # Settings
    ferret = 'F1901_Crumble'

    # Paths
    data_path = Path.cwd() / 'data' / f"{ferret}_Squid"

    # Plot across sessions
    all_chan_file = data_path / 'example_pgam_concatn.npz'
    plot_spatial_tuning_for_all_chans(all_chan_file)

    # Plot for each session
    for session in data_path.glob('*Squid*'):

        session_file = session / 'example_pgam_data.npz'

        plot_spatial_tuning_for_all_chans(session_file)


if __name__ == '__main__':
    main()