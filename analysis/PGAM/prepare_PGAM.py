""" Module used to prepare speaker grid data for PGAM 

Things we need:
    1. Binned spike counts (use 6 ms bins for direct comparison with Eduardo's work)
    2. Temporal input
        - an event series indicating the times of click presentation
    3. Spatial inputs
        - continuous series indicating the x and y positions of the head
        - circular series indicating head angle
        - sound location


Questions:
    Should we normalize pixel values? (probably)
    How do we create the correct not configuration for circular data?
    How do we add in activity of other neurons?
    Should we filter for likelihood of dlc results (haven't done this so far as unsure how to deal with missing data)

"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import statsmodels.api as sm
import yaml

sys.path.insert(0, str(Path.cwd()))
from lib import utils


#######################################
# Spike handling

def load_spike_times(file_path:Path) -> dict:
    """ Load spike times from text files and return as dictionary keyed by channel number"""

    output = dict()

    for chan_file in file_path.glob('*.txt'):

        spike_times = np.genfromtxt(chan_file, delimiter=',')

        # Tag the hemisphere from which this channel was recorded
        if 'SpikeTimes_2_C' in chan_file.stem:
            prefix = 'B'
        elif 'SpikeTimes_C' in chan_file.stem:
            prefix = 'A'
        else:
            prefix = 'Unknown'

        chan_name = f"{prefix}{chan_file.stem[-2:]}"

        print(f"Loaded spikes for {chan_name}")
        output[chan_name] = spike_times
    
    return output


def count_spikes(file_path:Path, bin_width:float) -> np.array:
    """
    counts: numpy.array of dim (T x N), where T is the total number 
    of time points, N is the number of neurons

    Notes:
        Input data: Spike times for each neuron (channel in this case as they're 
        not sorted) are stored as text files

        Output: columns of array are sorted by channel number
    """

    # Load spikes from text files
    spike_times = load_spike_times(file_path)
    n_neurons = len(spike_times)

    # Determine when the last spike in the session was, and thus how 
    # many time points we need
    max_time = max([max(v) for v in spike_times.values()])
    n_timepoints = int(np.ceil(max_time / bin_width))

    # Preassign spike counts
    spike_counts = np.zeros((n_timepoints-1, n_neurons))

    # Count spikes
    bin_edges = np.linspace(0.0, bin_width*n_timepoints, num=n_timepoints)
    chan_offsets = {'A':0, 'B':32}
    neu_names = ['' for i in range(len(spike_times))]
    
    for chan_str, times in spike_times.items():
        
        chan_num = int(chan_str[1:]) - 1            # change from one-based to zero-based
        chan_num += chan_offsets[chan_str[0]]       # map channels from left (A) and right (B) headstages to different number ranges
        
        spike_counts[:,chan_num], _ = np.histogram(times, bins=bin_edges)
        neu_names[chan_num] += chan_str

    return spike_counts, neu_names


####################################
# Stimulus handling

def get_trial_ids(stim:pd.DataFrame, bin_width:float, n_timepoints:int) -> np.array:
    """ Create array indicating trial number
     (This isn't strictly relevant to this project, but we'll use the block to make it work)

    Args:
        stim: dataframe containing stimulus data with 'Block' column indicating a grouping in time
        bin_width: sample interval
        n_timepoints: sample number

    Returns:
        trial_ids: np array with length n_timepoints, should look like a staircase when plotted 

    >>> trial_ids = get_trial_ids(stim, bin_width, n_timepoints)
    """

    # Get time of last stimulus within block, to use as end time
    blocks = (
        stim
        .groupby('Block')
        .max()
        .rename({'MCS_Time':'end'}, axis=1)
        .assign(end_idx = lambda x: np.floor(x.end/bin_width))
    )

    # Get end time of last block, to use as start time for next block (start at zero)
    blocks['start_idx'] = blocks['end_idx'].shift(1, fill_value=0)

    # Create an array filled with block numbers, based on block start and end times
    trial_ids = np.zeros(n_timepoints)              
    for block_num, b in blocks.iterrows():
        trial_ids[b.start_idx.astype(int) : b.end_idx.astype(int)] = block_num

    # Fill any timepoints after final end time with the last block
    trial_ids[b.end_idx.astype(int):] = block_num

    return trial_ids


def process_stimuli(stim:pd.DataFrame, bin_width:float, n_timepoints:int) -> np.array:
    """ Return binary signal indicating presence of stimulus
    
    TO DO: Expand to consider sound location relative to head / the world
     """

    # Create binary array indicating presence of sound
    ev_idx = np.floor( stim['MCS_Time'].to_numpy() / bin_width)
    ev_idx = ev_idx.astype(int)

    bin_ev = np.zeros(n_timepoints)
    bin_ev[ev_idx] = 1

    # Get sound angle w.r.t. head for each event
    h2s_theta = np.full_like(bin_ev, np.nan)
    for idx, val in zip(ev_idx, stim['h2s_theta'].to_numpy()):
        h2s_theta[idx] = val
    
    # Package event info into one dictionary
    return dict(
        binary = bin_ev,
        h2s_theta = h2s_theta
    )


#####################################
# Head Tracking 
def process_headtrack(session:Path, sync_mdl, bin_width, n_timepoints):
    """ 
    
    To do:
        Implement optional filter
     """

    # Load deeplabcut results
    track_file = next( session.rglob('*DLC*.csv'))
    df = utils.read_deeplabcut_csv(session / 'LEDs', track_file.name)

    # Remove values with low likelihoods
    likelihood_threshold = 0.6
    idx = df[df['blue_LEDlikelihood'] < likelihood_threshold].index

    df.loc[idx, 'blue_LEDx'] = np.nan
    df.loc[idx, 'blue_LEDy'] = np.nan

    # Interpolate missing values
    df['blue_LEDx'].interpolate(method='nearest', inplace=True)
    df['blue_LEDy'].interpolate(method='nearest', inplace=True)

    # # Mask bins where the animal is tracked while being placed in the box
    # idx = df[df['blue_LEDx'] > 440].index

    # df.loc[idx, 'blue_LEDx'] = np.nan
    # df.loc[idx, 'blue_LEDy'] = np.nan

    # Predict MCS times from frame numbers
    Xnew = df.index.to_numpy()
    Xnew = sm.add_constant(Xnew)
    df['MCS_time'] = sync_mdl.predict(Xnew)

    # Get time series for which spikes were counted
    bin_edges = np.linspace(0.0, bin_width*n_timepoints, num=n_timepoints)
    bin_centers = bin_edges + (bin_width/2)

    # Interpolate Blue LED position (use blue as proxy for head)
    blue_x = np.interp(x=bin_centers, xp=df['MCS_time'].to_numpy(), fp=df['blue_LEDx'].to_numpy())
    blue_y = np.interp(x=bin_centers, xp=df['MCS_time'].to_numpy(), fp=df['blue_LEDy'].to_numpy())

    # Constrain values to image limits (we don't yet know how to manage missing data)
    blue_x = np.clip(blue_x, 0, 440)
    blue_y = np.clip(blue_y, 0, 480)

    return blue_x, blue_y


def build_sync_model(X, Y):
    """ Build a simple linear model of form y = ax + b for estimating
    MCS time from video time """
    
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    return model.fit()
    

######################################
### Session handling
def crop_data_to_around_events(events, bin_width, variables, trial_ids, spike_counts):
    """ Crop data to within the first and last event 
     (avoids a lot of the noise at the start and end of the experiment, as 
     well as any lack of overlap between tracking data and MCS recording)
    """

    # Flag event samples
    idx = np.where(events == 1)[0]

    # Include a buffer of 1.2 seconds (for convolution with kernel)
    buffer_samps = int(np.round(1.2/bin_width))     

    # Make sure start and end indices fall within data
    start_idx = max([idx.min()-buffer_samps, 0])
    end_idx = min([idx.max()+buffer_samps, spike_counts.shape[0]])

    # Filter
    variables = variables[start_idx:end_idx,:]
    trial_ids = trial_ids[start_idx:end_idx]
    spike_counts = spike_counts[start_idx:end_idx,:]

    return variables, trial_ids, spike_counts


def process_session(session:Path, bin_width):
    """ Apply all preprocessing steps to data from a single recording session """
    
    # Load stimulus times according to different clocks 
    # (i.e. the timeline compatible with spike times)
    stim_file = next(session.glob('*Stim_LED_MCS.csv'))
    stim = pd.read_csv(stim_file, usecols=['MCS_Time','closest_frame','Block','h2s_theta'])    

    # Build linear regression model that takes frame number as input and returns
    # estimated time on MCS clock (for which spike times are referenced)
    sync_mdl = build_sync_model(X=stim['closest_frame'], Y=stim['MCS_Time'])

    # Load spikes (this determines the max time of the data)
    spike_counts, neu_names = count_spikes(session / 'spike_times', bin_width)

    neu_info = {k: {'unit_type':'MUA', 'area':'AC'} for k in neu_names}

    # Build variables from input features
    #   numpy.array of dim (T x M), T as above, M the number of task variables
    n_timepoints = spike_counts.shape[0]
    
    blue_x, blue_y = process_headtrack(session, sync_mdl, bin_width, n_timepoints)

    events = process_stimuli(stim, bin_width, n_timepoints)

    variable_names = ['Blue_X', 'Blue_Y', 'events']
    variables = np.zeros((n_timepoints, len(variable_names)))

    variables[:, 0] = blue_x
    variables[:, 1] = blue_y
    variables[:, 2] = events['binary']
    # variables[:, 3] = events['h2s_theta']

    trial_ids = get_trial_ids(stim, bin_width, n_timepoints)

    # Crop data to times around events (as this was when the tracking should have worked - no guarentees before or after)
    variables, trial_ids, spike_counts = crop_data_to_around_events(events['binary'], bin_width,
        variables, trial_ids, spike_counts)
    
    # Save for processing in Docker
    output_file = session / 'example_pgam_data.npz'
    np.savez(
        output_file, 
        counts = spike_counts,
        variables = variables,
        variable_names = variable_names,
        neu_names = neu_names,
        neu_info = neu_info, 
        trial_ids = trial_ids)

    return output_file
    

    
################################################
# Managing multiple sessions

def concat_sessions(data_dir, npz_files):
    """ Bring preprocessed data from multiple sessions together """

    counts, variables, trial_ids = [], [], []
    max_trial = 0

    # For each npz file (associated with one session)
    for file_idx, npz_f in enumerate(npz_files):

        data = np.load(npz_f, allow_pickle=True)

        # Build lists for concatenration
        counts.append(data['counts'])
        variables.append(data['variables'])

        trial_ids.append(data['trial_ids'] + max_trial)   # Avoid duplicating trial ids across sessions
        max_trial += data['trial_ids'].max()

        # Initialize names on first pass
        if file_idx == 0:
            variable_names = data['variable_names']
            neu_names = data['neu_names']
            neu_info = data['neu_info'].all()
        
        # Check for consistency on later passes
        else:
            assert all(variable_names == data['variable_names'])
            assert all(neu_names == data['neu_names'])
            assert neu_info == data['neu_info'].all()

    # Concatenate lists
    counts = np.concatenate(counts)
    variables = np.concatenate(variables)
    trial_ids = np.concatenate(trial_ids)

    # Save for processing in Docker
    output_file = data_dir / 'example_pgam_concatn.npz'
    np.savez(
        output_file, 
        counts = counts,
        variables = variables,
        variable_names = variable_names,
        neu_names = neu_names,
        neu_info = neu_info, 
        trial_ids = trial_ids)


###############################
# configuration
def prepare_configuration(order:int, bin_width:float):
    """ 
    
    Args:
        Order: number of coefficient of the polynomials that make up splines
        bin_width: sample period in seconds
    
     """

    x_knots = np.hstack(([160]*(order-1), np.linspace(160,440,15),[440]*(order-1)))
    x_knots = [float(k) for k in x_knots]

    y_knots = np.hstack(([0]*(order-1), np.linspace(0,480,15),[480]*(order-1)))
    y_knots = [float(k) for k in y_knots]

    return {
        'Blue_X' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': x_knots,
            'order': order,
            'is_temporal_kernel': False,
            'is_cyclic': [False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':bin_width 
        },
        'Blue_Y' : {
            'lam':10, 
            'penalty_type': 'der', 
            'der': 2, 
            'knots': y_knots,
            'order':order,
            'is_temporal_kernel': False,
            'is_cyclic': [False],
            'knots_num': np.nan,
            'kernel_length': np.nan,
            'kernel_direction': np.nan,
            'samp_period':bin_width 
        },
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
        'neuron_A':
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

###################
def main():

    # Settings
    ferret = 'F1901_Crumble'
    bin_width = 0.006           # Seconds

    # Paths
    data_path = Path.cwd() / 'data' / f"{ferret}_Squid"

    # Get data for each session tested
    npz_files = []

    for session in data_path.glob('*Squid*'):
        
        print(session)

        npz_files.append( 
            process_session(session, bin_width)
        )

    # Bring data together from multiple sessions
    concat_sessions(data_path, npz_files)

    # Prepare configuration (I have no idea what the hell I'm doing)
    order = 4   # the order of the spline is the number of coefficient of the polynomials

    cov_dict = prepare_configuration(order, bin_width)
    config_path = data_path / 'config_example_data.yml'

    with open(config_path, 'w') as outfile:
        yaml.dump(cov_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    main()