"""
PAGE 1
Test dashboard for visualizing and exploring neural activity

"""

import os, sys

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from skimage import io

from dash.dependencies import Output, Input
from pathlib import Path

from app import app



def get_file_dict(file_path, file_type):
    """
    Get files with specific extension in list of dictionaries
    compatible with dcc.Dropdown options
    
    Parameters:
    ----------
    file_path : str
        Parent directory containing files of interest
    file_type : str
        File extension with wildcard prefix

    >>> get_file_dict('/Task_Switching/ERPs', '*.h5')    

    Returns:
    --------
    file_dict : list
        List of dictionaries with 'label' and 'value' keys
        Note that values are posix paths to file locations
    """
    
    file_dict = []
    for file in Path(file_path).glob(file_type):
        file_dict.append({
            'label': file.stem,
            'value': str(file)
        })

    return file_dict


def get_angular_histogram( vals, n_bins=12):
        
        interval = (2 * np.pi) / n_bins
        theta_edges = np.linspace(-np.pi, np.pi, n_bins+1)
        theta_centers = theta_edges[0:-1] + (interval / 2)
        rho = np.zeros_like(theta_centers)

        for idx, bin_start in enumerate(theta_edges[0:-1]):

            rho[idx] = sum((vals >= bin_start) & (vals < (bin_start+interval)))

        return theta_centers, rho


def bin_by_angle(df, tStr, yStr, n_bins=12):

    interval = (2 * np.pi) / n_bins
    theta_edges = np.linspace(-np.pi, np.pi, n_bins+1)
    theta_centers = theta_edges[0:-1] + (interval / 2)
    rho = np.zeros_like(theta_centers)

    srf = []
    for theta in theta_edges[0:-1]:

        bin_data = df[(df[tStr] >= theta) & (df[tStr] < theta+interval)]

        srf.append({
            'theta': theta,
            'theta_d': (theta / np.pi) * 180,
            'mean_rate': bin_data[yStr].mean()
        })

        if abs(theta) == np.pi:         # Add wrap around circle
            srf.append({
                'theta': -theta,
                'theta_d': (-theta / np.pi) * 180,
                'mean_rate': bin_data[yStr].mean()
            })  
    
    srf = pd.DataFrame(srf)
    srf.sort_values(by='theta', inplace=True)

    return srf

def plot_psth(stim, spikes, t_start=-0.1, t_end=0.41, bin_width=0.01):
    """
    Plots spike rate vs time across all stimuli
    
    Parameters:
    ----------
    stim : pandas dataframe
        Contains stimulus times in MCS_Time column (not very user friendly!)
    spikes : numpy array
        Array of spike times 
    t_start : float
        Start time for plotting the psth
    t_end : float
        End time for plotting the psth (remember that the last bin value won't be included)   
    bin_width : float
        Duration of each psth bin in which spikes are counted
    ax : matplotlib axes
        Axes to plot PSTH if already defined
    
    Returns:
    --------
    ax : matplotlib axes
        Axes containing line plot of peri-stimulus time histogram
    """
    
    bin_edges = np.arange(t_start, t_end, bin_width)
    spike_count = np.zeros((stim.shape[0], len(bin_edges)-1), dtype=np.int64)

    for idx, row  in stim.iterrows():

        t_delta = spikes - row['MCS_Time']

        spike_count[idx], _ = np.histogram(t_delta, bins=bin_edges)

    spike_rate = spike_count / bin_width
    mean_sr = np.mean(spike_rate, axis=0)
    bin_centers = bin_edges[0:-1] + (bin_width/2)

    return px.line(x=bin_centers, y=mean_sr, title='Firing Rate vs. Time')



# List ferrets in project
ferret_dict = get_file_dict("data/", "F*")

# List channels available (from a set list)
chan_list = []
for headstage in ['A','B']:
    for chan in range(1,33):
        cStr = f"{headstage}{chan:02d}"
        chan_list.append({'label':cStr, "value":cStr})


layout = html.Div(
        children=[ 
            # Row 2 - Header
            html.Div(
                children=[
                    html.H1(children="Spatial Receptive Fields", className="header-title"),
                    html.P(
                        children="Pilot data from testing in the speaker grid.",                        
                        className="header-description",
                    ),
                ],
                className="header",
            ), 
            # Row 3 - Menu title (Ferret / Session dropdowns)
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(children="Ferret:", className="menu-title"),
                            dcc.Dropdown(
                                id="ferret-filter",
                                options=ferret_dict,
                                value=ferret_dict[-1]['value'],
                                clearable=False,
                                className="dropdown",
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Div(children="Site:", className="menu-title"),
                            dcc.Dropdown(
                                id="session-filter",
                                clearable=False,
                                className="dropdown",
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Div(children="Channel", className="menu-title"),
                            dcc.Dropdown(
                                id="chan-filter",
                                options=chan_list,
                                value=chan_list[16]['value'],
                                className="dropdown",
                            ),
                        ],
                    ),                    
                ],
                className="menu",
            ),       
            # Row 4 - Grid image and polar plots
            html.Div(
                children=[         
                    dbc.Row([
                        dbc.Col([
                            dcc.RadioItems(
                                    options=[
                                        {'label': 'Show calibrated image', 'value': 'Calibrated'},
                                        {'label': 'Show original image', 'value': 'Original'},                                        
                                    ],
                                    value='Calibrated',
                                    id="calibrated_im"
                                ),
                            dcc.Checklist(
                                    options=[
                                        {'label': 'Speakers', 'value': 'Speakers'},
                                        {'label': 'Head Position', 'value': 'Head'},                                        
                                    ],
                                    value=['Speakers','Head'],
                                    id="scatter_check"
                                ),
                            dbc.Row([
                                dbc.Col([
                                    html.P(children="Speaker color:",)
                                ],
                                width=3),
                                dbc.Col([
                                    dcc.RadioItems(
                                        options=[
                                            {'label': 'By speaker', 'value': 'BySpeaker'},
                                            {'label': 'By firing rate', 'value': 'ByRate'},                                        
                                        ],
                                        value='ByRate',
                                        id="speaker_color"
                                    ),
                                ]),
                            ]),
                            dcc.Graph(
                                id="grid_image",
                                config={"displayModeBar": False}
                                )   
                        ]),
                        dbc.Col([                            
                            dcc.Graph(
                                id="head_direction_hist",
                                config={"displayModeBar": False}
                                ),
                            dcc.Graph(
                                id="stim_head_xy",
                                config={"displayModeBar": False}
                            )                               
                        ])                    
                    ])                                                
                ], className="wrapper",
            ), 
            # Row 5 - Range slider test
            html.Div(
                children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Markdown("Select Distance Range (px):")
                        ],
                        width=2.5,
                        ),
                        dbc.Col([
                            dcc.RangeSlider(
                                id='distance_range',
                                min=0,
                                max=600,
                                step=1,
                                marks={0:'0', 200:'200', 400:'400', 600:'600'},
                                value=[0, 600]
                            )
                        ],
                        width=3,
                        ),
                    ],
                    no_gutters=True,
                    justify="center")                    
                ],
                className='wrapper'
            ),
            # Row 6 PSTHs and SRFs
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='PSTH'),
                    ]),
                    dbc.Col([
                        dcc.Graph(id='SRF'),
                    ])
                ]),
            ], 
            className='wrapper')
    ]
)


@app.callback(
    [
        Output("session-filter", "options"),
        Output("session-filter", "value"),
        ],
    [Input('ferret-filter', 'value')]
    )
def update_session_list(value):
    
    session_dict = get_file_dict(value, '*Squid*')
    
    return [session_dict, session_dict[-1]['value']]


@app.callback(
    [
        Output("grid_image", "figure"),
        Output("head_direction_hist", "figure"),
        Output("stim_head_xy", "figure"),
        Output("PSTH", "figure"),
        Output("SRF", "figure")
        ],
    [
        Input('session-filter', 'value'),
        Input('scatter_check', 'value'),
        Input('chan-filter', 'value'),
        Input('calibrated_im', 'value'),
        Input('speaker_color', 'value'),
        Input('distance_range', 'value')
        ]
    )
def plot_behavioral_data(value, scatter_opts, chan, calib_opt, speakerC, distRange):
    
    # Find a sample image for background
    if calib_opt == 'Calibrated':
        img = io.imread('calibrations/2021-05-31_SpeakerLocations_Corrected.tif')
    elif calib_opt == 'Original':
        img = io.imread('calibrations/2021_05_31_SpeakerLocations_Squid.tif')
    fig = px.imshow(img)
    
    # Load behavioral data
    stim_file = next(Path(value).glob('*StimSpikeCounts.csv'))
    stim = pd.read_csv(str(stim_file))
    
    stim[chan] = stim[chan] / 0.05  # Count to rate conversion

    # Filter behavioral data for distance
    stim = stim[(stim['h2s_distance'] >= distRange[0]) & (stim['h2s_distance'] < distRange[1])]

    # Extract speaker locations
    speakers = []
    for spk, s_data in stim.groupby(by='Speaker'):            
        speakers.append({
            'id':spk,
            'x': s_data['speak_xpix'].unique()[0],
            'y': s_data['speak_ypix'].unique()[0],
            'firing_rate': s_data[chan].mean()
            })
    
    speakers = pd.DataFrame(speakers)
    
    marker_dict = {'size': 12}

    if speakerC == 'BySpeaker':
        marker_dict['color'] = speakers["id"]
        marker_dict['showscale'] = False   
        marker_dict['colorscale'] =  'Blues_r' 
                    
    elif speakerC == 'ByRate':
        marker_dict['color'] = speakers['firing_rate']
        marker_dict['showscale'] = True
        marker_dict['colorscale'] = 'YlOrRd' 
        marker_dict['colorbar'] = {'title':'Mean Rate','x':1.15}
    
    fig.add_trace(
        go.Scatter(
            x=speakers['x'], 
            y=speakers['y'],                
            mode='markers',
            marker=marker_dict,
            visible=('Speakers' in scatter_opts)
            )
    )

    fig.add_trace(
        go.Scatter(
            x=stim['head_x'], 
            y=stim['head_y'],
            marker_color=stim['MCS_Time'],
            mode='markers',
            marker=dict(
                showscale=True,
                size=3,
                colorscale='Greens_r',
                colorbar={'title':'Time (s)'}
                ),
            name='Head Center',
            visible=('Head' in scatter_opts)
            )
    )

    fig.update_layout(
        xaxis={'tickmode':'array','tickvals':[]},
        yaxis={'tickmode':'array','tickvals':[]},
        title='Head Position',        
        height=500,
        width=500,
        showlegend=False,
        coloraxis_showscale=True)

    # Draw PSTH
    psth_file = str(stim_file).replace('StimSpikeCounts','PSTH')
    psth_df = pd.read_csv(psth_file)
    
    psth_df = psth_df[psth_df['Chan'] == chan]
    psth_ymax = psth_df['MeanRate'].max() * 1.2

    psth = go.Figure()
    
    psth.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, psth_ymax],
                mode = 'lines',
                line=dict(
                    color='darkgray'
                )
            )
        )


    psth.add_trace(
        go.Scatter(
            x=psth_df['Time'],
            y=psth_df['MeanRate'],
            fill = 'tozeroy'
        )
    )

    psth.update_layout(showlegend=False)
    psth.update_xaxes(title='Time post-stimulus (s)') 
    psth.update_yaxes(title='Spikes / s', range=[0, psth_ymax]) 



    # Draw distribution of head directions
    theta, rho = get_angular_histogram( stim['head_angle'].to_numpy())

    head_angle_fig = go.Figure(go.Barpolar(
        r=rho,
        theta=theta,
        thetaunit="radians",
        width=np.pi/6,
        marker_line_width=2,
        opacity=0.8
    ))

    head_angle_fig.update_layout(
        height=300,
        width=400,
        template=None,
        title='Head Angles in Image',
        polar = dict(
            radialaxis = dict(range=[0, max(rho)], showticklabels=False, ticks=''),
            angularaxis = dict(showticklabels=True)
        )
    )

    # Draw stimulus positions relative to head
    stim['h2s_theta_d'] = (stim['h2s_theta'] / np.pi) * 180
    
    if speakerC == 'BySpeaker':
    
        head_stim_fig = px.scatter_polar(stim,
            r='h2s_distance',
            theta="h2s_theta_d",
            color='Speaker',
            opacity=0.7,
            color_continuous_scale=px.colors.sequential.Blues_r
            )

    elif speakerC == 'ByRate':

        head_stim_fig = px.scatter_polar(stim,
            r='h2s_distance',
            theta="h2s_theta_d",
            color=chan,
            opacity=0.6,
            range_color=[0, 3*stim[chan].mean()],
            color_continuous_scale=px.colors.sequential.YlOrRd
            )
        
    head_stim_fig.update_layout(
        height=300,
        width=400,
        template=None,
        polar = dict(
            radialaxis = dict(
                angle = 90,
                showticklabels = False,
                ticks = ''
            ),
        ),
        title='Stimulus position w.r.t. Head')

    # Draw head-centred SRF
    srf = bin_by_angle(stim, 'h2s_theta', chan)

    srf_fig = px.line(srf, x='theta_d', y='mean_rate',
        labels={
            'theta_d': 'Sound angle w.r.t. Head',
            'mean_rate': 'Spikes / s'})
    
    srf_fig.update_layout(
        xaxis = dict(
            tickmode='linear',
            tick0 = -180,
            dtick = 90),
    )


    return [fig, head_angle_fig, head_stim_fig, psth, srf_fig]
