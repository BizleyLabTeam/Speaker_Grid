"""
PAGE 1
Test dashboard for visualizing and exploring neural activity

"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from skimage import io

from dash.dependencies import Output, Input
from pathlib import Path
from plotly.subplots import make_subplots

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



# List ferrets in project
ferret_dict = get_file_dict("data/", "F*")



layout = html.Div(
        children=[ 
            # Row 2 - Header
            html.Div(
                children=[
                    html.H1(children="Array PSTHs", className="header-title"),
                    html.P(
                        children="Find good channels quickly",                        
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
                                id="ferret-filter2",
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
                                id="session-filter2",
                                clearable=False,
                                className="dropdown",
                            ),
                        ]
                    ),                                     
                ],
                className="menu",
            ),
            # Row 3 - Images
            html.Div([
                dcc.Graph(id='rec_A'),                                
                dcc.Graph(id='rec_B'),                                
            ], 
            className='wrapper')
    ]
)


@app.callback(
    [
        Output("session-filter2", "options"),
        Output("session-filter2", "value"),
        ],
    [Input('ferret-filter2', 'value')]
    )
def update_session_list(value):
    
    session_dict = get_file_dict(value, '*Squid*')
    
    return [session_dict, session_dict[-1]['value']]


@app.callback(
    [
        Output("rec_A", "figure"),
        Output("rec_B", "figure")
        ],
    [
        Input('session-filter2', 'value'),
        ]
    )
def plot_images(value):
    
    session_dir = Path(value)
    psth_file = next( session_dir.glob('*PSTH.csv'))

    psth_df = pd.read_csv(str(psth_file))

    subplot_titles_A = [f"A{x:02d}" for x in range(1,33)]
    subplot_titles_B = [f"B{x:02d}" for x in range(1,33)]

    fig_A = make_subplots(rows=8, cols=4, subplot_titles=subplot_titles_A, shared_xaxes=True)
    fig_B = make_subplots(rows=8, cols=4, subplot_titles=subplot_titles_B)

    for headstage in ['A','B']:
        for chan in range(1, 33):

            chan_str = f"{headstage}{chan:02d}"
            chan_df = psth_df[psth_df['Chan'] == chan_str]

            # col = int(np.ceil(chan / 8))
            # row = int(chan - ((col-1) * 8))

            row = int(np.ceil(chan / 4))
            col = int(chan - ((row-1) * 4))

            if headstage == 'A':
                fig_A.add_trace(
                   go.Scatter(
                       x=chan_df['Time'],
                       y=chan_df['MeanRate']
                       ),
                    row = row,
                    col = col
                )
            elif headstage == 'B':                
                fig_B.add_trace(
                   go.Scatter(
                       x=chan_df['Time'],
                       y=chan_df['MeanRate']
                       ),
                       row=row,
                       col=col
                )

    # Find images
    
    # im_file_A = next( session_dir.glob('*LHS.png'))
    # im_file_B = next( session_dir.glob('*RHS.png'))
        
    # imgA = io.imread(str(im_file_A))
    # figA = px.imshow(imgA)
    # figA.update_layout(title=im_file_A.stem, height=1000)
    
    # imgB = io.imread(str(im_file_B))
    # figB = px.imshow(imgB)
    fig_A.update_layout(title='MCS_Recording', height=1000, showlegend=False)
    fig_B.update_layout(title='MCS_Recording_2', height=1000, showlegend=False)
    

    return [fig_A, fig_B]
