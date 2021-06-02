"""
    Test dashboard for visualizing and exploring neural activity
    recorded from neurons in ferrets switching between auditory 
    and visual localization
    
    Stephen Town - 2021
"""

import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.SLATE]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True

# import index