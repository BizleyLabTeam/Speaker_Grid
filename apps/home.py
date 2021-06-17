import dash_html_components as html
import dash_bootstrap_components as dbc

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Speaker Grid Demo", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        
        dbc.Row([
            dbc.Col(html.H5(children='Online repository for data from ferrets localizing sounds or lights'),
                     className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    children=[
                        html.H3(
                            children='Go to dashboards',
                            className="text-center"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Spatial Receptive Fields",
                                        href="/page1",
                                        color="primary"
                                        ),
                                    dbc.Button(
                                        "Array PSTHs",
                                        href="/page2",
                                        color="secondary"
                                        )
                                ],className="mt-2"
                                    ),
                                ]),
                            ],
                            body=True,
                            color="dark",
                            outline=True
                            ),
                    width=12,
                    className="mb-4"
                    ),
                    ], 
            className="mb-5"
        ),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Get the Code',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/stephentown42/Task_Switching",
                                                  color="primary",
                                                  className="mt-2"),                                       
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),               
        ], className="mb-5"),
    ])
])
