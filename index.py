import dash
from dash import dcc, html, Input, Output, State
from initial_network import InitialNetwork
from network import Network
import base64

app = dash.Dash(__name__)
initial_network = InitialNetwork()
network = Network()


app.layout = html.Div([
    html.Div([
        html.H2('Menu'),
        dcc.Upload(id='upload-data', children=html.Button('Load Data')),
        html.Br(),
        html.Button('Add Node', id='add-node-button'),
        dcc.Dropdown(
            id='node-type-dropdown',
            options=[
                {'label': 'Depot', 'value': 'depo'},
                {'label': 'Pre-Processing Unit', 'value': 'pre_processing'},
                {'label': 'Treatment Unit', 'value': 'treatment'},
                {'label': 'Combined Unit', 'value': 'combined'},
                {'label': 'W Client', 'value': 'w_client'},
                {'label': 'F Customer', 'value': 'f_customer'}
            ],
            placeholder='Select Node Type',
            style={'display': 'none'}
        ),
        dcc.Input(id='input-x', type='number', placeholder='X Coordinate', step=0.1, style={'display': 'none'}),
        dcc.Input(id='input-y', type='number', placeholder='Y Coordinate', step=0.1, style={'display': 'none'}),
        dcc.Input(id='input-capacity-demand', type='number', placeholder='Capacity/Demand', style={'display': 'none'}),
        dcc.Store(id='add-node-clicks', data=0),
        html.Br(),
        html.H4('Coefficient k:', className='centered-text'),
        dcc.Input(id='input-k', type='number', placeholder='k Value', value=network.k),
        html.H4('Coefficient w1:', className='centered-text'),
        dcc.Input(id='input-w1', type='number', placeholder='w1 Value', step=0.1, value=network.w1),
        html.H4('Coefficient w2:', className='centered-text'),
        dcc.Input(id='input-w2', type='number', placeholder='w2 Value', step=0.1, value=network.w2),
        html.Button('Set Variables', id='set-variables-button'),
        html.Button('Run Algorithm', id='run-algorithm-button', style={'margin-top': '20px'}),
        html.H4('History', className='centered-text'),
        html.Div([
            html.Button(html.Img(src='/assets/left.png', style={'height': '30px'}), id='prev-button', style={'border': 'none', 'background': 'none'}),
            html.Button(html.Img(src='/assets/right.png', style={'height': '30px'}), id='next-button', style={'border': 'none', 'background': 'none'})
        ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'gap': '10px'})
    ], className='menu-container'),
    html.Div([
        dcc.Graph(id='graph', style={'height': '100vh'}),
    ], className='graph-container'),
], style={'height': '100vh'})


@app.callback(
    [Output('node-type-dropdown', 'style'),
     Output('input-x', 'style'),
     Output('input-y', 'style'),
     Output('input-capacity-demand', 'style'),
     Output('add-node-clicks', 'data')],
    [Input('add-node-button', 'n_clicks')],
    [State('node-type-dropdown', 'style'),
     State('input-x', 'style'),
     State('input-y', 'style'),
     State('input-capacity-demand', 'style'),
     State('add-node-clicks', 'data')]
)
def toggle_node_inputs(n_clicks, dropdown_style, x_style, y_style, capacity_demand_style, add_node_clicks):
    if n_clicks:
        if add_node_clicks == 0:
            return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, 1
        else:
            return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, 0
    return dropdown_style, x_style, y_style, capacity_demand_style, add_node_clicks


@app.callback(
    [Output('graph', 'figure'),
     Output('input-k', 'value'),
     Output('input-w1', 'value'),
     Output('input-w2', 'value')],
    [Input('upload-data', 'contents'),
     Input('add-node-clicks', 'data'),
     Input('set-variables-button', 'n_clicks'),
     Input('run-algorithm-button', 'n_clicks'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('node-type-dropdown', 'value'),
     State('input-x', 'value'),
     State('input-y', 'value'),
     State('input-capacity-demand', 'value'),
     State('input-k', 'value'),
     State('input-w1', 'value'),
     State('input-w2', 'value')]
)
def update_graph(contents, add_node_clicks, set_vars_clicks, run_algo_clicks, prev_clicks, next_clicks,
                 filename, node_type, x, y, capacity_demand,
                 k, w1, w2):
    ctx = dash.callback_context

    if not ctx.triggered:
        return network.plot_network(), network.k, network.w1, network.w2

    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_input == 'upload-data':
        if contents is not None:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            with open(f'data/{filename}', 'wb') as f:
                f.write(decoded)
            initial_network.load_nodes_from_file(f'data/{filename}')
            network.copy_from_initial(initial_network)
        return network.plot_network(), network.k, network.w1, network.w2

    elif triggered_input == 'add-node-clicks' and add_node_clicks == 0:
        if node_type and x is not None and y is not None:
            if node_type in ['pre_processing', 'treatment', 'combined']:
                network.add_node(x, y, node_type, capacity=capacity_demand)
            elif node_type in ['w_client', 'f_customer']:
                network.add_node(x, y, node_type, demand=capacity_demand)
            else:
                network.add_node(x, y, node_type)

    elif triggered_input == 'set-variables-button':
        network.k = k
        network.w1 = w1
        network.w2 = w2

    elif triggered_input == 'run-algorithm-button':

        network.call_pipeline(initial_network)

        return network.plot_network(), network.k, network.w1, network.w2

    elif triggered_input == 'prev-button' and network.current_state > 0:
        network.current_state -= 1
        network.load_state(network.current_state)
        return network.plot_network(), network.k, network.w1, network.w2

    elif triggered_input == 'next-button' and network.current_state < len(network.history) - 1:
        network.current_state += 1
        network.load_state(network.current_state)
        return network.plot_network(), network.k, network.w1, network.w2

    return network.plot_network(), network.k, network.w1, network.w2


if __name__ == '__main__':
    app.run_server(debug=True)
