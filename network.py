# network.py

import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import copy
import time
import run
import csv
import os


# saving results locally
def save_csv(TLFIN, TLFIM, TLSIN, TLSIM, TLM, TRFIN, TRFIM, TRSIN, TRSIM, TRM,
             CLFIN, CLFIM, CLSIN, CLSIM, CLM, CRFIN, CRFIM, CRSIN, CRSIM, CRM,
             w1, w2, file_name):
    headers = ["Dataset", "Parameters", "Total cost L2R", "Improvement L2R", "Total runtime L2R", "Total cost R2L",
               "Improvement R2L", "Total runtime R2L"]

    dataset = file_name
    parameters = "w1 = " + str(w1) + ", w2 = " + str(w2)

    total_cost_left = str(round(CLFIN + CLFIM + CLSIN + CLSIM + CLM, 2))
    total_cost_right = str(round(CRFIN + CRFIM + CRSIN + CRSIM + CRM, 2))

    cost_improvement_left = str(round(100 - (CLFIM + CLSIM) / (CLFIN + CLSIN) * 100, 2)) + "%"
    cost_improvement_right = str(round(100 - (CRFIM + CRSIM) / (CRFIN + CRSIN) * 100, 2)) + "%"

    total_run_left = str(round(TLFIN + TLFIM + TLSIN + TLSIM + TLM, 2))
    total_run_right = str(round(TRFIN + TRFIM + TRSIN + TRSIM + TRM, 2))

    row = [dataset, parameters, total_cost_left, cost_improvement_left, total_run_left,
           total_cost_right, cost_improvement_right, total_run_right]

    file = "output.csv"

    file_exists = os.path.isfile(file)

    with open(file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file doesn't exist, write the headers
        if not file_exists:
            writer.writerow(headers)

        # Write the row
        writer.writerow(row)


class Network:
    def __init__(self):
        self.vehicles1 = 0
        self.vehicles2 = 0
        self.cap1 = 0
        self.cap2 = 0
        self.set_D = []
        self.set_P = []
        self.set_T = []
        self.set_M = []
        self.set_W = []
        self.set_F = []
        self.P = 0
        self.T = 0
        self.M = 0
        self.W = 0
        self.F = 0
        self.w1 = 10.0
        self.w2 = 100.0
        self.k = 5
        self.score = 0.0
        self.title = "Welcome!"
        self.nodes = []
        self.edges = []
        self.node_colors = {
            'depo': 'red',
            'pre_processing': 'blue',
            'treatment': 'green',
            'combined': 'orange',
            'w_client': 'grey',
            'f_client': 'pink'
        }
        self.history = []
        self.current_state = -1

    def save_state(self):
        # Clone the current state of the network
        state = {
            'nodes': copy.deepcopy(self.nodes),
            'edges': copy.deepcopy(self.edges),
            'vehicles1': self.vehicles1,
            'vehicles2': self.vehicles2,
            'cap1': self.cap1,
            'cap2': self.cap2,
            'P': self.P,
            'T': self.T,
            'M': self.M,
            'W': self.W,
            'F': self.F,
            'w1': self.w1,
            'w2': self.w2,
            'k': self.k,
            'score': self.score
        }
        self.history.append(state)
        self.current_state += 1

    def copy_from_initial(self, in_network):
        self.nodes = copy.deepcopy(in_network.nodes)
        self.edges = copy.deepcopy(in_network.edges)
        self.vehicles1 = in_network.vehicles1
        self.vehicles2 = in_network.vehicles2
        self.cap1 = in_network.cap1
        self.cap2 = in_network.cap2
        self.set_D = in_network.set_D
        self.set_P = in_network.set_P
        self.set_T = in_network.set_T
        self.set_M = in_network.set_M
        self.set_W = in_network.set_W
        self.set_F = in_network.set_F
        self.P = in_network.P
        self.T = in_network.T
        self.M = in_network.M
        self.W = in_network.W
        self.F = in_network.F
        self.w1 = in_network.w1
        self.w2 = in_network.w2
        self.k = in_network.k
        self.save_state()

    def load_state(self, state_index):
        # Load a specific state from the history
        state = self.history[state_index]
        self.nodes = state['nodes']
        self.edges = state['edges']
        self.vehicles1 = state['vehicles1']
        self.vehicles2 = state['vehicles2']
        self.cap1 = state['cap1']
        self.cap2 = state['cap2']
        self.P = state['P']
        self.T = state['T']
        self.M = state['M']
        self.W = state['W']
        self.F = state['F']
        self.w1 = state['w1']
        self.w2 = state['w2']
        self.k = state['k']
        self.score = state['score']

    def add_node(self, x, y, node_type, **kwargs):
        node = {
            'x': x,
            'y': y,
            'type': node_type,
            'params': kwargs
        }
        self.nodes.append(node)
        # self.save_state()  # Save the state after adding a node

    def add_edge(self, start_node_index, end_node_index):
        self.edges.append((start_node_index, end_node_index))

    def to_networkx_graph(self):
        G = nx.Graph()
        for i, node in enumerate(self.nodes):
            G.add_node(i, **node)
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])
        return G

    def node_formatting(self, vrp_nodes):

        self.nodes = []
        for node in vrp_nodes:
            # print(node)
            if node[3] == "depo":
                self.add_node(node[0], node[1], node[3])
            elif node[3] == "pre_processing" or node[3] == "treatment" or node[3] == "combined":
                self.add_node(node[0], node[1], node[3], capacity=node[2])
            else:
                self.add_node(node[0], node[1], node[3], demand=node[2])

    def plot_network(self):
        fig = go.Figure()
        unique_node_types = set()
        for node in self.nodes:
            if node['type'] not in unique_node_types:
                unique_node_types.add(node['type'])
                showlegend = True
            else:
                showlegend = False

            hover_text = f"Position is ({node['x']}, {node['y']})"
            if node['type'] != 'depo':
                if 'capacity' in node['params']:
                    hover_text += f"<br>Capacity is {node['params']['capacity']}"
                if 'demand' in node['params']:
                    hover_text += f"<br>Demand is {node['params']['demand']}"

            node_size = 20 if node['type'] == 'depo' else 15 if (node['type'] == 'pre_processing'
                                                                 or node['type'] == 'treatment'
                                                                 or node['type'] == 'combined') else 10

            node_name = "Depot" if node['type'] == 'depo' else \
                "Pre-processing unit" if node['type'] == 'pre_processing' else \
                    "Treatment unit" if node['type'] == 'treatment' else \
                        "Combined" if node['type'] == 'combined' else \
                            "Collection location" if node['type'] == 'w_client' else \
                                "Final location"

            fig.add_trace(go.Scatter(
                x=[node['x']],
                y=[node['y']],
                mode='markers',
                # marker=dict(color=self.node_colors[node['type']], size=node_size),
                marker=dict(
                    color=self.node_colors[node['type']],
                    size=node_size,
                    symbol='square' if node['type'] == 'depo' else 'circle'  # Set to square for depots
                ),
                name=node_name,
                showlegend=showlegend,
                text=hover_text,
                hoverinfo='text'
            ))

        # Create a colormap
        unique_values = list(set(edge[2] for edge in self.edges))
        colormap = px.colors.qualitative.Plotly
        color_mapping = {val: colormap[i % len(colormap)] for i, val in enumerate(unique_values)}

        # Dictionary to store the count of edges between node pairs
        edge_counts = {}

        for edge in self.edges:
            start_node = self.nodes[edge[0]]
            end_node = self.nodes[edge[1]]

            # Sort the nodes to ensure that (start, end) is treated the same as (end, start)
            node_pair = tuple(sorted([edge[0], edge[1]]))

            # Count the number of edges between this pair of nodes
            if node_pair not in edge_counts:
                edge_counts[node_pair] = 0
            edge_counts[node_pair] += 1

            # Calculate offset for curvature
            edge_index = edge_counts[node_pair]
            offset = 0.5 * (edge_index - 1)  # Adjust the 0.1 factor to control curvature strength

            # Midpoint for bezier curve
            mid_x = (start_node['x'] + end_node['x']) / 2
            mid_y = (start_node['y'] + end_node['y']) / 2

            # Apply perpendicular offset for the midpoint
            dx = end_node['x'] - start_node['x']
            dy = end_node['y'] - start_node['y']
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length == 0:  # Avoid division by zero
                length = 1
            norm_dx = -dy / length
            norm_dy = dx / length

            ctrl_x = mid_x + norm_dx * offset
            ctrl_y = mid_y + norm_dy * offset

            # Create the bezier curve points (simple quadratic approximation)
            bezier_x = [start_node['x'], ctrl_x, end_node['x']]
            bezier_y = [start_node['y'], ctrl_y, end_node['y']]

            # Set the edge color and line style
            if edge[2] == -1:
                edge_color = 'black'
                line_style = 'solid'
            else:
                edge_color = color_mapping[edge[2]]
                line_style = 'solid' if edge[3] else 'dot'

            # Add the curved edge to the figure
            fig.add_trace(go.Scatter(
                x=bezier_x,
                y=bezier_y,
                mode='lines',
                line=dict(color=edge_color, dash=line_style),
                showlegend=False
            ))

        # Adding dropdown menu for filtering
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": "Whole network",
                            "method": "restyle",
                            "args": [{"visible": True}],
                            "args2": [{"visible": "legendonly"}],  # Revert to default on reset
                        },
                        {
                            "label": "First VRP",
                            "method": "restyle",
                            # "args": [{"visible": [edge[3] and edge[2] != -1 for edge in self.edges]}],
                            "args": [{"visible": True}],
                        },
                        {
                            "label": "Second VRP",
                            "method": "restyle",
                            # "args": [{"visible": [not edge[3] and edge[2] != -1 for edge in self.edges]}],
                            "args": [{"visible": True}],
                        },
                        {
                            "label": "Facilities",
                            "method": "restyle",
                            # "args": [{"visible": [edge[2] == -1 for edge in self.edges]}],
                            "args": [{"visible": True}],
                        }
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 1.1,  # Slightly outside the plot area to the right
                    "xanchor": "left",
                    "y": 0.7,  # Positioning below the legend
                    "yanchor": "top",
                }
            ]
        )

        # Adding a label as an annotation
        fig.update_layout(
            annotations=[
                dict(
                    text="Visualize: ",  # Label text
                    x=1.25,  # Same x as the dropdown
                    xref="paper",
                    y=0.75,  # Slightly above the dropdown menu
                    yref="paper",
                    align="center",
                    font=dict(
                        size=16,  # Increase the font size here
                        # color="black"  # Optional: set font color
                    ),
                    showarrow=False
                )
            ]
        )

        return fig

    # Pipeline:
    # network -> initial solution for FE -> the best possible solution for FE ->
    # -> initial solution for SE -> the best possible solution for the SE -> minimum-cost bipartite matching
    def call_pipeline(self, init_network):

        P_cap_limit = float('Inf')
        M_cap_limit = float('Inf')

        print("Initial Solution First Echelon Left-Right")
        # print("Set D", init_network.set_D)
        start = time.time()
        result = run.get_initial(init_network, True, True, P_cap_limit, M_cap_limit)
        time_left_first_initial = time.time() - start
        cost_left_first_initial = result[2]

        # self.node_formatting(result[0])
        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]  # total cost
        self.save_state()
        # print("Cost", result[2])

        print("First Echelon Improvement")
        start = time.time()
        result = run.run(init_network, True, True, P_cap_limit, M_cap_limit)
        time_left_first_improved = time.time() - start - time_left_first_initial
        cost_left_first_improved = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]
        self.save_state()
        # print("Cost", result[2])

        P_cap_limit = result[3]
        M_cap_limit = result[4]
        final_edge_set = result[1]
        total_cost = result[2]
        p_fac_cap = result[5]

        s_fac_cap = result[5]
        s_initial = run.mid_MCB_matching(init_network, s_fac_cap, left_right=True)

        print("Initial Solution Second Echelon Left-Right")
        start = time.time()
        result = run.get_initial(init_network, False, True, P_cap_limit, M_cap_limit, s_initial)
        time_left_second_initial = time.time() - start
        cost_left_second_initial = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]  # total cost
        self.save_state()

        print("Second Echelon Improvement")
        start = time.time()
        result = run.run(init_network, False, True, P_cap_limit, M_cap_limit, s_initial)
        time_left_second_improved = time.time() - start - time_left_second_initial
        cost_left_second_improved = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]
        self.save_state()
        final_edge_set += result[1]
        total_cost += result[2]
        t_fac_cap = result[5]

        print("In the Matching")
        start = time.time()
        result = run.MCB_matching(init_network, p_fac_cap, t_fac_cap, True)
        time_left_matching = time.time() - start
        cost_left_matching = result[1]
        self.nodes = init_network.nodes
        self.edges = result[0]
        total_cost += result[1]
        self.save_state()
        final_edge_set += result[0]

        self.nodes = init_network.nodes
        self.edges = final_edge_set
        # self.score += result[2]
        self.save_state()
        left_right = total_cost

        print("\n\nRight to left!!!")

        P_cap_limit = float('Inf')
        M_cap_limit = float('Inf')

        print("Initial Solution First Echelon Right-Left")

        start = time.time()
        result = run.get_initial(init_network, True, False, P_cap_limit, M_cap_limit)
        time_right_first_initial = time.time() - start
        cost_right_first_initial = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]  # total cost
        self.save_state()
        print("Cost", result[2])

        print("First Echelon Improvement")
        start = time.time()
        result = run.run(init_network, True, False, P_cap_limit, M_cap_limit)
        time_right_first_improved = time.time() - start - time_left_first_initial
        cost_right_first_improved = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]
        self.save_state()

        P_cap_limit = result[3]
        M_cap_limit = result[4]
        final_edge_set = result[1]
        total_cost = result[2]
        p_fac_cap = result[5]

        s_fac_cap = result[5]
        s_initial = run.mid_MCB_matching(init_network, s_fac_cap, left_right=True)

        print("Initial Solution Second Echelon Right-Left")
        start = time.time()
        result = run.get_initial(init_network, False, False, P_cap_limit, M_cap_limit, s_initial)
        time_right_second_initial = time.time() - start
        cost_right_second_initial = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]  # total cost
        self.save_state()

        print("Second Echelon Improvement")
        start = time.time()
        result = run.run(init_network, False, False, P_cap_limit, M_cap_limit, s_initial)
        time_right_second_improved = time.time() - start - time_right_second_initial
        cost_right_second_improved = result[2]

        self.nodes = init_network.nodes
        self.edges = result[1]
        self.score = result[2]
        self.save_state()

        final_edge_set += result[1]
        total_cost += result[2]
        t_fac_cap = result[5]

        print("In Matching")

        start = time.time()
        result = run.MCB_matching(init_network, p_fac_cap, t_fac_cap, False)
        time_right_matching = time.time() - start
        cost_right_matching = result[1]

        self.nodes = init_network.nodes
        self.edges = result[0]
        total_cost += result[1]
        self.save_state()
        final_edge_set += result[0]

        print("Final")
        self.nodes = init_network.nodes
        self.edges = final_edge_set
        self.save_state()
        right_left = total_cost

        save_csv(time_left_first_initial,
                 time_left_first_improved,
                 time_left_second_initial,
                 time_left_second_improved,
                 time_left_matching,
                 time_right_first_initial,
                 time_right_first_improved,
                 time_right_second_initial,
                 time_right_second_improved,
                 time_right_matching,
                 cost_left_first_initial,
                 cost_left_first_improved,
                 cost_left_second_initial,
                 cost_left_second_improved,
                 cost_left_matching,
                 cost_right_first_initial,
                 cost_right_first_improved,
                 cost_right_second_initial,
                 cost_right_second_improved,
                 cost_right_matching,
                 init_network.w1,
                 init_network.w2,
                 init_network.file
                 )

        # add costs for everything

        print("Better is " + ("Left to Right" if right_left > left_right else "Right to Left"))
        print("Total cost Left to Right: ", str(left_right))
        print("Total cost Right to Left: ", str(right_left))
