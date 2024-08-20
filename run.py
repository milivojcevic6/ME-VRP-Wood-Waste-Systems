from vrp import VRP, euclidean_distance

import networkx as nx


def run(network, first_ech, leftRight, P_cap_limit, M_cap_limit, s_initial=None):
    w1 = network.w1
    w2 = network.w2
    k = network.k
    M = network.set_M

    if s_initial:
        for i in range(len(M)):
            temp_list = list(M[i])
            temp_list[2] = s_initial[len(s_initial) - len(M) + i]
            M[i] = tuple(temp_list)

    if (first_ech and leftRight) or (not first_ech and not leftRight):
        D = network.set_D[0]
        v = network.vehicles1
        cap = network.cap1
        P = network.set_P
        R = network.set_W
    else:
        D = network.set_D[1]
        v = network.vehicles2
        cap = network.cap2
        P = network.set_T
        R = network.set_F

    vrp = VRP(D, v, cap, P, M, R, P_cap_limit, M_cap_limit, w1, w2, k, first_ech, leftRight, s_initial)

    vrp.improve()

    edges = vrp.get_edges()

    s_cap_used = []
    for i in range(len(vrp.set_S)):
        s_cap_used.append((vrp.set_S[i][2] - vrp.s_cap[i]))

    print(s_cap_used)

    return vrp.nodes, edges, vrp.cost(), vrp.P_used, vrp.M_used, s_cap_used


def get_initial(network, first_ech, leftRight, P_cap_limit, M_cap_limit, s_initial=None):
    w1 = network.w1
    w2 = network.w2
    k = network.k
    M = network.set_M

    if s_initial:
        for i in range(len(M)):
            temp_list = list(M[i])
            temp_list[2] = s_initial[len(s_initial) - len(M) + i]
            M[i] = tuple(temp_list)

    if (first_ech and leftRight) or (not first_ech and not leftRight):
        D = network.set_D[0]
        v = network.vehicles1
        cap = network.cap1
        P = network.set_P
        R = network.set_W
    else:
        D = network.set_D[1]
        v = network.vehicles2
        cap = network.cap2
        P = network.set_T
        R = network.set_F

    vrp = VRP(D, v, cap, P, M, R, P_cap_limit, M_cap_limit, w1, w2, k, first_ech, leftRight, s_initial)

    edges = vrp.get_edges()

    return vrp.nodes, edges, vrp.cost()


# minimum-cost bipartite matching
def MCB_matching(network, cap1, cap2, left_right):
    A = network.set_P
    B = network.set_T

    if left_right:
        capA = cap1[:len(A)]
        capB = cap2[:len(B)]
    else:
        capA = cap2[:len(A)]
        capB = cap1[:len(B)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add source and sink
    G.add_node('s', demand=-sum(capA))
    G.add_node('t', demand=sum(capA) - sum(capB))

    # Connect source to A nodes
    for i, v in enumerate(A):
        G.add_edge('s', v, capacity=capA[i], weight=0)

    # Connect B nodes to sink
    for i, u in enumerate(B):
        G.add_edge(u, 't', capacity=u[2] - capB[i], weight=0)
        G.nodes[u]['demand'] = capB[i]

    # Connect A nodes to B nodes with costs
    for i, v in enumerate(A):
        for j, u in enumerate(B):
            cost = euclidean_distance(v, u)
            G.add_edge(v, u, capacity=float('inf'), weight=int(cost))

    flow_dict = nx.min_cost_flow(G)

    # Extract the matching from the flow
    matching = []
    cost = 0
    for i, v in enumerate(A):
        for j, u in enumerate(B):
            if flow_dict[v][u] > 0:
                matching.append((v[4], u[4], -1, flow_dict[v][u]))
                cost += flow_dict[v][u] * euclidean_distance(v, u)

    return matching, cost


# middle matching
def mid_MCB_matching(network, capA, left_right):

    if left_right:
        A = network.set_P
        B = network.set_T
    else:
        A = network.set_T
        B = network.set_P

    capM = capA[len(A):]
    capA = capA[:len(A)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add source and sink
    G.add_node('s', demand=-sum(capA))
    G.add_node('t', demand=sum(capA))

    # Connect source to A nodes
    for i, v in enumerate(A):
        G.add_edge('s', v, capacity=capA[i], weight=0)

    # Connect B nodes to sink
    for i, u in enumerate(B):
        G.add_edge(u, 't', capacity=sum(capA), weight=0)

    # Connect A nodes to B nodes with costs
    for i, v in enumerate(A):
        for j, u in enumerate(B):
            cost = euclidean_distance(v, u)  # Define your cost function here
            G.add_edge(v, u, capacity=float('inf'), weight=int(cost))

    flow_dict = nx.min_cost_flow(G)

    # Extract the matching from the flow
    capB = [0] * len(B)

    for i, v in enumerate(A):
        for j, u in enumerate(B):
            if flow_dict[v][u] > 0:
                capB[j] += flow_dict[v][u]

    s_initial = capB + capM

    return s_initial
