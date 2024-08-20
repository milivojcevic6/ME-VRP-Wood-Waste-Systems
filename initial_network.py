class InitialNetwork:

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
        self.w1 = 0.0
        self.w2 = 0.0
        self.k = 0
        self.file = ""
        self.score = 0.0
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

    def add_node(self, x, y, node_type, **kwargs):
        node = {
            'x': x,
            'y': y,
            'type': node_type,
            'params': kwargs
        }
        self.nodes.append(node)

    def add_edge(self, start_node_index, end_node_index):
        self.edges.append((start_node_index, end_node_index))

    def load_nodes_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Read first line of integers and floats
            first_line = lines[1].split()
            self.vehicles1 = int(first_line[0])
            self.vehicles2 = int(first_line[1])
            self.cap1 = int(first_line[2])
            self.cap2 = int(first_line[3])
            self.P = int(first_line[4])
            self.T = int(first_line[5])
            self.M = int(first_line[6])
            self.W = int(first_line[7])
            self.F = int(first_line[8])
            self.w1 = float(first_line[9])
            self.w2 = float(first_line[10])
            self.k = int(first_line[11])
            self.file = file_path

            current_line = 4  # Skip the header and first data line

            node_index = 0

            # Read depots
            for _ in range(2):
                parts = lines[current_line].split()
                x, y = float(parts[1]), float(parts[2])
                self.add_node(x, y, 'depo')
                self.set_D.append((float(parts[1]), float(parts[2]), -1, 'depo', node_index))
                current_line += 1
                node_index += 1

            # Read pre-processing units
            current_line += 2  # Skip the header
            for _ in range(self.P):
                parts = lines[current_line].split()
                x, y, capacity = float(parts[1]), float(parts[2]), int(parts[3])
                self.add_node(x, y, 'pre_processing', capacity=capacity)
                self.set_P.append((float(parts[1]), float(parts[2]), int(parts[3]), 'pre_processing', node_index))
                current_line += 1
                node_index += 1

            # Read treatment units
            current_line += 2  # Skip the header
            for _ in range(self.T):
                parts = lines[current_line].split()
                x, y, capacity = float(parts[1]), float(parts[2]), int(parts[3])
                self.add_node(x, y, 'treatment', capacity=capacity)
                self.set_T.append((float(parts[1]), float(parts[2]), int(parts[3]), 'treatment', node_index))
                current_line += 1
                node_index += 1

            # Read combined units
            current_line += 2  # Skip the header
            for _ in range(self.M):
                parts = lines[current_line].split()
                x, y, capacity = float(parts[1]), float(parts[2]), int(parts[3])
                self.add_node(x, y, 'combined', capacity=capacity)
                self.set_M.append((float(parts[1]), float(parts[2]), int(parts[3]), 'combined', node_index))
                current_line += 1
                node_index += 1

            # Read W clients
            current_line += 2  # Skip the header
            for _ in range(self.W):
                parts = lines[current_line].split()
                x, y, demand = float(parts[1]), float(parts[2]), int(parts[3])
                self.add_node(x, y, 'w_client', demand=demand)
                self.set_W.append((float(parts[1]), float(parts[2]), int(parts[3]), 'w_client', node_index))
                current_line += 1
                node_index += 1

            # Read F customers
            current_line += 2  # Skip the header
            for _ in range(self.F):
                parts = lines[current_line].split()
                x, y, demand = float(parts[1]), float(parts[2]), int(parts[3])
                self.add_node(x, y, 'f_client', demand=demand)
                self.set_F.append((float(parts[1]), float(parts[2]), int(parts[3]), 'f_client', node_index))
                current_line += 1
                node_index += 1
