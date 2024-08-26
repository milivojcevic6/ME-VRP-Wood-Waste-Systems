# ME-VRP-Wood-Waste-Systems

This repository contains the code for the system developed as part of the master's thesis. The thesis focuses on the Multi-echelon Vehicle Routing Problem (ME-VRP) for wood waste collection and pre-processing.
It was done by a student Milan Milivojčević, with mentorship of the assist. prof. Balázs Dávid, at the University of Primorska.

## Project Files

- **index.py**: Main file for running the code, contains the entire code for the Dash GUI.
- **initial_network.py**: Handles generating the initial network from input, including functions like `load_nodes_from_file()`.
- **network.py**: Contains the main class `Network` for graph visualization, including plot renderings and pipeline calls.
- **run.py**: Contains functions like `get_initial()` for finding initial solutions in VRP and `run()` for running the improvement process.
- **vrp.py**: Backbone of the project, containing the `VRP` class that creates network relations and logic, including methods like `improvement()` and `get_edges()`.
- **assets/style.css**: Contains CSS code for the UI design.
- **data**: Folder that contains 33 test scenarios from the thesis.

## How to Run

To run the project, simply execute the `index.py` file:

```bash
python index.py
```

Please ensure that all dependencies are installed by running:
```bash
pip install -r requirements.txt
```
