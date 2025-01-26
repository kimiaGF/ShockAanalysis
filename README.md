# Shock Analysis Toolkit for LAMMPS Simulations

This repository contains Python tools for analyzing shock behavior in materials using molecular dynamics simulations performed with LAMMPS. These scripts streamline the processing of simulation data, enabling in-depth analysis, binning, and visualization of key properties.

## Features

- **Simulation Data Processing:** Extract, bin, and label data from LAMMPS simulation dumps.
- **Customizable Binning:** Divide simulation data into spatial bins for localized analysis.
- **Hugoniot State Calculation:** Calculate key properties of materials under shock conditions.
- **Data Visualization:** Generate interactive plots using Plotly for time-series and spatial data.
- **Integrated Logging:** Monitor and debug processes with structured log outputs.

## Requirements

This toolkit requires the following Python libraries:

- `numpy`
- `matplotlib`
- `pandas`
- `plotly`
- `tqdm`
- `scipy`
- `pymatgen`
- `logging`

Install the dependencies using pip:

```bash
pip install numpy matplotlib pandas plotly tqdm scipy pymatgen
```

## Usage

### 1. Data Processing

To process simulation files and create a structured DataFrame for analysis:

```python
from shock_analysis import process_files

files = ["path/to/file1.dump", "path/to/file2.dump"]
sim_df = process_files(files, ts=0.2)  # Specify the timestep in fs
```

### 2. Binning Simulation Data

To bin data spatially:

```python
from shock_analysis import bin_data

data, bin_atom_map = bin_data(df, num_bins=100, bin_direction='z')
```

### 3. Visualizing Data

Generate interactive time-series plots:

```python
from shock_analysis import SimulationDump

sim = SimulationDump("path/to/dump.file")
fig = sim.plot_time_data(x="timestep", y="temperature")
fig.show()
```

### 4. Hugoniot State Calculation

To compute the average properties in specific regions and time intervals:

```python
hugoniot_data = sim.get_hugoniot_state(bin1=10, bin2=20, t1=0, t2=100)
print(hugoniot_data)
```

## File Structure

- `shock_analysis.py`: Main script containing all helper functions and classes for data processing and visualization.
- `logs/`: Directory to store log files.
- `data_files/`: Directory for input and output simulation files.

## Logging

Logs are stored in `logs/shock_analysis.log`. Adjust logging levels in the script if needed to monitor specific processes.

## Example Workflow

Below is an example workflow for processing and visualizing simulation data:

```python
from shock_analysis import SimulationDump

# Initialize a SimulationDump object
sim = SimulationDump("path/to/dump.file")

# Process and visualize time-series data
fig = sim.plot_time_data(x="timestep", y="temperature")
fig.show()

# Calculate Hugoniot state properties
hugoniot_data = sim.get_hugoniot_state()
print(hugoniot_data)
```

## Author
Kimia Ghaffari

## License
MIT License

## Acknowledgments
Special thanks to the developers of [pymatgen](https://pymatgen.org) and the scientific computing community for their contributions to molecular dynamics and data analysis.
