#%% Import libs 
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
from pyxtal import pyxtal
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from pymatgen.io.lammps.data import LammpsData
from shock_analysis import SimulationDump
# %%
crystal = pyxtal()

# virgin = 0, elastic = 15, post-shear = 40, unload = 68
CBCcrystal = [i for i in parse_lammps_dumps('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/orientation_shock_50-50-1000/B12-CBC/0/mid_chain/0.0/shock_-25.0_data/dump.crystalline_shock_chunk_-25.0')]

# virgin = 0, elastic = 15, amorphization = 20-30, post-shear = 40, unload = 68
CBCamorph = [i for i in parse_lammps_dumps('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/orientation_shock_50-50-1000/B12-CBC/0/mid_chain/0.0/shock_-25.0_data/dump.amorphous_shock_chunk_-25.0')]

# virgin = 0
# compressed = 15
# plastic = 40
CCCslab = [i for i in parse_lammps_dumps('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/orientation_shock_50-50-1000/B12-CCC/0/mid_chain/0.0/shock_-25.0_data/dump.slab_shock_chunk_-25.0')]

# %%
CCC = {'virgin': CCCslab[0], 'elastic': CCCslab[15], 'plastic': CCCslab[40]}
CBC_amorph = {'virgin': CBCamorph[0], 'elastic': CBCamorph[15], 'amorph_start': CBCamorph[20], 'amorph_end': CBCamorph[30], 'plastic': CBCamorph[40], 'unload': CBCamorph[68]}
CBC_crystal = {'virgin': CBCcrystal[0], 'elastic': CBCcrystal[15], 'plastic': CBCcrystal[40], 'unload': CBCcrystal[68]}
# %%
crystal.from_seed(CCC['virgin'].structure)
# %%
