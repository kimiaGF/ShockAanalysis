#%%

from pymatgen.io.lammps.outputs import LammpsDump
from io import StringIO
from glob import glob
#%%
path_to_lammps_dump = glob('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/NVT_capped/orientation_shock_50-50-500_v7/B11-Cp-CBC/0/mid_chain/0.0/shock_-25.0_data/dump.data_shock_chunk_-25.0')


files = []
for f in path_to_lammps_dump:
    polytype = f.split('/')[-4]
    vac_frac = float(f.split('/')[-3])
    velocity = -1*float(f.split('_')[-1])

    if vac_frac >= 0.005 and velocity >= 20 and velocity <= 30:
        files.append(f)
#%% find bad data

def filter_files(files,vel_list=[25.0,20.0,30.0], vac_list=[0.005,0.01,0.05,0.1],polytype_list = ['B11-Ce-CBC','B11-Cp-CBC','B12-CCC','B12-CBC']):
   
    filtered_files = []
    for f in files:
        polytype = f.split('/')[-4]
        vac_frac = float(f.split('/')[-3])
        velocity = -1*float(f.split('_')[-1])
        if vac_frac in vac_list and velocity in vel_list and polytype in polytype_list:
            filtered_files.append(f)
    return filtered_files

def find_bad_files(files):
    bad_files = []
    
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        n_bad = 0
        for i,line in enumerate(lines):
            #find bad lines
            if 'ITEM: TIMESTEP' in line and len(line) > len('ITEM: TIMESTEP\n'):
                bad_files.append(file)
                break
    return bad_files

def bad_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
    
    bad_lines = []
    
    for i,line in enumerate(lines):
        if 'ITEM: TIMESTEP' in line and len(line) > len('ITEM: TIMESTEP\n'):
            # print(i+1)
            bad_lines.append(i+1)
    
    return bad_lines
    
#%%
bad_files = find_bad_files(path_to_lammps_dump)
for file in bad_files:
    with open(file) as f:
        lines = f.readlines()

    n_bad = 0
    for i,line in enumerate(lines):
        #find bad lines
        if 'ITEM: TIMESTEP' in line and len(line) > len('ITEM: TIMESTEP\n'):
            n_bad += 1
    
    
    # print(file.split('B11-Cp-CBC')[-1])
    # print(n_bad)

#%%
len(bad_lines(bad_files[0]))


# %%
from pymatgen.io.lammps.outputs import parse_lammps_dumps
dumps = [i for i in parse_lammps_dumps(path_to_lammps_dump[0])]
# %%
