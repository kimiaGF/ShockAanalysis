#%% Import Libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import subprocess
pio.renderers.default = 'notebook'
# from dingsound import ding2 as ding
from scipy.signal import savgol_filter
import gzip
from tqdm.notebook import tqdm
from glob import glob 
import os 
import time 
import logging 
import pickle
from functools import lru_cache

from pymatgen.core.structure import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.outputs import LammpsDump, parse_lammps_dumps, parse_lammps_log
import numpy as np
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d


#Set up logger 
#----“----------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)s:%(message)s')

if os.path.isdir('logs') == False:
    os.mkdir('logs')
    
file_handler = logging.FileHandler('logs/shock_analysis.log','w+')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# Set up plotly figure layout template

#define custom colorway for plots
academic_colors = [
    "#000000",  # Black
    "#E24A33",  # Red
    "#348ABD",  # Blue
    "#8EBA42",  # Green
    "#988ED5",  # Purple
    "#FBC15E",  # Orange
    "#00C2F9",  # Cyan
    "#1F77B4",  # Dark Blue
    "#2CA02C",  # Dark Green
    "#D62728",  # Dark Red
    "#FFD700"   # Gold
]

# Define a layout template for plots
academic_template = {
    'layout': go.Layout(
        title={
            'font': {
                'family': 'Arial, sans-serif',
                'size': 18,
                'color': '#333'
            },
            'x': 0.5,  # Center the title
            'xanchor': 'center'
        },
        xaxis={
            'title': {
                'font': {
                    'family': 'Arial, sans-serif',
                    'size': 14,
                    'color': '#333'
                }
            },
            'tickfont': {
                'family': 'Arial, sans-serif',
                'size': 12,
                'color': '#333'
            },
            'linecolor': 'black',
            'ticks': 'outside',
            'showline': True,
            'mirror': 'ticks',
            'showgrid': False
        },
        yaxis={
            'title': {
                'font': {
                    'family': 'Arial, sans-serif',
                    'size': 14,
                    'color': '#333'
                }
            },
            'tickfont': {
                'family': 'Arial, sans-serif',
                'size': 12,
                'color': '#333'
            },
            'linecolor': 'black',
            'ticks': 'outside',
            'showline': True,
            'mirror': 'ticks',
            'showgrid': False
        },
        plot_bgcolor='white',
        margin={
            'l': 60,
            'r': 20,
            't': 40,
            'b': 60
        },
        legend={
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12,
                'color': '#333'
            },
            'bordercolor': '#E1E1E1',
            'borderwidth': 1,
            'x': 1,
            'y': 1,
            'xanchor': 'right',
            'yanchor': 'top'
        },
        width=600,
        height=400,
        hoverlabel={
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12,
                'color': '#333'
            }
        },
        colorway=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    )
}

#Data reading and parsing scripts helper functions
#--------------------------------------------------
def process_files(fs,ts=0.2):
    files = pd.DataFrame()
    for f in fs:
        polytype = f.split('/')[-6]
        angle = float(f.split('/')[-5])
        vac_frac = float(f.split('/')[-3])
        removal = f.split('/')[-4]
        velocity = float(f.split('_')[-1])
        
        df = pd.DataFrame(
            {
                'polytype': polytype,
                'angle': angle,
                'vac_frac': vac_frac,
                'removal': removal,
                'velocity': velocity,
                'file': f
            }, index=[0]
        )
        
        files = pd.concat([files, df], ignore_index=True)
    
    files['SimDump'] = files['file'].apply(lambda x: SimulationDump(x,timestep=ts))
    
    sim_df = files.set_index(['polytype','angle','removal','vac_frac','velocity']).loc[:,['SimDump']]
    
    return sim_df

def bin_data(df, num_bins, bin_direction='z', bin_id_name='bin_id'):
    """
    Evenly divides a Pandas DataFrame into bins and appends or replaces a column with bin ID numbers.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        num_bins (int): The number of bins to create.
        value_key (str): The column name in the DataFrame to use for binning.
        bin_id_col (str): The name of the column to store the bin ID numbers.

    Returns:
        pd.DataFrame: A new DataFrame with the 'bin_id_col' column added or replaced.
    """
    # Calculate bin edges with +- 1% tolerance to include edge cases
    bin_edges = [df[bin_direction].min()*0.99]
    bin_edges.extend(df[bin_direction].quantile(q=i/num_bins) for i in range(1, num_bins))
    bin_edges.append(df[bin_direction].max()*1.01)

    # Overwrite existing 'bin_id' columns
    if bin_id_name in df.columns:
        df[bin_id_name] = pd.cut(df[bin_direction], bins=bin_edges, labels=False)
    else:
        df.insert(loc=len(df.columns), column=bin_id_name, value=pd.cut(df[bin_direction], bins=bin_edges, labels=False))
    
    #get atom-bin mapping DF
    bin_atoms = df.groupby(bin_id_name).size().reset_index()
    bin_atoms = bin_atoms.rename(columns={0:'natoms'})
    min_atoms = bin_atoms['natoms'].min()
    max_atoms = bin_atoms['natoms'].max()
    avg_atoms = bin_atoms['natoms'].mean()
    logger.info(f'Number of bins: {num_bins}')
    logger.info(f'Average number of atoms in each bin: {avg_atoms}')
    logger.info(f'Min number of atoms in each bin: {min_atoms}')
    logger.info(f'Max number of atoms in each bin: {max_atoms}')
    bin_atom_map = df[['id','bin_id']]


    return df,bin_atom_map

def label_bins(df,atom_bin_map):
    return df.merge(atom_bin_map,on='id',how='left')

def get_dfs(dump_list:list,filename=None,num_bins=100,ts=0.2):
    """Converts list of LammpsDump objects to a single Pandas DataFrame with each timestep as a key

    Args:
        dump_list (list): LammpsDump object list containing snapshots from one simualtion
        filename(str): file path/name to save resulting dictionary to (optional)
        ts (float): Lammps timestep in fs (optional)

    Returns:
        dict: nested dictionary containing a pandas DataFrame for each snapshot in dump_list. The snapshots are keys.
    """

    all_df_data = {}
    for dump in dump_list:
        t = dump.timestep*ts
        df_temp = dump.data

        #take lowercase version of all keys 
        df = pd.DataFrame({key.lower(): value for key, value in df_temp.items()})
        
        #take reference information
        if t == 0:
            binned_df,atom_bin_map = bin_data(df,num_bins,'z','bin_id')
            
        else:
            #map atom_ids to their predetermined bins
            binned_df = label_bins(df,atom_bin_map)

        all_df_data[t] = binned_df
    
    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(all_df_data,f)
        logger.info(f'Data written to: {filename}')
    
    logger.info(f'Properties in df: {all_df_data[0].keys()}')
    

    return all_df_data

def load_data(filename:str):
    """Loads in dictionary containing pandas DataFrames

    Args:
        filename (str): file path pointing to binary file with snapshot data

    Returns:
        dict: dictionary containing multiple pd.DataFrames
    """
    with open(filename,'rb') as f:
        df = pickle.load(f)
    logger.debug('Loading snapshot dataframes...')
    return df

def extract_meta_from_log(path_to_log):
    """
    Extracts the impact velocity and timestep from the log file.
    """
    with open(path_to_log) as f:
        for line in f:
            if line.startswith('velocity	    all set NULL NULL') and '$' not in line: 
                velocity = float(line.split()[5])
                break

        for line in f:
            if len(line.split())>1 and 'timestep' == line.split()[0]:
                timestep = float(line.split()[1])
                break

    return velocity, timestep*1000

# from a list of properties, extract the file path to the dump file fitting the requested conditions
def get_file_list(vel_list,polytype_list,vac_frac_list,parent_dir='/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/vacancy_shock'):
    fs = []
    missing_data = {}
    if vel_list == 'all':
        vel_list = [-5.0,-10.0,-20.0,-25.0,-30.0,-40.0]
    if polytype_list == 'all':
        polytype_list = ['B11-Ce-CBC','B12-CBC','B12-CCC','B11-Cp-CBC']
    if vac_frac_list == 'all':
        vac_frac_list = [0.0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]

    for polytype in polytype_list:
        for vel in vel_list:
            for vac_frac in vac_frac_list:
                path_to_dir = f'{parent_dir}/{polytype}/{vac_frac}/shock_{vel}_data'
                if os.path.isdir(path_to_dir):
                    fs.append(path_to_dir)
                else:
                    if polytype not in missing_data.keys():
                        missing_data[polytype] = {}
                    if vac_frac not in missing_data[polytype].keys():
                        missing_data[polytype][vac_frac] = []
                    
                    missing_data[polytype][vac_frac].append(vel)
                    continue

    #print missing vels to logger
    for polytype in missing_data.keys():
        logger.warning(f'Missing data for {polytype}: ')
        for vac_frac in missing_data[polytype].keys():
            logger.warning(f'   Vacancy fraction: {vac_frac}')
            for vel in missing_data[polytype][vac_frac]:
                logger.warning(f'       {vel}')
    return fs        


# Compute properties from data 
#--------------------------------------------------
#extract conditions within a specific bin 
def bin_conditions(labeled_df:dict,calc_bin:int,v_imp:float,timestep=None):

    if timestep:
        time_list = [timestep]
        print(time_list)
    else:
        time_list = list(labeled_df.keys())

    
    #initialize time-series property list
    shear = []
    sigx = []
    sigz = []
    sigy = []
    tauxy=[]
    tauxz=[]
    tauyz=[]
    sigvm = []
    temperature = []
    v = []
    hyd_press = []
    bin_id = []
    z = []
    for t in time_list:
        #average dict by 'bin_id'
        df = labeled_df[t].sort_values(by='bin_id')
        df_temp = df.copy()
        df['c_p[1]']= df_temp['c_p[1]']/df_temp['c_vor[1]']
        df['c_p[2]']= df_temp['c_p[2]']/df_temp['c_vor[1]']
        df['c_p[3]']= df_temp['c_p[3]']/df_temp['c_vor[1]']
        df['c_p[4]']= df_temp['c_p[4]']/df_temp['c_vor[1]']
        df['c_p[5]']= df_temp['c_p[5]']/df_temp['c_vor[1]']
        df['c_p[6]']= df_temp['c_p[6]'] /df_temp['c_vor[1]']
        

        v2 = (df_temp['vx']**2 + df_temp['vy']**2 + (df_temp['vz']-v_imp)**2)*10000
        
        #mass in kg
        m = df_temp['mass']/(6.022e26)

        #kinetic energy in J adjusted for rigid body motion
        df['ke'] = 0.5 * m * v2
        df['T'] = (2*df['ke'])/(3*1.380649e-23)

        avg_df = df.groupby('bin_id').mean().reset_index()

        num_bins = len(avg_df)
        if 'c_vor[1]' in avg_df.keys():
            logger.debug('Using voronoi volume')
            if t == 0:
                #take average of 
                v0 = avg_df['c_vor[1]'][int(num_bins/3):int(num_bins*2/3)].mean()*1.02
                logger.debug(f'Reference volume = {v0}')
                # plot_3d_scatter_with_color(df)

            volume = avg_df['c_vor[1]'][calc_bin]
        else:
            logger.debug('Did not find voronoi volume in dump properties')
            mins = df.groupby('bin_id').min()   
            maxes = df.groupby('bin_id').max()
            n_atoms = df.groupby('bin_id').count()['id']

            lx = maxes['x']-mins['x']
            ly = maxes['y']-mins['y']
            lz = maxes['z']-mins['z']

            volume = lx*ly*lz/n_atoms
            if t == 0:
                v0 = volume
                
        v.append(volume/v0)
        
        
        sx = avg_df['c_p[1]']*-0.0001
        sigx.append(sx[calc_bin])
        
        sy = avg_df['c_p[2]']*-0.0001
        sigy.append(sy[calc_bin])
        
        sz = avg_df['c_p[3]']*-0.0001
        sigz.append(sz[calc_bin])
        
        txy = avg_df['c_p[4]']*-0.0001 
        tauxy.append(txy[calc_bin])
        
        txz = avg_df['c_p[5]']*-0.0001 
        tauxz.append(txz[calc_bin])
        
        tyz = avg_df['c_p[6]']*-0.0001 
        tauyz.append(tyz[calc_bin])
        
        temperature.append(df['T'][calc_bin])
        
        #vonmises stress
        svm = np.sqrt(0.5*((sx-sy)**2+(sy-sx)**2+(sz-sx)**2)+3*(tyz**2 + txz**2 + txy**2))
        sigvm.append(svm[calc_bin])
        #shear stress
        shr = 0.5*(sz - 0.5 * (sx + sy))
        shear.append(shr[calc_bin])


        #hydrostatic pressure
        press = (sx + sy + sz) / 3
        hyd_press.append(press[calc_bin])
        bin_id.append(calc_bin)

        z.append(avg_df['z'][calc_bin])

    dict = {
        'timestep': time_list,
        'z': z,
        'Hydrostatic Pressure': hyd_press,
        'sig x': sigx,
        'sig y': sigy,
        'sig z': sigz,
        'txy': tauxy,
        'txz': tauxz,
        'tyz': tauyz,
        'shear stress': shear,
        'von mises stress': sigvm,
        'volumetric strain': v,
        'temperature': temperature, 
        'bin_id': bin_id           
    }

    return dict

def get_large_dataframe(sim_df):
    new_df = pd.DataFrame()
    if type(sim_df) == dict:
        sim_list = sim_df['SimDump']
    else:
        sim_list = sim_df
        
    for sim in sim_list:
        df = sim.get_data().reset_index()
        # sim.get_wave_velocities(thresh=thresh,weights=weights,plot=plot)
        df['polytype'] = sim.polytype
        df['angle'] = sim.angle
        df['velocity'] = sim.velocity
        df['removal'] = sim.removal_type
        df['vac_frac'] = sim.vac_frac        
        # df['p_slope'] = sim.p_slope
        # df['e_slope'] = sim.e_slope
        new_df = pd.concat([new_df,df])
        
    return new_df

# define dump object
class SimulationDump:
    """
    Class to handle a single simulation dump file. Contains methods to extract and process data from LAMMPS dump file.
    """
    def __init__(self, dump_file:str, log_file:str=None,num_bins=100,timestep=0.2):
        """
        Initialize the SimulationDump object with the path to a LAMMPS dump file and log file.
            Attributes:
                dump_file (str): path to LAMMPS dump file
                log_file (str): path to LAMMPS log file
                dump_list (list): list of LammpsDump objects
                velocity (float): impact velocity of the simulation
                ts (float): LAMMPS timestep in fs
                polytype (str): polytype of the simulation
                angle (float): angle of the simulation
                removal_type (str): removal type of the simulation
                vac_frac (float): vacancy fraction of the simulation
                num_bins (int): number of bins to divide the data into (defaults to 100)
        """
        self.dump_file = dump_file
        self.log_file = log_file
        if log_file:
            try:
                self.velocity, self.ts = extract_meta_from_log(log_file)
            except Exception as e:
                logger.warning(f"Error extracting meta from log file: {e}")
                self.velocity = float(dump_file.split('_')[-1])
                self.ts = timestep
        else:
            try:
                dump_dir = dump_file.split('dump.data_shock_chunk')[0]
                log_file = glob(os.path.join(dump_dir,'log.*'))[0]
                self.velocity, self.ts = extract_meta_from_log(log_file)
            except Exception as e:
                logger.info('No log file found')
                self.velocity = float(dump_file.split('_')[-1])
                self.ts = timestep

        self.polytype = dump_file.split('/')[-6]
        self.angle = float(dump_file.split('/')[-5])
        self.removal_type = dump_file.split('/')[-4]
        self.vac_frac = float(dump_file.split('/')[-3])
        self.num_bins = num_bins
        

    def __str__(self):
        string = f'Polytype: {self.polytype}\n ------------------- \n Angle: {self.angle}\n Removal Type: {self.removal_type}\n Vacancy Fraction: {self.vac_frac}\n Velocity: {self.velocity}\n Number of bins: {self.num_bins}'
        string = 'SimDump Object'
        return string
        
    @lru_cache(None)
    def get_bin_data(self,calc_bin=8):
        """
            Extracts the data from a specific bin in the simulation dump file.
            Args:
                calc_bin (int): bin number to extract data from
            Returns:
                pd.DataFrame: Pandas DataFrame containing the data from the specified bin
        """
        data = pd.DataFrame(bin_conditions(self.data,v_imp=self.velocity,calc_bin=calc_bin))
        return data

    @property
    def summary(self):
        """
            Prints summary information about SimulationDump object.
        """
        print(f'Polytype: {self.polytype}\n ------------------- \n Angle: {self.angle}\n Removal Type: {self.removal_type}\n Vacancy Fraction: {self.vac_frac}\n Velocity: {self.velocity}\n Number of bins: {self.num_bins}')

    @property
    @lru_cache(None)
    def dump_list(self):
        """
            Reads the dump file and returns a list of LammpsDump objects.
            Returns:
                list: list of LammpsDump objects
        """
        try:
            dump_list = [i for i in parse_lammps_dumps(self.dump_file)]
            return dump_list
        except Exception as e:
            logger.error(f"Error reading dump file: {e}")
            print(e)
            logger.error(f"Path to dump file: {self.dump_file}")
            return None

    @property
    @lru_cache(None)
    def data(self):
        """
            Extracts the data from the simulation dump file and bins it.
            Returns:
                dict: nested dictionary containing a pandas DataFrame for each snapshot in dump_list. The snapshots are keys.
        """
        return get_dfs(self.dump_list,num_bins=self.num_bins,ts=self.ts)

    @lru_cache(None)
    def get_data(self,save=True):
        """
            Extracts the data from the simulation dump file and bins it.
            Returns:
                pd.DataFrame: Multi-index Pandas DataFrame containing the data from all bins in the simulation for all timesteps.
                    - Indices: bin_id, timestep
        """
        #check to see if processed data exists in file path of object
        dir_path = self.dump_file.split(self.dump_file.split('/')[-1])[0]
        logger.info(f'Looking for processed data in: {dir_path}')
        logger.info('------------------------------')
        #check if data has been processed in the past
    
        if os.path.isfile(os.path.join(dir_path,'processed_data.pkl')):
            with open(os.path.join(dir_path,'processed_data.pkl'),'rb') as f:
                profile_data = pickle.load(f)
                logger.info(f'Processed data found in {dir_path}')
            return profile_data
        
        logger.info('Did not find processed data, processing data...')
        #if no processed data exists, process the data
        for bin_id in range(self.num_bins):
            if bin_id == 0:
                profile_data = self.get_bin_data(bin_id)
            else:
                profile_data = pd.concat([profile_data,self.get_bin_data(bin_id)])
        logger.info('Data processed') 
        
        # set multi-index
        profile_data = profile_data.set_index(['bin_id','timestep'])
        
        # save processed data
        if save:
            with open(os.path.join(dir_path,'processed_data.pkl'),'wb') as f:
                pickle.dump(profile_data,f)
            logger.info(f'Data saved to {dir_path}')
        else:
            logger.info('Data not saved')
            
        return profile_data

    @lru_cache(None)
    def get_hugoniot_state(self,bin1=None,bin2=None,t1=None,t2=None):
        if not bin1:
            bin1 = int(self.num_bins/3)
        if not bin2:
            bin2 = int(self.num_bins*2/3)
        if not t1:
            t1 = int(self.get_data().reset_index()['timestep'].max()/3)
        if not t2:
            t2 = int(self.get_data().reset_index()['timestep'].max()*2/3)
        idx = pd.IndexSlice

        #get average values of the data in hugoniot states
        df_hug = self.get_data().loc[idx[bin1:bin2,t1:t2],:]
        avg_df = df_hug.mean()
        avg_df['velocity']=self.velocity/10
        avg_df['polytype']=self.polytype
        avg_df['angle']=self.angle
        
        self.hug = avg_df
        
        return avg_df
        
        
    def plot_time_data(self, x, y, filename=None, existing_fig=None, layout_updates=None):
        dump_df = self.get_data()
        time_data = dump_df.reorder_levels(['timestep', 'bin_id'])
        avg_df = time_data.groupby(['timestep', 'bin_id']).mean()

        idx = pd.IndexSlice

        fig = px.line(
            avg_df.reset_index(),
            x=x,
            y=y,
            animation_frame='timestep',
            range_y=[
                min(avg_df.loc[idx[:, 3:95], y])*0.9,
                max(avg_df.loc[idx[:, 3:95], y]*1.1)
            ],
            range_x=[
                min(avg_df.loc[idx[:, :], x])*0.9,
                max(avg_df.loc[idx[:, :], x])*1.1
            ],
            line_shape='linear'  # Add this line to remove the smoothing effect
        )

        if existing_fig:
            for trace in existing_fig.data:
                fig.add_trace(trace)


        # set title and labels
        fig.update_layout(
            title=f'{self.polytype} - {self.angle}º - {self.removal_type} - {self.vac_frac} - {self.velocity/10} km/s',
            xaxis_title=x,
            yaxis_title=y
        )

        if layout_updates:
            fig.update_layout(**layout_updates)

        if filename:
            fig.write_html(filename)
            logger.info(f'Plot saved to: {filename}')

        return fig

    def plot_bin_data(self,x,y,calc_bin=8,color='black',filename=None,existing_fig=None,layout_updates=None):
        idx = pd.IndexSlice
        dump_df = self.get_data().reset_index().set_index('bin_id')
        avg_df = dump_df.loc[idx[calc_bin]]
        
        # avg_df[colorby] = self.angle
        
        fig = px.line(
            avg_df,
            x=x,
            y=y,
            range_y=[
                min(avg_df.loc[idx[3:95],y])*0.9,
                max(avg_df.loc[idx[3:95],y])*1.1
                ],
            range_x=[
                min(avg_df.loc[idx[:],x])*0.9,
                max(avg_df.loc[idx[:],x])*1.1
                ]
            )

        fig.update_traces(line_color=color, line_width=3)
        
        if existing_fig:
            for trace in existing_fig.data:
                fig.add_trace(trace)
            
        #set title and labels
        fig.update_layout(academic_template['layout'])
        
        fig.update_layout(
            title=f'{self.polytype} - {self.angle}º - {self.removal_type} - {self.vac_frac} - {self.velocity/10} km/s',
            xaxis_title=x,
            yaxis_title=y,
            showlegend=True
        )

        if layout_updates:
            fig.update_layout(**layout_updates)
            
        if filename:
            fig.write_html(filename)
            logger.info(f'Plot saved to: {filename}')
        
        return fig
        

# %%
