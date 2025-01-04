#%%
import pandas as pd 
from shock_analysis import SimulationDump
from glob import glob
import logging
import os
import multiprocessing
from tqdm import tqdm
import time 
#%%
#--------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)s:%(message)s')

if os.path.isdir('logs') == False:
    os.mkdir('logs')
    
file_handler = logging.FileHandler('logs/processing_data.log','w+')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
#%%
def file_list(fs):
    files = pd.DataFrame()
    num_processed = 0
    for f in fs:
        dir_path = f.split(f.split('/')[-1])[0]
        if 'processed_data.pkl' in os.listdir(dir_path):
            logger.debug(f'File {f} already processed')
            num_processed += 1
            continue
        
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
        
    return files

# %%
def process_file(file):
    dir_path = file.split(file.split('/')[-1])[0]
    file_name = file.split(file.split('/')[-8])[-1]
    if 'processed_data.pkl' in os.listdir(dir_path):
        logger.debug(f'File {file_name} already processed')
        return
    try:
        logger.info(f'Processing file {file_name}...')
        sim = SimulationDump(file)
        sim.get_data()
        logger.debug(f'Processed file {file_name}')
    except Exception as e:
        print(e)
        logger.error(f'Error in file {file_name} with error {e}')

def analyze_files(fs):
    processed = glob('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/orientation_shock_50-50-1000/B*/*/*/*/shock_*/processed_*')
    logger.info(f'Found {len(fs)} dump files')
    logger.info(f'Found {len(processed)} processed files')

    #find completed runs based on their logfile
    completed = []
    incomplete = []

    for f in fs:
        dir_path = f.split(f.split('/')[-1])[0]
        log_file = glob(f'{dir_path}/log.*')[0]

        with open(log_file) as fl:
            lines = fl.readlines()
            if len(lines) > 0 and 'log             log.B' in lines[-1] or 'Total wall time' in lines[-1]:    
                completed.append(f)
                #create a "DONE" file
                with open(f'{dir_path}/DONE', 'w') as fl:
                    fl.write('DONE')
            else:
                for l in lines[-100:]:
                    if 'Loop time' in l:
                        completed.append(f)
                        #create a "DONE" file
                        with open(f'{dir_path}/DONE', 'w') as fl:
                            fl.write('DONE') 
                        break
                    elif l == lines[-1]:
                        incomplete.append(f)
                        
                

    num_unproc = len(fs) - len(processed)
    logger.info(f'Files to process: {num_unproc}')
    
    
    files = file_list(fs)
    files.set_index(['polytype','angle','removal','vac_frac'],inplace=True)
    logger.info(f'{files}')
    
    for f in tqdm(completed, desc='Processing files'):
        process_file(f)

# %%
if __name__ == '__main__':
    
    fs = glob('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/orientation_shock_50-50-1000/B*/*/*/*/shock_*/dump.data_*')

    processed = glob('/home/kimia.gh/blue2/B4C_ML_Potential/B4C_LAMMPS/DPMD_GPU/orientation_shock_50-50-1000/B*/*/*/*/shock_*/processed_*')
    logger.info(f'Found {len(fs)} dump files')
    logger.info(f'Found {len(processed)} processed files')

    #find completed runs based on their logfile
    completed = []
    incomplete = []

    for f in fs:
        dir_path = f.split(f.split('/')[-1])[0]
        log_file = glob(f'{dir_path}/log.*')[0]

        with open(log_file) as fl:
            lines = fl.readlines()
            if len(lines) > 0 and 'log             log.B' in lines[-1] or 'Total wall time' in lines[-1]:    
                completed.append(f)
                #create a "DONE" file
                with open(f'{dir_path}/DONE', 'w') as fl:
                    fl.write('DONE')
            else:
                for l in lines[-100:]:
                    if 'Loop time' in l:
                        completed.append(f)
                        #create a "DONE" file
                        with open(f'{dir_path}/DONE', 'w') as fl:
                            fl.write('DONE') 
                        break
                    elif l == lines[-1]:
                        incomplete.append(f)
                        
                

    num_unproc = len(fs) - len(processed)
    logger.info(f'Files to process: {num_unproc}')
    
    
    files = file_list(fs)
    files.set_index(['polytype','angle','removal','vac_frac'],inplace=True)
    logger.info(f'{files}')
    
    for f in tqdm(completed, desc='Processing files'):
        process_file(f)


