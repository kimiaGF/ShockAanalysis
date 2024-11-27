#%%
"""
This module implements classes and methods for processing LAMMPS output
files (log and dump).
"""

from __future__ import annotations

import re
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING

import time
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from monty.io import zopen
from monty.json import MSONable

from pymatgen.io.lammps.data import LammpsBox

if TYPE_CHECKING:
    from typing import Any

__author__ = "Kiran Mathew, Zhi Deng"
__copyright__ = "Copyright 2018, The Materials Virtual Lab"
__version__ = "1.0"
__maintainer__ = "Zhi Deng"
__email__ = "z4deng@eng.ucsd.edu"
__date__ = "Aug 1, 2018"

#%%

class LammpsDump(MSONable):
    """Object for representing dump data for a single snapshot."""

    def __init__(self, timestep: int, natoms: int, box: LammpsBox, data: pd.DataFrame) -> None:
        """
        Base constructor.

        Args:
            timestep (int): Current time step.
            natoms (int): Total number of atoms in the box.
            box (LammpsBox): Simulation box.
            data (pd.DataFrame): Dumped atomic data.
        """
        self.timestep = timestep
        self.natoms = natoms
        self.box = box
        self.data = data

    @classmethod
    def from_str(cls, string: str) -> LammpsDump:
        """
        Constructor from string parsing.

        Args:
            string (str): Input string.
        """
        lines = string.split("\n")
        time_step = int(lines[1])
        n_atoms = int(lines[3])
        box_arr = np.loadtxt(StringIO("\n".join(lines[5:8])))
        bounds = box_arr[:, :2]
        tilt = None
        if "xy xz yz" in lines[4]:
            tilt = box_arr[:, 2]
            x = (0, tilt[0], tilt[1], tilt[0] + tilt[1])
            y = (0, tilt[2])
            bounds -= np.array([[min(x), max(x)], [min(y), max(y)], [0, 0]])
        box = LammpsBox(bounds, tilt)
        
        data_head = lines[8].replace("ITEM: ATOMS", "").split()
        #print how long it takes to read the data
        start = time.time()
        
        data = cudf.read_csv(StringIO("\n".join(lines[9:])), names=data_head, delim_whitespace=True)
        time_diff = time.time()-start
        print(f"Time to read data: {time_diff:.2f} s")
        
        return cls(time_step, n_atoms, box, data)

    @classmethod
    def from_dict(cls, dct: dict) -> LammpsDump:
        """
        Args:
            d (dict): Dict representation.

        Returns:
            LammpsDump
        """
        items = {"timestep": dct["timestep"], "natoms": dct["natoms"]}
        items["box"] = LammpsBox.from_dict(dct["box"])
        items["data"] = pd.read_json(dct["data"], orient="split")
        return cls(**items)

    def as_dict(self) -> dict[str, Any]:
        """Returns: MSONable dict."""
        dct: dict[str, Any] = {}
        dct["@module"] = type(self).__module__
        dct["@class"] = type(self).__name__
        dct["timestep"] = self.timestep
        dct["natoms"] = self.natoms
        dct["box"] = self.box.as_dict()
        dct["data"] = self.data.to_json(orient="split")
        return dct


def parse_lammps_dumps(file_pattern):
    """
    Generator that parses dump file(s).

    Args:
        file_pattern (str): Filename to parse. The timestep wildcard
            (e.g., dump.atom.'*') is supported and the files are parsed
            in the sequence of timestep.

    Yields:
        LammpsDump for each available snapshot.
    """
    files = glob(file_pattern)
    if len(files) > 1:
        pattern = file_pattern.replace("*", "([0-9]+)").replace("\\", "\\\\")
        files = sorted(files, key=lambda f: int(re.match(pattern, f).group(1)))

    for filename in files:
        with zopen(filename, mode="rt") as file:
            dump_cache = []
            
            for line in file:
                # encountering a new timestep within the same file 
                if line.startswith("ITEM: TIMESTEP"):
                    if len(dump_cache) > 0:
                        yield LammpsDump.from_str("".join(dump_cache))
                    dump_cache = [line]
                else:
                    dump_cache.append(line)
            yield LammpsDump.from_str("".join(dump_cache))

def load_file_to_gpu(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Move contents to GPU
    data_gpu = cp.asarray(bytearray(file_content, 'utf-8'))
    return data_gpu

def find_timestep_indices(data_gpu,keyword='ITEM: TIMESTEP'):
    keyword_gpu = cp.asarray(bytearray(keyword, 'utf-8'))
    data_len = len(data_gpu)
    keyword_len = len(keyword_gpu)
    
    option = 1
    # Find the indices of the keyword
    ## Option 1:
    if option == 1:
        timestep_indices = cp.where(cp.all(data_gpu[:-keyword_len+1] == keyword_gpu[0], axis=0))
    
    ## Option 2:
    if option == 2:
        def match_kernel(data, keyword, data_len, keyword_len):
            results = cp.zeros(data_len,dtype=cp.int32)
            for i in range(data_len-keyword_len):
                if cp.all(data[i:i+keyword_len] == keyword):
                    results[i] = 1 #position where keyword starts
            return results
        
        indices = match_kernel(data_gpu, keyword_gpu, data_len, keyword_len)
        timestep_indices = cp.where(indices == 1)[0]
    
    return timestep_indices

def process_lammps_file(file_path):
    data_gpu = load_file_to_gpu(file_path)
    timestep_indices = find_timestep_indices(data_gpu)
    chunks = cp.split(data_gpu, timestep_indices)
    
    for chunk in chunks:
        yield LammpsDump.from_str(chunks)
        
    
    
    
    
    
def parse_lammps_log(filename: str = "log.lammps") -> list[pd.DataFrame]:
    """
    Parses log file with focus on thermo data. Both one and multi line
    formats are supported. Any incomplete runs (no "Loop time" marker)
    will not be parsed.

    Notes:
        SHAKE stats printed with thermo data are not supported yet.
        They are ignored in multi line format, while they may cause
        issues with dataframe parsing in one line format.

    Args:
        filename (str): Filename to parse.

    Returns:
        [pd.DataFrame] containing thermo data for each completed run.
    """
    with zopen(filename, mode="rt") as file:
        lines = file.readlines()
    begin_flag = (
        "Memory usage per processor =",
        "Per MPI rank memory allocation (min/avg/max) =",
    )
    end_flag = "Loop time of"
    begins, ends = [], []
    for idx, line in enumerate(lines):
        if line.startswith(begin_flag):
            begins.append(idx)
        elif line.startswith(end_flag):
            ends.append(idx)

    def _parse_thermo(lines: list[str]) -> pd.DataFrame:
        multi_pattern = r"-+\s+Step\s+([0-9]+)\s+-+"
        # multi line thermo data
        if re.match(multi_pattern, lines[0]):
            timestep_marks = [idx for idx, line in enumerate(lines) if re.match(multi_pattern, line)]
            timesteps = np.split(lines, timestep_marks)[1:]
            dicts = []
            kv_pattern = r"([0-9A-Za-z_\[\]]+)\s+=\s+([0-9eE\.+-]+)"
            for ts in timesteps:
                data = {}
                step = re.match(multi_pattern, ts[0])
                assert step is not None
                data["Step"] = int(step.group(1))
                data.update({k: float(v) for k, v in re.findall(kv_pattern, "".join(ts[1:]))})
                dicts.append(data)
            df = pd.DataFrame(dicts)
            # rearrange the sequence of columns
            columns = ["Step"] + [k for k, v in re.findall(kv_pattern, "".join(timesteps[0][1:]))]
            df = df[columns]
        # one line thermo data
        else:
            df = pd.read_csv(StringIO("".join(lines)), delim_whitespace=True)
        return df

    runs = []
    for b, e in zip(begins, ends):
        runs.append(_parse_thermo(lines[b + 1 : e]))
    return runs

# %%
