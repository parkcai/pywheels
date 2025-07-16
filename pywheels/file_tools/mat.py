import os
import random
import numpy as np
from numpy.typing import NDArray
from typing import List
from scipy.io import loadmat
from scipy.io import savemat
from scipy.io import whosmat
from ..i18n import translate


__all__ = [
    "merge_mat_files",
    "check_mat_part",
    "load_mat_part",
]


def merge_mat_files(
    input_mat_paths: List[str], 
    output_mat_path: str,
)-> None:

    for mat_path in input_mat_paths:
        
        if not os.path.exists(mat_path) or not mat_path.endswith('.mat'):
            
            print(
                translate("无效的路径: %s，请检查该路径下是否存在一个 `.mat` 文件") % (mat_path)
            )
            
            return
        
    merged_data = {}
    
    for mat_path in input_mat_paths:

        data = loadmat(mat_path)
        
        for key, value in data.items():

            if key.startswith('__'):
                continue

            if key not in merged_data:
                merged_data[key] = value
    
    savemat(
        file_name = output_mat_path, 
        mdict = merged_data, 
        do_compression = True,
    )


def check_mat_part(
    mat_path: str, 
    attribute: str,
)-> bool:

    if not os.path.isfile(mat_path): return False
    
    try:
        
        variables = whosmat(mat_path)
        variable_names = [var[0] for var in variables]
        
        if attribute in variable_names:
            return True
        
        return False
    
    except Exception:
        return False


mat_buffer = dict()
mat_buffer_capacity = 1


def load_mat_part(
    mat_path: str, 
    attribute: str,
)-> NDArray:

    if mat_path in mat_buffer:
        
        assert attribute in mat_buffer[mat_path], \
            translate("Error: mat_path %s invalid, or the file doesn't contain attribute %s.")\
            % (mat_path, attribute)
            
        return np.array(mat_buffer[mat_path][attribute])
    
    else:
        
        assert check_mat_part(mat_path, attribute), \
            translate(f"Error: mat_path %s invalid, or the file doesn't contain attribute %s.")\
            % (mat_path, attribute)
            

        mat_data = loadmat(mat_path)

        if len(mat_buffer) == mat_buffer_capacity:
            mat_buffer.pop(random.choice(list(mat_buffer.keys())))
        
        mat_buffer[mat_path] = mat_data
        return np.array(mat_data[attribute])
    