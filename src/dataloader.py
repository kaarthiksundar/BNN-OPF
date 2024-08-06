""" Loads the Google data set from the paper 
    OPFData: Large-scale datasets for AC optimal
    power flow with topological perturbations
"""
    
import egret.parsers.matpower_parser as matpower_parser
from torch_geometric.datasets import OPFDataset 
from torch_geometric.loader import DataLoader
import numpy as np
import math
from scipy.sparse import csr_matrix

def load_training_data_from_mat(
    data_folder: str, 
    case: str, 
    log
):
    
    case_file = data_folder + case + '.m'
    case_data = matpower_parser.create_model_data_dict(case_file)
    
    train_ds = OPFDataset(
        data_folder, 
        case_name = case, 
        split = 'train' 
    )
    
    # training_loader = DataLoader(train_ds, batch_size = 4, shuffle = True)
    Y = construct_admittance_matrix(case_data)
    print(Y.toarray())
    # bus_dict = case_data['elements']['bus']
    # branch_dict = case_data['elements']['branch']
    # gen_dict = case_data['elements']['generator']
    # bus_list = list(bus_dict.keys())
    # branch_list = list(branch_dict.keys())
    # gen_list = list(gen_dict.keys())   
    

# does not support DC lines and switches (ignore them for now)
def construct_admittance_matrix(data: dict):
    
    mva_base = data['elements']['generator']['1']['mbase']
    buses = [ (key, val) for key, val in data['elements']['bus'].items() ]
    buses.sort(key = lambda x: int(x[0]))

    idx_to_bus = [key for (key, _) in buses]
    bus_to_idx = { x[0] : i for (i, x) in enumerate(buses)}

    I = []
    J = []
    V = []

    for (i, branch) in data['elements']['branch'].items():
        if (branch['in_service'] == False): 
            continue
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']
        if (from_bus not in bus_to_idx): 
            continue 
        if (to_bus not in bus_to_idx):
            continue
        f_bus = bus_to_idx[from_bus]
        t_bus = bus_to_idx[to_bus]
        rs = branch['resistance']
        xs = branch['reactance']
        bs = branch['charging_susceptance']
        y = 1/(rs + xs * 1j)
        lc_fr = bs/2.0 * 1j 
        lc_to = bs/2.0 * 1j 
        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = branch['transformer_phase_shift']
        tr = tau * math.cos(math.radians(shift))
        ti = tau * math.sin(math.radians(shift))
        t = tr + ti * 1j 

        I.append(f_bus)
        J.append(t_bus)
        V.append(-y/np.conjugate(t))

        I.append(t_bus)
        J.append(f_bus)
        V.append(-(y/t))
        
        I.append(f_bus)
        J.append(f_bus)
        V.append((y + lc_fr)/(t.real**2 + t.imag**2))
        
        I.append(t_bus)
        J.append(t_bus)
        V.append(y + lc_to)       

    for (i, shunt) in data['elements']['shunt'].items():
        shunt_bus = shunt['bus']
        bus = bus_to_idx[shunt_bus]
        ys = shunt['gs'] / mva_base + shunt['bs'] / mva_base * 1j
        I.append(bus)
        J.append(bus)
        V.append(ys)
    
    data = np.array(V, dtype=complex)
    row = np.array(I, dtype=int)
    col = np.array(J, dtype=int)
    shape = (len(buses), len(buses))
    return csr_matrix((data, (row, col)), shape=shape)
