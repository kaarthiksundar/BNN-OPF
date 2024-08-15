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
from classes import BranchAdmittanceMatrix, Component, BusTypeIdx, GenCostCoeff, Data, Limits, OPFData
import logging
import jax.numpy as jnp
import tqdm
import pathlib
import json

def load_data(
    data_folder: str, case: str, 
    log: logging.Logger, 
    num_train: int, num_test: int):
    case_file = data_folder + case + '.m'
    case_data = matpower_parser.create_model_data_dict(case_file)
    log.info('Parsed case file')
    Y = construct_admittance_matrix(case_data)
    buses, branches = Y[0], Y[1]
    y_bus, y_branch = Y[2], Y[3]
    bus_to_idx, idx_to_bus = Y[4], Y[5]
    branch_to_idx, idx_to_branch = Y[6], Y[7]
    gen_info = get_generator_info(case_data)
    log.info('Admittance matrices created')
    gens = gen_info[0] 
    gen_to_idx, idx_to_gen = gen_info[1], gen_info[2], 
    p_min, p_max = gen_info[3], gen_info[4]
    q_min, q_max = gen_info[5], gen_info[6]
    q_cost_coeff = gen_info[7] 
    l_cost_coeff = gen_info[8]
    c_cost_coeff = gen_info[9]
    log.info('Generator info parsed')
    ref_bus_idx = np.array(list(
        filter(lambda x: buses[x][1]['matpower_bustype'] == 'ref', range(len(buses)))
    ))
    pv_bus_idx = np.array(list(
        filter(lambda x: buses[x][1]['matpower_bustype'] == 'PV', range(len(buses)))
    ))
    pq_bus_idx = np.setdiff1d(
        range(len(buses)), np.concatenate([ref_bus_idx, pv_bus_idx])
    )
    non_ref_bus_idx = np.sort(np.concatenate([pq_bus_idx, pv_bus_idx]))
    v_min = jnp.array([bus['v_min'] for _, bus in buses])
    v_max = jnp.array([bus['v_max'] for _, bus in buses])
    va_ref = jnp.array([bus[1]['va'] for (i, bus) in enumerate(buses) if i in ref_bus_idx]).squeeze(-1)
    log.info('Bus info parsed')
    
    log.info('Reading training and testing data sets')
    dataset = OPFDataset(data_folder, case_name = case)
    sample_info = get_samples(dataset, gens, buses, case_data, num_train, num_test)
    loads = sample_info[0] 
    load_to_idx, idx_to_load = sample_info[1], sample_info[2] 
    demand_train, gen_train, v_train, obj_train = sample_info[3]
    demand_test, gen_test, v_test, obj_test = sample_info[4]
    log.info('ML data set created')
    
    # create components 
    bus_components = Component(buses, bus_to_idx, idx_to_bus)
    branch_components = Component(branches, branch_to_idx, idx_to_branch)
    gen_components = Component(gens, gen_to_idx, idx_to_gen)
    load_components = Component(loads, load_to_idx, idx_to_load)  
    
    # create bus type ids and generator cost coefficients
    bus_type_idx = BusTypeIdx(ref_bus_idx, non_ref_bus_idx, pv_bus_idx, pq_bus_idx)
    gen_cost = GenCostCoeff(q_cost_coeff, l_cost_coeff, c_cost_coeff)
    
    # training and testing dataset 
    train = Data(demand_train, gen_train, v_train, obj_train) 
    test = Data(demand_test, gen_test, v_test, obj_test)
    
    return OPFData(
        case_name = case,
        case_data = case_data, 
        buses = bus_components,
        branches = branch_components, 
        gens = gen_components, 
        loads = load_components, 
        y_bus = y_bus,
        y_branch = y_branch, 
        bus_type_idx = bus_type_idx, 
        gen_cost = gen_cost, 
        pg_bounds = Limits(lower = p_min, upper = p_max),
        qg_bounds = Limits(lower = q_min, upper = q_max),
        vm_bounds = Limits(lower = v_min, upper = v_max), 
        va_ref = va_ref,
        train = train,  
        test = test,
    )
    

# does not support DC lines and switches (ignore them for now)
def construct_admittance_matrix(data: dict):
    mva_base = data['elements']['generator']['1']['mbase']
    buses = [ (key, val) for key, val in data['elements']['bus'].items() ]
    buses.sort(key = lambda x: int(x[0]))
    
    branches = [ (key, val) for key, val in data['elements']['branch'].items() 
                if val['in_service'] == True]
    branches.sort(key = lambda x: int(x[0]))

    idx_to_bus = [key for (key, _) in buses]
    bus_to_idx = { x[0] : i for (i, x) in enumerate(buses) }
    
    idx_to_branch = [key for (key, _) in branches]
    branch_to_idx = { x[0] : i for (i, x) in enumerate(branches) }
    
    branch_matrix_list = []

    I = []
    J = []
    V = []

    for (i, branch) in branches:
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
        
        I_branch = [] 
        J_branch = [] 
        V_branch = [] 
        bus_pair = (from_bus, to_bus)
        idx_pair = (f_bus, t_bus)
        
        
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
        I_branch.append(0)
        J_branch.append(1)
        V_branch.append(-y/np.conjugate(t))

        I.append(t_bus)
        J.append(f_bus)
        V.append(-(y/t))
        I_branch.append(1)
        J_branch.append(0)
        V_branch.append(-(y/t))
        
        I.append(f_bus)
        J.append(f_bus)
        V.append((y + lc_fr)/(t.real**2 + t.imag**2))
        I_branch.append(0)
        J_branch.append(0)
        V_branch.append((y + lc_fr)/(t.real**2 + t.imag**2))
        
        I.append(t_bus)
        J.append(t_bus)
        V.append(y + lc_to)  
        I_branch.append(1)
        J_branch.append(1)
        V_branch.append(y + lc_to)     
        
        val = np.array(V_branch, dtype=complex)
        row = np.array(I_branch, dtype=int)
        col = np.array(J_branch, dtype=int)
        branch_matrix_list.append(BranchAdmittanceMatrix(
            admittance_matrix = csr_matrix((val, (row, col)), shape=(2, 2)),
            bus = bus_pair, 
            idx = idx_pair,
            thermal_limit = branch['rating_long_term']/mva_base)
        )
        
    y_branch = branch_matrix_list

    for (i, shunt) in data['elements']['shunt'].items():
        shunt_bus = shunt['bus']
        bus = bus_to_idx[shunt_bus]
        ys = shunt['gs'] / mva_base + shunt['bs'] / mva_base * 1j
        I.append(bus)
        J.append(bus)
        V.append(ys)
    
    val = np.array(V, dtype=complex)
    row = np.array(I, dtype=int)
    col = np.array(J, dtype=int)
    shape = (len(buses), len(buses))
    y_bus = csr_matrix((val, (row, col)), shape=shape)
    
    return (buses, branches, y_bus, y_branch, bus_to_idx, idx_to_bus, branch_to_idx, idx_to_branch)

# create generator info
def get_generator_info(data: dict): 
    mva_base = data['elements']['generator']['1']['mbase']
    gens = [ (key, val) for key, val in data['elements']['generator'].items() ]
    gens.sort(key = lambda x: int(x[0]))
    idx_to_gen = [key for (key, _) in gens]
    gen_to_idx = { x[0] : i for (i, x) in enumerate(gens) }
    
    p_max = jnp.array([gen['p_max']/mva_base for _, gen in gens])
    p_min = jnp.array([gen['p_min']/mva_base for _, gen in gens])
    q_max = jnp.array([gen['q_max']/mva_base for _, gen in gens])
    q_min = jnp.array([gen['q_min']/mva_base for _, gen in gens])
    q_coeff = jnp.array([gen['p_cost']['values'][2] * mva_base**2 for _, gen in gens])
    l_coeff = jnp.array([gen['p_cost']['values'][1] * mva_base for _, gen in gens])
    c_coeff = jnp.array([gen['p_cost']['values'][0] for _, gen in gens])
    return gens, gen_to_idx, idx_to_gen, p_min, p_max, q_min, q_max, q_coeff, l_coeff, c_coeff


# split into training and testing data and return data 
def get_samples(dataset: OPFDataset, 
                 gens: list, 
                 buses: list, 
                 case_data: dict, 
                 num_train: int, 
                 num_test: int): 
    
    num_groups = 20
    loads = [ (key, val) for key, val in case_data['elements']['load'].items() if val['in_service'] == True ]
    idx_to_load = [ key for (key, _) in loads ]
    load_to_idx = { x[0] : i for (i, x) in enumerate(loads) }
    demand_train = np.zeros((num_train * num_groups, len(loads)), dtype=complex)
    demand_test = np.zeros((num_test * num_groups, len(loads)), dtype=complex) 
    gen_train = np.zeros((num_train * num_groups, len(gens)), dtype=complex) 
    gen_test = np.zeros((num_test * num_groups, len(gens)), dtype=complex) 
    v_train = np.zeros((num_train * num_groups, len(buses)), dtype=complex) 
    v_test = np.zeros((num_test * num_groups, len(buses)), dtype=complex)
    obj_train = np.zeros((num_train * num_groups), dtype=float)
    obj_test = np.zeros((num_test * num_groups), dtype=float)
    
    for group in tqdm.tqdm(range(num_groups)): 
        tmp_dir = pathlib.PurePath(
            dataset.raw_dir,
            'gridopt-dataset-tmp',
            dataset._release,
            dataset.case_name,
            f'group_{group}')
        files = sorted(pathlib.Path(tmp_dir).glob('*.json'))[:num_train+num_test]
        train_files = files[:num_train]
        test_files = files[num_train:(num_train+num_test)]
        for (i, name) in enumerate(train_files):
            j = i + group * num_train
            with (open(name)) as f: 
                obj = json.load(f)
            grid = obj['grid']
            solution = obj['solution']
            demand_train[j, :] = [d[0] + d[1] * 1j for d in grid['nodes']['load']]
            gen_train[j, :] = [g[0] + g[1] * 1j for g in solution['nodes']['generator']]
            v_train[j, :] = [v[1] * math.cos(v[0]) + v[1] * math.sin(v[0]) * 1j for v in solution['nodes']['bus']]
            obj_train[j] = obj['metadata']['objective']
        for (i, name) in enumerate(test_files): 
            j = i + group * num_test
            with (open(name)) as f:
                obj = json.load(f)
            grid = obj['grid']
            solution = obj['solution']
            demand_test[j, :] = [d[0] + d[1] * 1j for d in grid['nodes']['load']]
            gen_test[j, :] = [g[0] + g[1] * 1j for g in solution['nodes']['generator']]
            v_test[j, :] = [v[1] * math.cos(v[0]) + v[1] * math.sin(v[0]) * 1j for v in solution['nodes']['bus']]  
            obj_test[j] = obj['metadata']['objective']
            
    train = (demand_train, gen_train, v_train, obj_train)
    test = (demand_test, gen_test, v_test, obj_test)
            
    return loads, load_to_idx, idx_to_load, train, test