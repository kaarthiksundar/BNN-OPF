""" Definition of all the classes """ 

from typing import Literal, List, Dict, Tuple
from dataclasses import dataclass
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

@dataclass 
class SampleCounts:
    num_groups: int 
    num_train_per_group: int 
    num_test_per_group: int 
    num_unsupervised_per_group: int 
    batch_size: int
 
@dataclass
class BranchAdmittanceMatrix: 
    admittance_matrix: jax.Array 
    bus: Tuple[str, str]
    idx: Tuple[int, int]
    thermal_limit: float 

@dataclass 
class Component: 
    components: List 
    component_to_idx: Dict
    idx_to_component: List 

@dataclass 
class Limits: 
    lower: jax.Array 
    upper: jax.Array
    
@dataclass 
class GenCostCoeff:
    q: jax.Array 
    l: jax.Array 
    c: jax.Array 
    
@dataclass 
class BusTypeIdx: 
    ref: np.ndarray
    non_ref: np.ndarray 
    pv: np.ndarray 
    pq: np.ndarray

@dataclass 
class Data: 
    demand: np.ndarray 
    generation: np.ndarray
    voltage: np.ndarray 
    objective: np.ndarray
    
@dataclass 
class UnsupervisedData: 
    demand: np.ndarray

# input data arranged as [pd, qd] 
def get_X(data: type[Data | UnsupervisedData]) -> jax.Array: 
    return jnp.array(np.concatenate([
        np.real(data.demand),
        np.imag(data.demand)
        ], axis=1))
    
# output data  arranged as [pg, qg, vm, va]
def get_Y(data: Data) -> jax.Array:
    return jnp.array(np.concatenate([
        np.real(data.generation), np.imag(data.generation), 
        np.abs(data.voltage), np.angle(data.voltage)
        ], axis=1))
    
# normalization values for data 
def get_mean(val: jax.Array) -> jax.Array: 
    return jnp.mean(val, axis=0)
def get_std(val: jax.Array) -> jax.Array:
    return jnp.std(val, axis=0) + 1e-6

class OPFData():
    r"""The main class that holds all the network, training and testing data

    :class:`OPFData` holds the network data, training and testing data for the 
    Bayesian Neural Networks

    Args:
        case_name (str): The name of the original pglib-opf case.
        case_data (dict): The EGRET case data parsed from the .m file 
        buses (Component): buses with their id maps 
        branches (Component): branches with their id maps
        gens (Component): generators with their id maps
        loads (Component): load with their id maps
        y_bus (csr_matrix): scipy sparse matrix for the bus admittance matrix
        y_branch (List[BranchAdmittanceMatrix]): list of 2x2 branch admittance matrices with their thermal limits 
        bus_type_idx (BusTypeIdx): ref, non_ref, pv and pq bus ids 
        gen_cost (GenCostCoeff): cost coefficient for the generators
        pg_bounds (Limits): real power generation bounds 
        qg_bounds (Limits): reactive power generation bounds 
        vm_bounds (Limits): voltage magnitude bounds
        va_ref (jax.Array): ref bus voltage angles (this are fixed)
        train: Data, 
        test: Data, 
        unsupervised: UnsupervisedData
        batch_size: int
    """
    def __init__(
        self, 
        case_name: Literal[
            'pglib_opf_case14_ieee',
            'pglib_opf_case30_ieee',
            'pglib_opf_case57_ieee',
            'pglib_opf_case118_ieee',
            'pglib_opf_case500_goc',
            'pglib_opf_case2000_goc',
            'pglib_opf_case6470_rte',
            'pglib_opf_case4661_sdet'
            'pglib_opf_case10000_goc',
            'pglib_opf_case13659_pegase',
        ],
        case_data: Dict, 
        buses: Component, 
        branches: Component, 
        gens: Component, 
        loads: Component,
        y_bus: csr_matrix, 
        y_branch: List[BranchAdmittanceMatrix], 
        bus_type_idx: BusTypeIdx,
        gen_cost: GenCostCoeff,
        pg_bounds: Limits, 
        qg_bounds: Limits, 
        vm_bounds: Limits, 
        va_ref: jax.Array,
        train: Data, 
        test: Data, 
        unsupervised: UnsupervisedData,
        batch_size: int
        ) -> None:
        
        self.case_name = case_name
        self.case_data = case_data
        self.buses = buses
        self.branches = branches 
        self.gens = gens 
        self.loads = loads
        self.gen_bus_idx = [buses.component_to_idx[gen['bus']] for (_, gen) in gens.components] 
        self.load_bus_idx = [buses.component_to_idx[load['bus']] for (_, load) in loads.components]
        self.y_bus = y_bus 
        self.y_branch = y_branch 
        self.bus_type_idx = bus_type_idx 
        self.gen_cost = gen_cost 
        self.pg_bounds = pg_bounds
        self.qg_bounds = qg_bounds 
        self.vm_bounds = vm_bounds 
        self.va_ref = va_ref
        self.train = train 
        self.test = test 
        self.unsupervised = unsupervised
        self.batch_size = batch_size
        self.num_batches = len(range(0, train.shape[0], self.batch_size))
        self.X_train = get_X(self.train)
        self.Y_train = get_Y(self.train)
        self.X_test = get_X(self.test)
        self.Y_test = get_Y(self.test)
        self.X_unsupervised = get_X(self.unsupervised)
        X_data = jnp.concatenate([self.X_train, self.X_unsupervised], axis=0)
        self.X_mean = get_mean(X_data)
        self.X_std = get_std(X_data)
        self.Y_mean = get_mean(self.Y_train)
        self.Y_std = get_std(self.Y_train)
        self.X_train_norm = (self.X_train - self.X_mean) / self.X_std 
        self.X_test_norm = (self.X_test - self.X_mean) / self.X_std 
        self.X_unsupervised_norm = (self.X_unsupervised - self.X_mean) / self.X_std 
        self.Y_train_norm = (self.Y_train - self.Y_mean) / self.Y_std 
        self.Y_test_norm = (self.Y_test - self.Y_mean) / self.Y_std
        
    def get_num_buses(self) -> int:
        return len(self.buses.components)
    
    def get_num_gens(self) -> int: 
        return len(self.gens.components)
    
    def get_num_loads(self) -> int: 
        return len(self.loads.components)
    
    def get_batch(self, i: int, io: str, norm: bool, ru: str) -> jax.Array: 
        """ get the batched data

        Args:
            i (int): batch number (0 <= i < self.num_batches)
            io (str): 'i' means input X, 'o' means output Y
            norm (bool): True means normalized, False means un-normalized
            ru (str): 'r' means train, 'u' means unsupervised. 
            
            it is not possible to have ru = 'u' and io = 'o'

        Returns:
            jax.Array: corresponding jax Array
        """
        assert(i < self.num_batches)
        fr = i 
        to = i + self.batch_size
        if io == 'i': 
            if res == 'r': 
                if norm: 
                    return X_train_norm[fr:to] 
                else: 
                    return X_train[fr:to] 
            else: 
                if norm: 
                    return X_unsupervised_norm[fr:to]
                else:
                    return X_unsupervised[fr:to]
        else: 
            if norm: 
                return Y_train_norm[fr:to]
            else: 
                return Y_train[fr:to]