""" Definition of all the classes """ 

from typing import Literal, List, Dict, Tuple
from dataclasses import dataclass
from scipy.sparse import csr_matrix

    
@dataclass
class BranchAdmittanceMatrix: 
    admittance_matrix: csr_matrix 
    bus: Tuple[str, str]
    idx: Tuple[int, int]
    thermal_limit: float 


class OPFData():
    r"""The main class that holds all the network, training and testing data

    :class:`OPFData` holds the network data, training and testing data for the 
    Bayesian Neural Networks


    Args:
        case_name (str): The name of the original pglib-opf case.
        case_data (dict): The EGRET case data parsed from the .m file 
        y_bus (csr_matrix): scipy sparse matrix for the bus admittance matrix
        y_branch : list of 2x2 branch admittance matrices with their thermal limits 
        idx_to_bus: id -> bus id (str) 
        bus_to_idx: bus id (str) -> id 
        idx_to_branch: id -> branch id (str)
        branch_to_idx: branch id (str) -> id 
        quad_cost_coeff: quadratic cost coefficients for the gen. cost 
        lin_cost_coeff: linear cost coefficients for gen. cost 
        const_cost_coeff: constant cost coefficients for the gen. cost 
        pg_min: min. active generation limits
        pg_max: max. active generation limits
        qg_min: min. reactive generation limits  
        qg_max: max. reactive generation limits
        v_min: min. voltage magnitude limits  
        v_max: max. voltage magnitude limits
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
        buses: List, 
        branches: List, 
        gens: List, 
        loads: List,
        y_bus: csr_matrix, 
        y_branch: List[BranchAdmittanceMatrix], 
        idx_to_bus: List,
        bus_to_idx: Dict, 
        idx_to_branch: List,
        branch_to_idx: Dict,
        idx_to_gen: List, 
        gen_to_idx: Dict,
        ref_bus_idx: List, 
        non_ref_bus_idx: List, 
        pv_bus_idx: List, 
        pq_bus_idx: List, 
        quad_cost_coeff: List, 
        lin_cost_coeff: List, 
        const_cost_coeff: List, 
        p_min: List, 
        p_max: List, 
        q_min: List, 
        q_max: List, 
        v_min: List, 
        v_max: List) -> None:
        
        self.case_name = case_name
        self.case_data = case_data
        self.buses = buses
        self.branches = branches 
        self.gens = gens 
        self.loads = loads
        self.y_bus = y_bus 
        self.y_branch = y_branch 
        self.bus_to_idx = bus_to_idx
        self.idx_to_bus = idx_to_bus 
        self.branch_to_idx = branch_to_idx 
        self.idx_to_branch = idx_to_branch 
        self.ref_bus_idx = ref_bus_idx 
        self.non_ref_bus_idx = non_ref_bus_idx 
        self.pv_bus_idx = pv_bus_idx 
        self.pq_bus_idx = pq_bus_idx 
        self.quad_cost_coeff = quad_cost_coeff 
        self.lin_cost_coeff = lin_cost_coeff 
        self.const_cost_coeff = const_cost_coeff 
        self.p_min = p_min 
        self.p_max = p_max 
        self.q_min = q_min 
        self.q_max = q_max 
        self.v_min = v_min 
        self.v_max = v_max