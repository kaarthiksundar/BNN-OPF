""" Loads the Google data set from the paper 
    OPFData: Large-scale datasets for AC optimal
    power flow with topological perturbations
"""
    
import egret.parsers.matpower_parser as matpower_parser
from torch_geometric.datasets import OPFDataset 
from torch_geometric.loader import DataLoader


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
    
    training_loader = DataLoader(train_ds, batch_size = 4, shuffle = True)
    
    bus_dict = case_data['elements']['bus']
    branch_dict = case_data['elements']['branch']
    gen_dict = case_data['elements']['generator']
    bus_list = list(bus_dict.keys())
    branch_list = list(branch_dict.keys())
    gen_list = list(gen_dict.keys())   
    
