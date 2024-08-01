""" Loads the Google data set from the paper 
    OPFData: Large-scale datasets for AC optimal
    power flow with topological perturbations
"""
    
import egret.parsers.matpower_parser as matpower_parser


def load_training_data_from_mat(
    training_folder: str, 
    case_file: str, 
    log
):
    
    case_data = matpower_parser.create_model_data_dict(case_file)
    
    bus_dict = case_data['elements']['bus']
    branch_dict = case_data['elements']['branch']
    gen_dict = case_data['elements']['generator']
    bus_list = list(bus_dict.keys())
    branch_list = list(branch_dict.keys())
    gen_list = list(gen_dict.keys())   
    
