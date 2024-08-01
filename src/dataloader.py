""" For now using case57.py (copied from pypower) 
    Long term use EGRET to read the pglib cases and 
    transition to a json training data parser (from a .mat)
    This is currently being done to compare with existing work 
    by Pascal's group and the paper 
    DC3: A learning method for optimization with hard constraints
"""
    
import scipy 
import egret.parsers.matpower_parser as matpower_parser


def load_training_data_from_mat(
    trainingfile: str, 
    casefile: str, 
    log
):
    
    training_data = scipy.io.loadmat(trainingfile)
    case_data = matpower_parser.create_model_data_dict(casefile)
    
    bus_dict = case_data['elements']['bus']
    branch_dict = case_data['elements']['branch']
    gen_dict = case_data['elements']['generator']
    bus_list = list(bus_dict.keys())
    branch_list = list(branch_dict.keys())
    gen_list = list(gen_dict.keys())
    print(gen_list)    
    
