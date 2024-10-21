import argparse

class Config(object):
    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
                

def get_config(**optional_kwargs):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_dir', default='/mnt/data/members/fusion/SEED/ExtractedFeatures')
    parser.add_argument('--subject', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='CCMTL')
    parser.add_argument('--lstm_hidden_size', type=int, default=8)
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--w_mode', type=str, default='w')
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--modulator', action='store_true')
    parser.add_argument('--save_file_name', type=str, default='results.csv')
    parser.add_argument('--reduction_ratio', type=int, default=2)
    parser.add_argument('--n_units', type=int, default=78)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--b1', default=0.5)
    parser.add_argument('--b2', default=0.999)
    parser.add_argument('--data_choice', type=str, default='seed')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_epochs',type=int, default=500)
    
    kwargs = parser.parse_args()
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)
    
    return Config(**kwargs)
