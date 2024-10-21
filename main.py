from solver import Solver
from data_loader import get_loader
from config import get_config

if __name__=='__main__':
    
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')
    
    train_data_loader =get_loader(train_config, shuffle=True)
    dev_data_loader =get_loader(dev_config, shuffle=False)
    test_data_loader = get_loader(test_config, shuffle=False)

    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)
    
    solver.build()
    
    solver.train()
    
    loss, acc, f1=solver.evaluate(test_data_loader,is_load=True)
    
    print(loss, acc, f1)
