from solver import Solver
from data_loader import get_loader
from config import get_config

if __name__=='__main__':
    
    test_acc_list=[]
    for subject in range(1,16):
        train_config = get_config(mode='train')
        dev_config = get_config(mode='dev')
        test_config = get_config(mode='test')
        
        train_config.subject=dev_config.subject=test_config.subject=subject
        
        train_data_loader =get_loader(train_config, shuffle=True)
        dev_data_loader =get_loader(dev_config, shuffle=False)
        test_data_loader = get_loader(test_config, shuffle=False)

        solver = Solver
        solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)
        
        solver.build()
        
        solver.train()
        
        loss, acc, f1=solver.evaluate(test_data_loader,is_load=True)
        test_acc_list.append(acc)
        print(loss, acc, f1)
    print('mean acc / std')
    print(np.mean(test_acc_list), np.std(test_acc_list))
