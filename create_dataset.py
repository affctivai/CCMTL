import pickle
import os
import scipy.io as scio
import random

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
   

class SEED:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)
        SUBJECT = str(config.subject)
        
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')            
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            
            
        except:
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            file_names = os.listdir(DATA_PATH)
            labels = scio.loadmat(os.path.join(DATA_PATH,'label.mat'))['label'][0]+1
            for fn in file_names:
                if fn in ['readme.txt', 'label.mat']:
                    continue
                exp_subject = fn.split('_')[0]
                
                if exp_subject==SUBJECT: 
                    exp_path = os.path.join(DATA_PATH,fn)
                    mat = scio.loadmat(exp_path)
                    for i in range(1, 16):
                        key='de_LDS'+str(i)
                        feature=mat[key]
                        feature = feature.transpose(1,0,2)
                        if i<10:
                            for fe in feature:
                                train.append((fe,labels[i-1]))
                        else:
                            for fe in feature:
                                test.append((fe,labels[i-1]))
            random.shuffle(train)
            offset=int(len(train)*0.7)
            self.dev=train[2814:]
            train=train[:2814]
            

            to_pickle(train, DATA_PATH+'/train.pkl')
            to_pickle(dev, DATA_PATH+'/dev.pkl')
            to_pickle(test, DATA_PATH+'/test.pkl')

    def get_data(self, mode):
        if mode=='train':
            return self.train
        elif mode=='dev':
             return self.dev
        elif mode=='test':
            return self.test
 
