from sklearn.metrics import f1_score

import torch.nn as nn
import torch
import models
import torch.optim as optim
import os

def create_dir(full_path,new_dir_path,cur_dir_idx):
    splits = full_path.split(os.path.sep)

    if cur_dir_idx!=0:
        new_dir_path=os.path.join(new_dir_path,splits[cur_dir_idx])
        
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)

    if cur_dir_idx <(len(splits)-1):
        create_dir(full_path, new_dir_path,cur_dir_idx+1)

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):
        self.train_config = train_config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        

        self.model = model

    def build(self, cuda = True):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        if cuda:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = getattr(models, self.train_config.model_type)(self.train_config).to(self.device)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.train_config.learning_rate, betas = (self.train_config.b1, self.train_config.b2))
                    
    '''Code adapted from: https://github.com/declare-lab/MISA/blob/master/src/solver.py'''
    def train(self):

        self.saveroot = saveroot=f'checkpoints/{self.train_config.model_type}'
        if self.train_config.modulator:
            self.saveroot = saveroot = os.path.join(self.saveroot,'Modulator')
        else:
            self.saveroot = saveroot = os.path.join(self.saveroot,'No Modulator')
        self.saved_model_name = saved_model_name='sub%s_model.ckpt'%self.train_config.subject
        self.save_optim_name=save_optim_name = 'sub%s_optim_best.std'%self.train_config.subject
  
        save_path= os.path.join(saveroot,self.train_config.data_choice)
        '''create save_path if it does not exist'''
        create_dir(save_path, save_path.split(os.path.sep)[0], 0) 

        curr_patience = patience = 7
        best_valid_loss=float('inf')
        num_trials=self.train_config.num_trials

        for epoch in range(self.train_config.num_epochs):
            self.model.train()

            self.train_help(epoch, self.train_data_loader)

            valid_loss, valid_acc, valid_f1_score=self.evaluate(self.dev_data_loader)

            if valid_loss<best_valid_loss:
                curr_patience=patience
                best_valid_loss = valid_loss
                print(f'========Epoch: {epoch}========')
                print('valid_loss: {:.5f} | valid_acc: {:.3f} | valid_f1_score: {:.3f}'.format(valid_loss, valid_acc, valid_f1_score))
                print('current patience: {}'.format(curr_patience))
                torch.save(self.model.state_dict(), os.path.join(saveroot,self.train_config.data_choice, saved_model_name))
                
                torch.save(self.optimizer.state_dict(), os.path.join(saveroot,self.train_config.data_choice, save_optim_name))
               
            else:
                curr_patience-=1
                if curr_patience <= 0:
                    self.model.load_state_dict(torch.load(os.path.join(saveroot,self.train_config.data_choice, saved_model_name)))
                    self.optimizer.load_state_dict(torch.load(os.path.join(saveroot, self.train_config.data_choice, save_optim_name)))
                    curr_patience = patience
                    num_trials-=1
                    print('trials left: ', num_trials)

            if num_trials<=0:
                print('Running out of patience, training stops!')
                return ''

    def train_help(self, epoch, data_loader):
        left_epochs=4
        
        for features, labels in data_loader:
            features=features.reshape(-1, 310)
            features=torch.unsqueeze(features,1)

            features=features.to(self.device)
            
            if left_epochs == 4:
                self.optimizer.zero_grad()
                    
            left_epochs-=1
            labels= labels.type(torch.LongTensor)
            labels = labels.to(self.device)


            outputs = self.model(features)
          
            loss = self.criterion(outputs, labels)
        
            loss.backward()

            if not left_epochs:
                self.optimizer.step()                
                left_epochs = 4

    # Test the model
    def evaluate(self, data_loader, is_load=False):
        if is_load:
            self.model.load_state_dict(torch.load(os.path.join(self.saveroot, self.train_config.data_choice, self.saved_model_name)))
        self.model.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_loss=0.0
            epoch_f1_score=0.0

            for features, labels in data_loader:
                features=features.reshape(-1, 310)
                features=torch.unsqueeze(features,1)
            
                features = features.to(self.device)

                labels=labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                epoch_loss, correct, total, epoch_f1_score=self.eval_help(features, labels, epoch_loss, epoch_f1_score, total, correct)

            epoch_loss/=len(data_loader)
            epoch_f1_score/=len(data_loader)
            
        return epoch_loss,(100 * (correct / total)), 100*epoch_f1_score

    def eval_help(self,features, labels, epoch_loss, epoch_f1_score, total, correct):

        outputs = self.model(features)
            
        loss = self.criterion(outputs, labels)
        epoch_loss+=loss

        predicted = torch.argmax(outputs.data, 1)

        epoch_f1_score+=f1_score(list(labels.detach().cpu().numpy()), list(predicted.detach().cpu().numpy()),average='weighted')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        return epoch_loss, correct, total, epoch_f1_score
        
