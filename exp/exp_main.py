from model import lstm_attention, InvT_Encoder, Classifier
from exp.exp_basic import Exp_Basic
from data.statcast_dataset import get_loader

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from torch import optim
from torch.optim import lr_scheduler 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        
        self.location_classifier = Classifier.Classifier(output_size = 4).to(self.device)
        self.type_classifer = Classifier.Classifier(output_size = (len(self.args.pitch_name_list) + 1)).to(self.device)
        
    def _build_model(self):
        model_dict = {
            'lstm_attention': lstm_attention,
            'invT': InvT_Encoder
        }
        
        model = model_dict[self.args.model].Model(self.args)
        return model
        
    def _get_data(self, flag):
        if flag == 'train':
            dataset, loader, type_weight, loc_weight = get_loader(self.args, flag)
            return dataset, loader, type_weight, loc_weight
        else:
            dataset, loader = get_loader(self.args, flag)
            return dataset, loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self, type_weight = None, location_weight = None):
        criteria = {
            "type": nn.CrossEntropyLoss(weight=type_weight),
            "location": nn.CrossEntropyLoss(weight=location_weight)            
        }
        return criteria
    
    def valid(self, valid_data, valid_loader, criterion):
        total_loss, total_type_loss, total_location_loss = [], [], []
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                
                batch_x, real_seq_len, mask, target = data['padded_data'], data["real_sequence_length"], data["padding_mask"], data["targets"]
                batch_x = batch_x.float().to(self.device)
                batch_y = target.float()
                
                batch_real_seq_len = real_seq_len.float().to(self.device)
                # batch_mask = mask.float().to(self.device)

                backbone_output = self.model(batch_x, batch_real_seq_len) if self.args.model in 'LSTM' else self.model(batch_x)
            
                # pitch type
                pitch_type_output = self.type_classifer(backbone_output)
                
                if self.args.model in 'LSTM':
                    pred_type = pitch_type_output.to(self.device)
                    
                else:
                    pred_type = pitch_type_output[:, self.args.target_dim[0], :].squeeze(-1)
                    
                true_type = target[:, 0].to(self.device).long()
                
                type_loss = criterion['type'](pred_type, true_type)
                total_type_loss.append(type_loss)
                
                # pitch location
                pitch_location_output = self.location_classifier(backbone_output)
                
                if self.args.model in 'LSTM':
                    pred_location = pitch_location_output.to(self.device)
                    
                else:
                    pred_location = pitch_location_output[:, self.args.target_dim[1], :].squeeze(-1)
                    
                true_location = target[:, 1].to(self.device).long()
                
                location_loss = criterion['location'](pred_location, true_location)
                total_location_loss.append(location_loss)
                
                loss = self.args.ld * type_loss + (1- self.args.ld) * location_loss
                total_loss.append(loss)
                
        total_loss = np.average([loss.detach().cpu().numpy() for loss in total_loss])
        type_loss = np.average([tl.detach().cpu().numpy() for tl in total_type_loss])
        location_loss = np.average([ll.detach().cpu().numpy() for ll in total_location_loss])
        self.model.train()
        return total_loss, type_loss, location_loss
    
    def train(self, setting):
        train_data, train_loader, type_weight, location_weight = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'valid')
        
        type_weight, location_weight = torch.FloatTensor(type_weight).to(self.device), torch.FloatTensor(location_weight).to(self.device)
        
        model_path = os.path.join(self.args.model_save_pth, setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(type_weight = type_weight, location_weight = location_weight)
        prev_valid_loss = 1e9
        
        best_model_path = model_path + '/' + 'checkpoint.pth'
        
        for epoch in range(self.args.epoch):
            
            iter_count = 0
            train_total_loss, train_type_loss, train_location_loss = [], [], []
            
            self.model.train()
            
            for i, data in enumerate(tqdm(train_loader)):
                
                batch_x, real_seq_len, mask, target = data['padded_data'], data["real_sequence_length"], data["padding_mask"], data["targets"]
                
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_real_seq_len = real_seq_len.float().to(self.device)
                # batch_mask = mask.float().to(self.device)
                
                backbone_output = self.model(batch_x, batch_real_seq_len) if self.args.model in 'LSTM' else self.model(batch_x)
                
                # pitch type
                pitch_type_output = self.type_classifer(backbone_output)
                
                if self.args.model in 'LSTM':
                    pred_type = pitch_type_output.to(self.device)
                    
                else:
                    pred_type = pitch_type_output[:, self.args.target_dim[0], :].squeeze(-1)
                    
                true_type = target[:, 0].to(self.device).long()
                type_loss = criterion['type'](pred_type, true_type)
                train_type_loss.append(type_loss.item())
                
                # pitch location
                pitch_location_output = self.location_classifier(backbone_output)
                
                if self.args.model in 'LSTM':
                    pred_location = pitch_location_output.to(self.device)
                    
                else:
                    pred_location = pitch_location_output[:, self.args.target_dim[1], :].squeeze(-1)
                    
                true_location = target[:, 1].to(self.device).long()
                location_loss = criterion['location'](pred_location, true_location)
                train_location_loss.append(location_loss.item())
                
                # total loss  
                total_loss = self.args.ld * type_loss + (1 - self.args.ld) * location_loss
                train_total_loss.append(total_loss)
                total_loss.backward()
                model_optim.step()
                
            train_loss, epoch_type_loss, epoch_location_loss = np.average([loss.detach().cpu().numpy() for loss in train_total_loss]), np.average(train_type_loss), np.average(train_location_loss)
            valid_loss, valid_type_loss, valid_location_loss = self.valid(valid_data, valid_loader, criterion)
            
            print("Epoch: {0} | Train Loss: {1: .7f}, Train Pitch Type Loss: {2: .7f}, Train Pitch Location Loss: {3: .7f}, Valid Loss: {4: .7f}".format(
                epoch+1, train_loss, epoch_type_loss, epoch_location_loss, valid_loss
            ))
            
            if prev_valid_loss > valid_loss:
                print(f"Model update at epoch {epoch+1}, valid_loss: {valid_loss}")
                torch.save({
                    'backbone': self.model.state_dict(),
                    'type_classifier': self.type_classifer.state_dict(),
                    'location_classifier': self.location_classifier.state_dict()}
                           ,best_model_path)
                prev_valid_loss = valid_loss
                
                # self.model.load_state_dict(torch.load(best_model_path))
                
        return self.model
    
    def test(self, setting, model_path = None):
        
        test_data, test_loader = self._get_data(flag = 'test')
        
        if model_path is None:
            print('Loading Model...')
            pth = torch.load(os.path.join('./save/' + setting, 'checkpoint.pth'))
            
        else:
            pth = torch.load(model_path)
            
        self.model.load_state_dict(pth['backbone'])
        self.type_classifer.load_state_dict(pth['type_classifier'])
        self.location_classifier.load_state_dict(pth['location_classifier'])
        
        pitch_type_true, pitch_type_predict = [], []
        pitch_location_true, pitch_location_predict = [], []
        
        result_path = './test_results/' + setting + '/'
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                
                batch_x, real_seq_len, mask, target = data['padded_data'], data["real_sequence_length"], data["padding_mask"], data["targets"]
                batch_x = batch_x.float().to(self.device)
                batch_y = target.float()
                
                batch_real_seq_len = real_seq_len.float().to(self.device)
                # batch_mask = mask.float().to(self.device)
                
                backbone_output = self.model(batch_x, batch_real_seq_len) if self.args.model in 'LSTM' else self.model(batch_x)
                
                # pitch type
                pitch_type_output = self.type_classifer(backbone_output)
                
                if self.args.model in 'LSTM':
                    pred_type = pitch_type_output.detach().cpu().numpy()
                else:
                    pred_type = pitch_type_output[:, self.args.target_dim[0], :].detach().cpu().numpy()
                    
                true_type = target[:, 0].detach().cpu().long().numpy()
                
                pred_type = np.argmax(pred_type, axis = 1)
                pitch_type_predict.append(pred_type)
                pitch_type_true.append(true_type)
                
                # pitch location
                pitch_location_output = self.location_classifier(backbone_output)
                
                if self.args.model in 'LSTM':
                    pred_location = pitch_location_output.detach().cpu().numpy()
                else:
                    pred_location = pitch_location_output[:, self.args.target_dim[1], :].detach().cpu().numpy()
                    
                true_location = target[:, 1].detach().cpu().long().numpy()
                
                pred_location = np.argmax(pred_location, axis = 1)
                pitch_location_predict.append(pred_location)
                pitch_location_true.append(true_location)
        
        pitch_type_predict, pitch_type_true = np.array(pitch_type_predict).reshape(-1), np.array(pitch_type_true).reshape(-1)
        pitch_location_predict, pitch_location_true = np.array(pitch_location_predict).reshape(-1), np.array(pitch_location_true).reshape(-1)
        
        ## Accuracy metrics
        
        type_acc, location_acc = accuracy_score(pitch_type_true, pitch_type_predict), accuracy_score(pitch_location_true, pitch_location_predict)
        type_prec, location_prec = precision_score(pitch_type_true, pitch_type_predict, average = 'weighted'), precision_score(pitch_location_true, pitch_location_predict, average = 'weighted')
        type_rec, location_rec = recall_score(pitch_type_true, pitch_type_predict, average = 'weighted'), recall_score(pitch_location_true, pitch_location_predict, average = 'weighted')
        type_f1, location_f1 = f1_score(pitch_type_true, pitch_type_predict, average = 'weighted'), f1_score(pitch_location_true, pitch_location_predict, average = 'weighted')
        
        ## Confusion Matrix
        pitch_mapping = {pitch: idx for idx, pitch in enumerate(self.args.pitch_name_list, start=0)}
        pitch_mapping['Others'] = len(pitch_mapping) + 1
        pitch_labels = list(pitch_mapping.keys())
        type_cm, location_cm = confusion_matrix(pitch_type_true, pitch_type_predict), confusion_matrix(pitch_location_true, pitch_location_predict)
        
        type_cm_percentage = type_cm / type_cm.sum(axis=1)[:, np.newaxis]  
        type_cm_percentage = np.round(type_cm_percentage * 100, 2)
        
        location_cm_percentage = location_cm / location_cm.sum(axis = 1)[:, np.newaxis]
        location_cm_percentage = np.round(location_cm_percentage * 100, 2)
        
        ## pitchtype
        plt.figure(figsize = (20, 10))
        sns.heatmap(type_cm_percentage, annot = True , cmap = 'Blues', xticklabels=pitch_labels, yticklabels=pitch_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Pitch Type Classification Heatmap')
        plt.savefig(result_path + 'pitch_classification.png')
        plt.close()
        
        ## Location
        plt.figure(figsize = (20, 10))
        sns.heatmap(location_cm_percentage, annot = True , cmap = 'Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Pitch Location Classification Heatmap')
        plt.savefig(result_path + 'pitch_location_classification.png')
        plt.close()

        print('Pitch Type Classifier F1 Score: {} | Pitch Location Classifier F1 Score: {}'.format(
            type_f1,
            location_f1
        ))
        
        with open(result_path + 'test results.txt', 'w', encoding = 'utf-8') as file:
            file.write(setting + "  \n")
            file.write('Pitch Type Accuracy: {}, Pitch Type Precision: {}, Pitch Type Recall:{}, Pitch Type F1:{}'.format(
                type_acc,
                type_prec,
                type_rec,
                type_f1
            ) + "   \n")
            file.write('Pitch Location Accuracy: {}, Pitch Location Precision: {}, Pitch Location Recall:{}, Pitch Location F1: {}'.format(
                location_acc,
                location_prec,
                location_rec,
                location_f1
            ) + "   \n")
            file.write('\n')
            file.close()
        
        return