import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb
from src.Dataloader import (Dataset_Custom, Dataset_Custom_Stddev,
                            Dataset_Custom_Sunpy, Dataset_Pred)
from src.model_informer import *
from src.SkipMissingValueSampler import SkipMissingValueBatchSampler
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

from exp.exp_basic import Exp_Basic

warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'FT':FlareTransformerRegression,
            'FT_linear':FlareTransformerRegressionLastLinear,
            'FT_MAE':FlareTransformerRegressionMAE,
            'FT_MAE_linear':FlareTransformerRegressionrLastLinearMAE,
            'FT_IMG':FlareTransformerRegressionWithoutPhys,
            'FT_MAE_IMG':FlareTransformerRegressionMAEWithoutPhys,
        }
        # if self.args.model=='regression' or self.args.model=='regression_linear':
        #     e_layers = self.args.e_layers if self.args.model=='regression' or  self.args.model=='regression_linear' else self.args.s_layers
        #     model = model_dict[self.args.model](
        #         self.args.enc_in,
        #         self.args.dec_in, 
        #         self.args.c_out, 
        #         self.args.seq_len, 
        #         self.args.label_len,
        #         self.args.pred_len, 
        #         self.args.factor,
        #         self.args.d_model, 
        #         self.args.n_heads, 
        #         e_layers, # self.args.e_layers,
        #         self.args.d_layers, 
        #         self.args.d_ff,
        #         self.args.dropout, 
        #         self.args.attn,
        #         self.args.embed,
        #         self.args.freq,
        #         self.args.activation,
        #         self.args.output_attention,
        #         self.args.distil,
        #         self.args.mix,
        #         self.device
        #     ).float()
        e_layers = self.args.e_layers
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            e_layers, # self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'Flare':Dataset_Custom,
            'Flare_stddev':Dataset_Custom_Stddev,
            'Flare_sunpy':Dataset_Custom_Sunpy,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        if self.args.data == 'Flare_sunpy':
            data_set = Data(
            root_path=args.root_path,
            csv_name=args.csv_name,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            year=args.year,
            )
            print(flag, len(data_set))
            batch_sampler = SkipMissingValueBatchSampler(data_set, batch_size, shuffle=shuffle_flag, drop_last=drop_last)
            data_loader = DataLoader(
                data_set,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers
            )

        
        else:
            data_set = Data(
            root_path=args.root_path,
            feat_path=args.feat_path,
            magnetogram_path=args.magnetogram_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            year=args.year,
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_mag,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_mag, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_mag,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_mag, batch_y, batch_x_mark, batch_y_mark)
                # NOTE:only y_t+24 is used
                # pred = pred[:,-1:,:]
                # true = true[:,-1:,:]
                # print(f"pred: {pred.shape}, true: {true.shape}")
                # print(f"true: {true}")

                # calculate loss without missing value

                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'vali_loss': vali_loss,
                'test_loss': test_loss
            })
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_mag,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_mag, batch_y, batch_x_mark, batch_y_mark)
            # NOTE:only y_t+24 is used
            # pred = pred[:,-1:,:]
            # true = true[:,-1:,:]
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        wandb.log({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        })
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def test_with_loading_model(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        best_model_path = path+'/'+'checkpoint.pth'
        print(f"best model path: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        trues = []
        prev_seqs = []
        
        for i, (batch_x,batch_mag,batch_y,batch_x_mark,batch_y_mark, batch_y_test) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_mag, batch_y, batch_x_mark, batch_y_mark)
            # NOTE:only y_t+24 is used
            # pred = pred[:,-1:,:]
            # true = true[:,-1:,:]
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            batch_y_test = batch_y_test.float()
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y_test = batch_y_test[:,:,f_dim:].to(self.device)
            prev_seqs.append(batch_y_test.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        prev_seqs = np.array(prev_seqs)
        # print('prev_seqs shape:', prev_seqs.shape)
        prev_seqs = prev_seqs.reshape(-1, prev_seqs.shape[-2], prev_seqs.shape[-1])
        # print('prev_seqs shape:', prev_seqs.shape)


        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        wandb.log({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        })
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        np.save(folder_path+'prev_seqs.npy', prev_seqs)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_mag,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_mag, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_mag, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_mag = batch_mag.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_mag, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_mag, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        # batch_y = batch_y[:,-1:,f_dim:].to(self.device)


        return outputs, batch_y
