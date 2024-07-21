import argparse
import os
from pathlib import Path as p
from time import time
import numpy as np
import random

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (load_data, set_data_plot, plot,
                   AutoEncoder, DEC,
                   pretrain, train, get_initial_center, 
                   accuracy)


def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=6764, type=int, help='batch size')
    arg.add_argument('-pre_epoch', default=100, type=int, help='epochs for train Autoencoder')
    arg.add_argument('-epoch', default=100, type=int, help='epochs for train DEC')
    arg.add_argument('-k', type=int, help='num of clusters')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-worker', default=4, type=int, help='num of workers')
    arg.add_argument('-seed', default=1000, type=int, help='torch random seed')
    arg = arg.parse_args()
    return arg
    

def main():
    arg = get_arg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #device = 'cpu'
    
    #if not os.path.exists(arg.save_dir):
    #    os.makedirs(arg.save_dir, exist_ok=True) 
    #else:
    #    for path in p(arg.save_dir).glob('*.png'):
    #        path.unlink()
        
    if arg.seed is not None:
        random.seed(10)
        np.random.seed(10)
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    tr_ds, test_ds = load_data(arg.bs, arg.worker)
         
    print('\ntrain num:', len(tr_ds.dataset))
    print('test num:', len(test_ds.dataset))
    # for visualize
    set_data_plot(tr_ds, test_ds, device)
    # train autoencoder
    ae = AutoEncoder().to(device)  
    
    print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M')
    
    opt = AdamW(ae.parameters())
    #print(ae)  
    print('*' * 50)
    print('load the best ae ...')
    ae_model = torch.load(f'{arg.save_dir}/fine_tune_AE.pt', device)
    
    #print(ae)
    #exit()
    
    ae_model.eval()
    with torch.no_grad():
        for x, y in test_ds:
            x = x.to(device)
            x = x.float()
            #truth.append(y)
            out1,out_encode = ae_model(x)
            
            #print(out_encode.shape)
            #exit()
    
    encode_data_out = out_encode.tolist()
    #print(len(encode_data_out))

    print('*' * 50)
    print('load the best DEC ...')
    dec = torch.load(f'{arg.save_dir}/DEC.pt', device)
    #exit()
    model = dec.encoder
    truth, pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_ds:
            x = x.to(device)
            x = x.float()
            #truth.append(y)
            out = model(x)
    
    raw_data = x.tolist()
    real_label = y.tolist()
    encode_data = out.tolist()
    data2name = {}
    for lin in open('../../esm_encode/encode_data_1708','r',encoding = 'utf-8'):
        name,label,data = lin.strip().split('\t')
        data2name[data] = name
        
    #exit()
    with open('encode_dec','w',encoding = 'utf-8')as f,open('name2label','w',encoding = 'utf-8')as f_name,open('encode_ae','w',encoding = 'utf-8')as f_ae:
        for i,j,k,l in zip(raw_data,real_label,encode_data,encode_data_out):
            
            name =  data2name[str(i)]
            
            new_ = '\t'.join([str(name),str(j),str(k)])
            new_ae = '\t'.join([str(name),str(j),str(l)])
            
            new2label = '\t'.join([str(name),str(j)])
            f.write(new_ + '\n')
            f_name.write(new2label + '\n')
            f_ae.write(new_ae + '\n')
            

    print('Evaluate ...')
    
    #exit()
    #acc = accuracy(dec, test_ds, device)
    acc, f1, ari, nmi, precision, recall = accuracy(dec, test_ds, device)
    print(f'test acc: {acc:.4f}')
    print('*' * 50)
    #plot(dec, arg.save_dir, 'test')

    #print(f'\ntrain AE time: {t1 - t0:.2f} s')
    #print(f'get inititial time: {t3 - t2:.2f} s')
    #print(f'train DEC time: {t5 - t4:.2f} s')

    
    
if __name__ == '__main__':
    main()
