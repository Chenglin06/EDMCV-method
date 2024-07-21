import torch
import tqdm
import torch
import esm
import torch.hub 
torch.hub.set_dir('/data/duqimeng/anjisuan_bigmodel')
import random

def split_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


all_data = []
#with open('encode_data_new_','w',encoding = 'utf-8')as f:
#my_list = list(range(32001, 32061))
my_list = list(range(0, 30000))
#print(len(my_list))
#exit()

#for i,j in enumerate(open('output','r',encoding = 'utf-8')):
for j,i in zip(open("raw_data_1708",'r',encoding = 'utf-8'),my_list):
    #if len(j) > 5:
    data,label = j.strip().split('\t')
    if len(data) > 1020:
        data = data[:1020]
    if len(data) <= 1020:
        data = data 
    
    data_mutation = data
    data_ = (str(label) + '_' + str(i)+'_' + str(data),data_mutation)
    all_data.append(data_)

small_lists = list(split_list(all_data, 4))


model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

for one in small_lists:

    batch_labels, batch_strs, batch_tokens = batch_converter(one)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    
    print(batch_labels)
    print(batch_tokens.shape)
    print('================')
    print(batch_lens.shape)
    #exit()
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
    
    print(token_representations)
    print(token_representations.shape)
    
    with open('encode_data_1708','a',encoding = 'utf-8')as f:
        
        for i, tokens_len in enumerate(batch_lens):
            dd = token_representations[i, 1 : tokens_len - 1].mean(0)
            dd = dd.tolist()
            #print(dd)
            name_ = batch_labels[i]
            label,num,name = name_.split('_')
            
            aa = name+ '\t' + label+ '\t' + str(dd)+ '\n'
            #print(aa)
            f.write(aa)
