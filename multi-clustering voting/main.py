import config as CF
import model as m
import glob
import os
import time
import numpy as np

def load_data(file_):
    data=[]
    file_lst=[]
    for line in open(file_,"r",encoding="utf-8"):
        name,label,da =line.strip().split("\t")
        #print(name)
        #exit()
        #label = name.split('_')[0]
        #print(label)
        #pythonexit()
        #da = [float(i) for i in da.split('<=>')]
        da = eval(da)
        #da = for i in da.split('<=>')
        #print(len(da))
        #xexit()
        #eval(da) 
        label_='\t'.join([name,label])
        file_lst.append(label_)
        data.append(da)
    return file_lst,data
    


st=time.time()
#file_lst,data=load_data("../unsupervised/encode.txt") #encode_data_waimai.txt
file_lst,data=load_data("../dec_cluster/encode_dec") #encode_data_waimai.txt 

#file_lst,data=load_data("raw_data")

print("sample number is %s, sample dim is %s"%(len(data),len(data[0])))
class_num=CF.config["class_num"]
for model_type in CF.config["type"]: 
    model= m.model(class_num=class_num)
    model.build(model_type)
    model.run(data)
    model.show(file_lst)

print("聚类完成，一共用时:%s秒"%(time.time()-st))
