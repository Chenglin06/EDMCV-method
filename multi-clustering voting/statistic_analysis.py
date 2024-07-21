import glob
import os
from tqdm import tqdm

data_path = "result/"
#save_path = "right_result/"
label_dic={}
mode_lst = glob.glob(data_path+"*")
for mode_name in mode_lst:

    #print(mode_name)
    #exit()
    label_lst = glob.glob(mode_name+"/*")
    for label_name in label_lst:
        #aa = 
        for i in open(label_name,'r',encoding = 'utf-8'):
            name,label = i.strip().split('\t') 
        #file_lst = glob.glob(label_name+"/*")
        
        #for file_name in file_lst:
            #label = file_name.split("/")[-1].split("_")[0]
            label_dic[label] = label_dic.get(label,0)+1
        #name = "_".join(file_name.split("/")[-3:-1])
        
        #print(label_name) 
        #exit()
        max_num = max(label_dic.values())
        label_ = [k for k,v in label_dic.items() if v==max_num]
        all_num = sum(label_dic.values())
        acc = max_num/all_num
        print("%s文件夹中，%s 为最多类，其准确率为%4f"%(label_name, label_,acc))
        print(label_dic)
        label_dic={}

    print("++++++++++++++++++++++++++++++++++++++++++++")


