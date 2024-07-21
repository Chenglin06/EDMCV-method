# -*- coding: gbk -*-
import glob
from collections import Counter
import os
import pandas as pd


def find_max(data_dict):
    new_dict = {}
    max_length = max(len(values) for values in data_dict.values())
    longest_key = None
    longest_values = []
    
    for key, values in data_dict.items():
        if len(values) == max_length:
            longest_key = key
            longest_values = values
            new_dict[longest_key] = longest_values
    return new_dict



def combination_dict(data_lst):
    merged_dict = {}
    for d in data_lst:
        for key, values in d.items():
            if key in merged_dict:
                merged_dict[key].extend(values)
            else:
                merged_dict[key] = values
    return merged_dict


        
        
        
        
def find_elements_appeared_three_times(lst):  #找出列表中出现三次的元素
    counter = Counter(lst)
    result = [element for element, count in counter.items() if count == 3]
    return result
        
        
        
        
        


def jisuan(filtered_list,aall_result):         
    three_meth_dict = []
    all_label2file_ = []
    for one in filtered_list:  # result/agg
        meth = one.split('/')[-1]
        meth2list = []
        all_label = []
        label2file = []
        for cate in glob.glob(one + '/*'):  #result/agg/0.txt
            #if 'rest' not in cate:
            catedata = []
            label2data = {}
            for i in open(cate,'r',encoding = 'utf-8'):
                data,label = i.strip().split('\t')
                all_label.append(label)
                if label in label2data:
                    label2data[label].append(i)
                else:
                     label2data[label] = [i]
                catedata.append(i.strip())

            max_dict = find_max(label2data)
            
            for k,v in max_dict.items():
                label2file.append((k,cate))
                all_label2file_.append((k,cate))
            meth2list.append(max_dict)
        label2file_ = {}  
        for key, value in label2file:
            if key in label2file_:
                label2file_[key].append(value)
            else:
                label2file_[key] = [value]
    
        #print(len(meth2list))
        result_dict = combination_dict(meth2list)
        print('=====================================')
        #print(type(result_dict))
        three_meth_dict.append(result_dict)
        all_re = []
        for k,v in result_dict.items(): 
            all_re.extend(v)
            all_ = []
            for i in label2file_[k]:
                for ii in open(i,'r',encoding = 'utf-8'):
                    all_.append(ii)
            
            #print(meth)
            #exit()
            acc = round((len(v)/int(len(all_))),4)
            #print(all_)
            
            #print(len(all_))
            #exit()
            print(f'方法:{meth} 聚类效果：类别{k}准确率:{acc}')
            #exit()
            if meth in aall_result:
                aall_result[meth].append({k:acc})
            else:
                aall_result[meth] = [{k:acc}]
                
        all_acc = round(len(all_re)/len(all_label),4)
        
        print(f'方法:{meth} 聚类效果overall准确率:{all_acc}')
        
        if meth in aall_result:
            aall_result[meth].append({'ovelall':all_acc}) #{'rest':0}
            
            
            #aall_result[meth].append({'rest':0})
        #element_count = dict(Counter(all_label))
    #print(aall_result)
    #exit()
    
    return len(all_label),aall_result

if __name__ == '__main__':
    data_path = 'result'
    
    aall_result = {}
    
    #计算单独三种方法的准确率
    filtered_list = [item for item in glob.glob(data_path + '/*') if "bagging" not in item]
    print(filtered_list)
    #exit()
    all_,aall_result, = jisuan(filtered_list,aall_result)
    
    print('-----------------------------')
    
    #计算单独bagging的准确率
    bagging_path = [item for item in glob.glob(data_path + '/*') if "bagging" in item]
    
    source_file = bagging_path[0] + '/rest.txt'      # 替换为源文件的路径
    destination_folder = '结果/'                     # 替换为目标文件夹的路径
    move_command = f'mv {source_file} {destination_folder}'  # 对于 Windows，可以使用 'move' 命令
    os.system(move_command)
    all_bagging,aall_result = jisuan(bagging_path,aall_result)
    
    rest_num = 0
    for ii in open(destination_folder + 'rest.txt','r'):
        rest_num += 1
        
    
    rest_acc = round((rest_num/all_),4)
    
    
    print(f'聚类效果丢弃率:{rest_acc}')
    aall_result['bagging'].append({'rest_acc':rest_acc})
    print(aall_result)
    data = aall_result
    
    
    
    # 找到最长的列表长度
    max_length = max(len(values) for values in data.values())
    #exit()
    # 填充缺失的值为 NaN
    for key in data:
        while len(data[key]) < max_length:
            data[key].append({})
    print('-------------------')
    print(data)
    #exit()
    # 将字典数据转换为 DataFrame
    df = pd.DataFrame(data)
    
    print(df)
    exit()
   