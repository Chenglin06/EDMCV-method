# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random

# def load_data(file_path,select_num):
def load_data(file_path):
    data_dict = {}
    with open(file_path, 'r') as f:
        for info in f.readlines():#[:180]:
            name,label,data = info.strip().split('\t')
            
            #name,data = info.strip().split('\t')
            data = eval(data)
            #print(data)
            #exit()
            #label = name.split('_')[0]

            if label not in data_dict:
                data_dict[label] = []
            if label in data_dict:
                data_dict[label].append(data)
    all_label= []
    all_data = []


    for k,v in data_dict.items():
        all_label.extend([k]*len(v))
        all_data.extend(v)
    label_lst = all_label
    data_np = np.array(all_data)
    # print(data_np.shape)
    # print(len(label_lst))
    # exit()
    return label_lst, data_np

    '''
    for k,v in data_dict.items():
        if len(v) > select_num:
            all_label.extend([k]*select_num)
            v = list(v)
            
            # print(v[0][0])
            # random.shuffle
            random.shuffle(v)
            # print(v[0][0])
            # exit()
            all_data.extend(v[:select_num])
            # print(all_data)
    
    label_lst = all_label
    
    data_np = np.array(all_data)
    print(data_np.shape)
    print(len(label_lst))
    exit()
    return label_lst, data_np
    '''

def plot_embedding(data, label):
    # 随机生成颜色和形状
    color_dict = {}
    marker_dict = {}
    # color_lst = ["red", "blue",'pink','yellow']
    color_lst = ["red", "blue"]
    for lbl in label:
        if lbl not in color_dict:

            color_dict = {"0": "red", "1": "blue"}
            marker_dict = {"0": ".", "1": "*"}

            '''
            # print(color_lst) 
            chose_color = random.choice(color_lst)
            color_lst = [x for x in color_lst if x != chose_color]
            
            color_dict[lbl] = chose_color#"#" + "%06x" % random.randint(0, 0xFFFFFF)
            marker_dict[lbl] = random.choice([".", "*"])
            '''

    # 绘制散点图
    fig, ax = plt.subplots()
    for i in range(data.shape[0]):
        lbl = label[i]
        ax.scatter(data[i, 0], data[i, 1], color=color_dict[lbl], marker=marker_dict[lbl])


    # 初始化图例元素列表
    handles = [] 

    # 遍历不同的标签 
    for lbl in color_dict:
      handle = plt.plot([],[],color=color_dict[lbl], marker=marker_dict[lbl], ls='', label=lbl)[0] 
      handles.append(handle) 
    ax.legend(handles=handles)

    plt.xticks([])
    plt.yticks([])
    plt.savefig('t-SNE_dec.png',dpi = 600)


if __name__ == "__main__":
    #获取label和data对应的信息，data需要时numpy格式
    # num = int(input('每一个类别选取多少个数据绘制图像:'))
    #file_path = 'encode.txt'
    file_path = './encode_dec'
    # label, data = load_data(file_path,num)
    label, data = load_data(file_path)

    # T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=4)
    result = tsne.fit_transform(data)
    # exit()
    # T-SNE后画图需要归一化处理
    max_, min_ = np.max(result, 0), np.min(result, 0)
    normalized_data = (result - min_) / (max_ - min_)
    plot_embedding(normalized_data, label)
