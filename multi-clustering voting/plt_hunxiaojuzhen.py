import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob
from collections import Counter
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, accuracy_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)

def plot_confusion_matrix(true_labels, predicted_labels, classes, name):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=classes)
    accuracy = accuracy_score(true_labels, predicted_labels)
    error_rate = 1 - accuracy
    
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    metrics_name = "_".join([f"class{i}_recall_{recall[i]:.4f}_precision_{precision[i]:.4f}_specificity_{specificity[i]:.4f}" for i in range(len(classes))])
    file_name = f"{name}_{metrics_name}_F1_{f1.mean():.4f}_Acc_{accuracy:.4f}_Mcc_{mcc:.4f}.png"
    plt.savefig(f'confusion_matrix/{file_name}', dpi=600)
    plt.close()
    
    
    metrics_dict = {
        'Precision 0': [precision[0]],
        'Precision 1': [precision[1]],
        'Recall 0': [recall[0]],
        'Recall 1': [recall[1]],
        'Specificity 0': [specificity[0]],
        'Specificity 1': [specificity[1]],
        'F1 Score': [np.mean(f1)],
        'MCC': [mcc],
        
        'Accuracy': [accuracy]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    csv_file_name = f"confusion_matrix/{name}_metrics.csv"
    df_metrics.to_csv(csv_file_name, index=False)


def most_common_keys(counts):
    max_count = counts.most_common(1)[0][1]
    return [key for key, count in counts.items() if count == max_count]

def find_key_for_value(d, target_value):
    for key, values in d.items():
        if target_value in values:
            return key
    return None

def get_julei(name):
    dict_true = {}
    dict_1 = {}
    true_1 = []
    pre_1 = []
    panthh = 'result/'+name + '/*' 
    path_lst = glob.glob(panthh)
    for line in path_lst:
        if 'rest' in line:
            continue
        lst_1 = []
        lst_2 = []
        for i in open(line,'r',encoding="utf-8"):
            sample,label = i.strip().split('\t')
            dict_true[sample]=label
            lst_1.append(label)
            lst_2.append(sample)
        true_counts = Counter(lst_1) 
        max_label = most_common_keys(true_counts)
        if max_label[0] not in dict_1:
            dict_1[max_label[0]]=[]
        dict_1[max_label[0]].extend(lst_2)
    for k,v in dict_true.items():
        kk = find_key_for_value(dict_1, k)
        true_1.append(v)
        pre_1.append(kk)
    return true_1, pre_1


def create_directory(directory_name):
    try:
        os.makedirs(directory_name, exist_ok=True)
        print(f"Directory '{directory_name}' created successfully or already exists.")
    except Exception as e:
        print(f"Error creating directory {directory_name}: {e}")



def delete_directory(directory_name):
    try:
        if os.path.exists(directory_name) and os.path.isdir(directory_name):
            shutil.rmtree(directory_name)
            print(f"Directory '{directory_name}' has been deleted.")
        else:
            print(f"Directory '{directory_name}' does not exist or is not a directory.")
    except Exception as e:
        print(f"Error deleting directory {directory_name}: {e}")

if __name__=="__main__":
    #classes = ['low','medium','high']
    classes = ['0','1']
    import os
    import shutil
    
    delete_directory("confusion_matrix")
    create_directory("confusion_matrix")
    
    name_lst = ['bagging','kmeans','agg','birch']
    for name in name_lst:
        true_labels, predicted_labels = get_julei(name)
        plot_confusion_matrix(true_labels, predicted_labels, classes, name)
    
    