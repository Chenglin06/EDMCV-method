import pickle
import csv 
if __name__=="__main__":
    with open("final_result.pkl","rb")as f:
        data=pickle.load(f)
    header = ['method','0','1','overall','reject']
    da = []
    for one in data.keys():
        precision = data[one]['precision']
        overall = data[one]['overall']
        reject = data[one]['reject']
        data_lst = ['0','2']
        list_keys = [k for k,v in precision.items()]
        for i in data_lst:
            if i not in list_keys:
                precision[i] = 0
        data_8 = precision['0']
        data_18 = precision['1']
        #data_14 = precision['2']
        #data_15 = precision['3']
        add = [data_8,data_18,overall, reject]
        fin = [one] + add
        da.append(fin)

    file_name = '结果/'+'result' + '.csv'
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(da)
