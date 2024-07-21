import glob
import pickle

t={}
file_s=glob.glob("result_final/*")

for file_ in file_s:
    method=file_.split("/")[-1]
    t[method]={}
    labels=glob.glob(file_+"/*")

    for la in labels:
        label=la.split("/")[-1]
        t[method][label]=[]
        ims=glob.glob(la+"/*")
        
        for im in ims:
            name=im.split("/")[-1]
            t[method][label].append(name)

with open("result_final/result_label2name","wb")as f:
    pickle.dump(t,f)
