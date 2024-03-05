import os 
import os.path as osp
import pandas as pd
import datetime

if __name__ == 'main':
   
    df = pd.DataFrame() 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = "./save/merged_{}.csv".format(timestamp) 

    csv_files = []

    for dirpath, dirnames, filenames in os.walk("./save/"):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            csv_files += [osp.join(dirpath, filename)]
    
    dfs = []
    for f in csv_files:
        dfs += [pd.read_csv(f)]

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(filename)