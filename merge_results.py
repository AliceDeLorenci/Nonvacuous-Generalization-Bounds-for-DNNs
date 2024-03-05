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
    
    df_mlp = df[df["nn_type"] == "mlp"]
    df_cnn = df[df["nn_type"] == "cnn"]
    
    df_mlp = df_mlp.drop(columns = ['nn_type', 'kernel_size', 'nfilters', 'nin_channels'])
    df_cnn = df_cnn.drop(columns = ['nn_type'])
    
    df_mlp.to_csv("./save/mlp_{}.csv".format(timestamp))
    df_cnn.to_csv("./save/cnn_{}.csv".format(timestamp))