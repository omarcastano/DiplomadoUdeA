import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def PlotTimeSeriesValidation(dataset):
    N=np.max(dataset.index)
    fig, ax = plt.subplots(figsize=(18,8))
    dataset.plot(0,1,figsize=(18,8), ax=ax)
    ax.axvspan(dataset.index[0], dataset.index[int(N*0.8)], color='red' ,alpha=0.5)
    ax.axvspan(dataset.index[int(N*0.8)], dataset.index[N] ,alpha=0.5)
    plt.text(dataset.index[int(N*0.3)], 620, "80% Training set", fontsize=15)
    plt.text(dataset.index[int(N*0.85)], 620, "20% Test set", fontsize=15)


    
#funcion que nos ayuda a transofrmar los datos
#funcion que nos ayuda a transofrmar los datos
def transform_sequence_from_dataframe(dataset, window_size_x, window_size_y, target_variable , return_df=False):
    ds = pd.DataFrame(index=range(len(dataset)))
    for cols in dataset.columns.values:
        for i in range(window_size_x):
            ds[f"{cols}(t + {i})"] = dataset[cols].shift(-i).values

    for j in range(i+1, (i+1)+window_size_y):
        ds[f'{target_variable}(t+{j})'] = dataset[target_variable].shift(-j).values

    ds.dropna(inplace=True)
    
    if window_size_y ==1:
        X = ds.iloc[:,:-1].to_numpy().reshape((-1, window_size_x ,len(dataset.columns)))
        y = ds.iloc[:,-1].to_numpy().reshape((-1, 1))
    else:
        X = ds.iloc[:,:window_size_x].to_numpy().reshape((-1, window_size_x ,len(dataset.columns)))
        y = ds.iloc[:, (i+1):(i+1)+window_size_y].to_numpy().reshape((-1, window_size_y, 1))

    if return_df:
        return X, y, ds
    else:
        return X, y


