from scipy import signal
import scipy
import numpy as np
import pandas as pd

def psd_function(df):
    return signal.welch(df, 200 , noverlap=0, scaling="spectrum")

def de(df):
    return scipy.stats.differential_entropy(df)

def dasm_function(psd):
    left_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[0,2,4]])
    right_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[1,3,5]])
    return np.abs(left_hemisphere_psd-right_hemisphere_psd)/(left_hemisphere_psd+right_hemisphere_psd)

def asm_function(psd):
    left_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[0,2,4]])
    right_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[1,3,5]])
    return (np.sum(left_hemisphere_psd, axis=0) - np.sum(right_hemisphere_psd, axis=0)) / np.sum(left_hemisphere_psd + right_hemisphere_psd, axis=0)

def feature_extraction(dataframe_to_filter):
    #calcolo psd + de + dasm + asm
    f, psd = psd_function(dataframe_to_filter)
    entropy = de(psd)
    dasm = dasm_function(psd)
    asm = asm_function(psd)
    # combinazione risultati psd + de + dasm + asm (approccio che fa esplodere la dimensionalitas del dataset)
    #print("Dim",dataframe_to_filter.shape,psd.shape)
    return np.concatenate((psd.flatten(),entropy,dasm.flatten(),asm))