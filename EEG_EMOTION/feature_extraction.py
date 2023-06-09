from scipy import signal
import scipy
import numpy as np
import pandas as pd


# Calcolo Feature PSD a 200Hz utilizzando segmentazione default
def psd_function(df):
    return signal.welch(df, 200, noverlap=0, scaling="spectrum")


# Calcolo Feature PSD a 200Hz utilizzando segmentazione 4 secondi (200x4)
def psd_function_nperseg(df):
    return signal.welch(df, 200, noverlap=0, scaling="spectrum", nperseg=800)


# Calcolo Feature DE
def de(df):
    return scipy.stats.differential_entropy(df)


# Calcolo Feature DASM
def dasm_function(psd):
    left_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[0, 2, 4]])
    right_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[1, 3, 5]])
    return np.abs(left_hemisphere_psd - right_hemisphere_psd) / (left_hemisphere_psd + right_hemisphere_psd)


# Calcolo Feature ASM
def asm_function(psd):
    left_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[0, 2, 4]])
    right_hemisphere_psd = np.array(pd.DataFrame(psd).iloc[[1, 3, 5]])
    return (np.sum(left_hemisphere_psd, axis=0) - np.sum(right_hemisphere_psd, axis=0)) / np.sum(
        left_hemisphere_psd + right_hemisphere_psd, axis=0)


# Calcolo di tutte le feature utilizzando la segmentazione di default
def feature_extraction(dataframe_to_filter):
    f, psd = psd_function(dataframe_to_filter)
    entropy = de(psd)
    dasm = dasm_function(psd)
    asm = asm_function(psd)
    return np.concatenate((psd.flatten(), entropy, dasm.flatten(), asm))


# Calcolo di tutte le feature utilizzando la segmentazione a 4 secondi
def feature_extraction_nperseg(dataframe_to_filter):
    f, psd = psd_function_nperseg(dataframe_to_filter)
    entropy = de(dataframe_to_filter)
    dasm = dasm_function(psd)
    asm = asm_function(psd)
    return np.concatenate((psd.flatten(), dasm.flatten(), asm, entropy )), psd.flatten(), entropy, dasm.flatten(), asm

