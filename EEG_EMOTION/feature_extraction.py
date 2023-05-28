from scipy import signal
import scipy
import numpy as np
import pandas as pd


# Calcolo Feature PSD con 1000Hz utilizzando segmentazione default
def psd_function(df):
    return signal.welch(df, 200, noverlap=0, scaling="spectrum")


# Calcolo Feature PSD con 1000Hz utilizzando segmentazione 1 secondo
def psd_function_nperseg(df):
    return signal.welch(df, 200, noverlap=0, scaling="spectrum", nperseg=1000)


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


# Calcolo di tutte le feature senza segmentazione
def feature_extraction(dataframe_to_filter):
    f, psd = psd_function(dataframe_to_filter)
    entropy = de(psd)
    dasm = dasm_function(psd)
    asm = asm_function(psd)
    return np.concatenate((psd.flatten(), entropy, dasm.flatten(), asm))


# Calcolo di tutte le feature con segmentazione a 1 secondo
def feature_extraction_nperseg(dataframe_to_filter):
    f, psd = psd_function_nperseg(dataframe_to_filter)
    entropy = de(psd)
    dasm = dasm_function(psd)
    asm = asm_function(psd)
    return np.concatenate((psd.flatten(), entropy, dasm.flatten(), asm)), psd.flatten(), entropy, dasm.flatten(), asm
