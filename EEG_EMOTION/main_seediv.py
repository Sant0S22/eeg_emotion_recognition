import re
import scipy.io
import pandas as pd
import os
import numpy as np
import label_dataset_utility

# Directory Dataset SEED-IV (Da importare nella cartella)
directory = "\\SEED_IV Database"

# Label Emozione per video in ogni sessione
session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
targets = session1_label + session2_label + session3_label

# Carica Nome dei Channel (Sensori EEG)
channel_order = pd.read_excel(os.path.join(directory, "Channel Order.xlsx"), engine="openpyxl", header=None)
# Prende gli indici delle righe corrispondenti ai sensori significativi
sens_indexes = np.array(channel_order[channel_order[0].isin(['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'])].index)

# Crea oggetto Dataframe che conterrà i dati estratti
pd_de = pd.DataFrame()

directory = os.path.join(directory, 'eeg_feature_smooth')
if os.path.exists(directory):
    lista_cartella_main = os.listdir(directory)
    # For che Scansiona le cartelle delle sessioni
    for file in lista_cartella_main:
        sec = os.path.join(directory, file)
        if os.path.isdir(sec):
            session = int(sec[-1])
            files1 = os.listdir(sec)
            # For che Scansiona i file .Mat nelle cartelle sessioni
            for file1 in files1:
                j = 24 * (int(session) - 1)
                if file1.endswith(".mat"):
                    file1 = os.path.join(sec, file1)
                    data = scipy.io.loadmat(file1)
                    keys = list(data.keys())
                    print("file1", file1)
                    de_LDS_key = []
                    # Scansiono tutte le key del file e salvo solo quelle che contengono la sottostringa "de_LDS"
                    for key in keys:
                        if "de_LDS" in key:
                            de_LDS_key.append(key)
                    # Scansiono le Key che ho salvato al passo precedente ed estraggo i dati utilizzando queste
                    for key in de_LDS_key:
                        append_video = np.array([])
                        tmp_d = data[key]
                        # Selezione dei 6 canali Rilevanti
                        tmp_d = tmp_d[sens_indexes]
                        # Vettorizzazione del dataset , da X x Y x Z a X x Y
                        for j in range(0, tmp_d.shape[1]):
                            for i in range(0, 5):
                                append_video = np.append(append_video, tmp_d[i][j])

                        # Aggiunta label sulle tuple e concatenazione sul dataset globale
                        id_person = re.findall("\d+_", file1)
                        id_person = re.findall("\d+", id_person[0])
                        video = re.findall("\d+", key)
                        target = targets[j]
                        j += 1
                        pd_de = label_dataset_utility.add_labels_and_concat(append_video, pd_de, id_person, video,
                                                                            target, session)
                    # Stampa Shap del Dataframe in Costruzione
                    print(pd_de.shape)

# Crea la Cartella CSV se non esiste nel path
isExist = os.path.exists("CSV")
if not isExist:
   os.makedirs("CSV")
   
# Stampa dei Dataset Su file .csv
pd_de.to_csv('CSV/dataset_de_LDS_SEEDIV.csv')
