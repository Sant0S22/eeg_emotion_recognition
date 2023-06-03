import scipy.io
import pandas as pd
import os
import feature_extraction
import preprocessing
import label_dataset_utility


# Creazione oggetti DataFrame
df_global = pd.DataFrame()
df_psd = pd.DataFrame()
df_de = pd.DataFrame()
df_asm = pd.DataFrame()
df_dasm = pd.DataFrame()

# Scelta Path Assouluto Dataset SEEDIV
# directory = "C:\\Users\\grazi\\Desktop\\Materiale FVAB\\SEED_IV Database\\SEED_IV Database"
directory = "C:\\Users\\santo\\Downloads\\SEED_IV Database\\SEED_IV Database"
#directory = "D:\\DATASET FVAB\\SEED_IV Database\\SEED_IV Database\\"

# Label Sessioni
session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
session2_label = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
session3_label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
targets = session1_label + session2_label + session3_label

# Carica Nome dei Channel
channel_order = pd.read_excel(os.path.join(directory, "Channel Order.xlsx"), engine="openpyxl", header=None)
directory = os.path.join(directory, 'eeg_raw_data')
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
                    labels = list(data.keys())
                    size = len(labels)
                    print("file1", file1)
                    for i in range(3, size, 1):
                        id_user = labels[i]
                        df = pd.DataFrame(data[id_user])
                        df = preprocessing.drop_channels(df, channel_order)

                        downsampled = preprocessing.down_sampling(df.T)
                        filtered = preprocessing.band_pass_filter(downsampled)
                        filtered_dataset = pd.DataFrame(filtered).T
                        # Se si vuole eseguire un analisi sulle feature non segmentate a 4 sec
                        # utilizzare il metodo feature_extraction.feature_extraction
                        result, psd, entropy, dasm, asm = feature_extraction.feature_extraction_nperseg(filtered_dataset)
                        # result = feature_extraction.feature_extraction(filtered_dataset)

                        # Reshape e Standardizzazione delle tuple con le feature
                        psd = label_dataset_utility.reshape_and_scaling(psd)
                        entropy = label_dataset_utility.reshape_and_scaling(entropy)
                        asm = label_dataset_utility.reshape_and_scaling(asm)
                        dasm = label_dataset_utility.reshape_and_scaling(dasm)

                        # Creazione delle label
                        id_person, video, target = label_dataset_utility.create_label(file1, labels, targets, j, i)
                        j += 1

                        # Aggiunta label sulle tuple e concatenzazione sul dataset globale
                        df_global = label_dataset_utility.add_labels_and_concat(result, df_global, id_person, video, target, session)
                        df_psd = label_dataset_utility.add_labels_and_concat(psd, df_psd, id_person, video, target, session)
                        df_de = label_dataset_utility.add_labels_and_concat(entropy, df_de, id_person, video, target, session)
                        df_asm = label_dataset_utility.add_labels_and_concat(asm, df_asm, id_person, video, target, session)
                        df_dasm = label_dataset_utility.add_labels_and_concat(dasm, df_dasm, id_person, video, target, session)

                        print("Shape Dataset FULL : ", df_global.shape)
                        print("Shape Dataset Feature : ", df_psd.shape, df_de.shape, df_asm.shape, df_dasm.shape)

# Stampa dei Dataset Su file .csv
df_global.to_csv('CSV/dataset_eeg_nseg.csv')
df_psd.to_csv('CSV/dataset_eeg_psd_stand.csv')
df_de.to_csv('CSV/dataset_eeg_de_stand.csv')
df_asm.to_csv('CSV/dataset_eeg_asm_stand.csv')
df_dasm.to_csv('CSV/dataset_eeg_dasm_stand.csv')
