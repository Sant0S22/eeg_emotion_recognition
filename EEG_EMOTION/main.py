import re
import scipy.io
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import feature_extraction
import preprocessing


# Funzione utilizzata per effettuare reshape di un dataset in un array monodimensionale
# per poter standardizzare il vettore con oggetto StandardScaler
def reshape_and_scaling(df_scale):
    tmp = df_scale.reshape(-1, 1)
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    return scaler.transform(tmp)


# Crea Label da utilizzare per descrivere ogni tupla del Dataset
def create_label(file_name, labels_video, targets_emotion, n, m):
    person = re.findall("\d+_", file_name)
    person = re.findall("\d+", person[0])
    video_label = re.findall("\d+", labels_video[m])
    target_label = targets_emotion[n]
    return person, video_label, target_label


# Funzione utilizzata per aggiungere le label di persona , video , emozione e sessione
# alla tupla creata
def add_labels_and_concat(df_raw, df_concat, label_person, label_video, label_target, label_session):
    df_tmp = pd.DataFrame(df_raw)
    df_tmp = df_tmp.T
    df_tmp['id_user'] = label_person
    df_tmp['session'] = label_session
    df_tmp['video'] = label_video
    df_tmp['emotion'] = label_target
    df_concat = pd.concat([df_concat, df_tmp])
    return df_concat


# Creazione oggetti DataFrame
df_global = pd.DataFrame()
df_psd = pd.DataFrame()
df_de = pd.DataFrame()
df_asm = pd.DataFrame()
df_dasm = pd.DataFrame()

# Scelta Path Assouluto Dataset SEEDIV
# directory = "C:\\Users\\grazi\\Desktop\\Materiale FVAB\\SEED_IV Database\\SEED_IV Database"
# directory = "C:\\Users\\santo\\Downloads\\SEED_IV Database\\SEED_IV Database"
directory = "D:\\DATASET FVAB\\SEED_IV Database\\SEED_IV Database\\"

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
                        # Se si vuole eseguire un analisi sulle feature non segmentate ad 1 sec
                        # utilizzare il metodo feature_extraction.feature_extraction
                        result, psd, entropy, dasm, asm = feature_extraction.feature_extraction_nperseg(filtered_dataset)
                        # result = feature_extraction.feature_extraction(filtered_dataset)

                        # Reshape e Standardizzazione delle tuple con le feature
                        psd = reshape_and_scaling(psd)
                        entropy = reshape_and_scaling(entropy)
                        asm = reshape_and_scaling(asm)
                        dasm = reshape_and_scaling(dasm)

                        # Creazione delle label
                        id_person, video, target = create_label(file1, labels, targets, j, i)
                        j += 1

                        # Aggiunta label sulle tuple e concatenzazione sul dataset globale
                        df_global = add_labels_and_concat(result, df_global, id_person, video, target, session)
                        df_psd = add_labels_and_concat(psd, df_psd, id_person, video, target, session)
                        df_de = add_labels_and_concat(entropy, df_de, id_person, video, target, session)
                        df_asm = add_labels_and_concat(asm, df_asm, id_person, video, target, session)
                        df_dasm = add_labels_and_concat(dasm, df_dasm, id_person, video, target, session)

                        print("Shape Dataset FULL : ", df_global.shape)
                        print("Shape Dataset Feature : ", df_psd.shape, df_de.shape, df_asm.shape, df_dasm.shape)

# Stampa dei Dataset Su file .csv
df_global.to_csv('dataset_eeg_nseg.csv')
df_psd.to_csv('dataset_eeg_psd_stand.csv')
df_de.to_csv('dataset_eeg_de_stand.csv')
df_asm.to_csv('dataset_eeg_asm_stand.csv')
df_dasm.to_csv('dataset_eeg_dasm_stand.csv')
