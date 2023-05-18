import re
import scipy.io
import pandas as pd
import os
import feature_extraction
import preprocessing

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

df_global = pd.DataFrame()
directory = os.path.join(directory, 'eeg_raw_data')
if os.path.exists(directory):
    lista_cartella_main = os.listdir(directory)
    # For che Scansiona le cartelle delle sessioni
    for file in lista_cartella_main:
        sec = os.path.join(directory, file)
        if os.path.isdir(sec):
            print(sec)
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
                    maximum = 0
                    for i in range(3, size, 1):
                        id_user = labels[i]
                        df = pd.DataFrame(data[id_user])
                        # print("Original", df.shape)
                        df = preprocessing.drop_channels(df, channel_order)
                        # print("DropChannel", df.shape)
                        downsampled = preprocessing.down_sampling(df.T)
                        # print("DownSample", downsampled.shape)
                        filtered = preprocessing.band_pass_filter(downsampled)
                        # print("Filtered", filtered.shape)
                        filtered_dataset = pd.DataFrame(filtered).T
                        result = feature_extraction.feature_extraction(filtered_dataset)
                        # print("shape result", result.shape)

                        # Label colonne df finale
                        id_person = re.findall("\d+_", file1)
                        id_person = re.findall("\d+", id_person[0])
                        video = re.findall("\d+", labels[i])
                        target = targets[j]
                        j += 1

                        df_tmp = pd.DataFrame(result).T
                        df_tmp['id_user'] = id_person
                        df_tmp['session'] = session
                        df_tmp['video'] = video
                        df_tmp['emotion'] = target
                        # print(df_tmp.head())
                        # print("shape tmp", df_tmp.shape)
                        df_global = pd.concat([df_global, df_tmp])
                        print("shape df", df_global.shape)

print("HEAD", df_global.head())
print("TAIL", df_global.tail())
print("DESCRIBE", df_global.describe())
print("COLUMNS", df_global.columns)
df_global.to_csv('dataset_eeg.csv')
