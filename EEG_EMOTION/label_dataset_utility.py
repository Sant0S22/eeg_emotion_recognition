import re
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Funzione utilizzata per effettuare reshape di un dataset in un array monodimensionale
# per poter standardizzare i valori utilizzando uno StandardScaler
def reshape_and_scaling(df_scale):
    tmp = df_scale.reshape(-1, 1)
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    return scaler.transform(tmp)


# Crea le label da utilizzare per descrivere ogni tupla del Dataset
def create_label(file_name, labels_video, targets_emotion, n, m):
    person = re.findall("\d+_", file_name)
    person = re.findall("\d+", person[0])
    video_label = re.findall("\d+", labels_video[m])
    target_label = targets_emotion[n]
    return person, video_label, target_label


# Funzione utilizzata per aggiungere le label persona, video, emozione e sessione alla tupla creata
def add_labels_and_concat(df_raw, df_concat, label_person, label_video, label_target, label_session):
    df_tmp = pd.DataFrame(df_raw)
    df_tmp = df_tmp.T
    df_tmp['id_user'] = label_person
    df_tmp['session'] = label_session
    df_tmp['video'] = label_video
    df_tmp['emotion'] = label_target
    df_concat = pd.concat([df_concat, df_tmp])
    return df_concat
