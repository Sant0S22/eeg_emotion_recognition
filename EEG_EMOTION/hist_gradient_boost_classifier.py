import pandas as pd
import numpy as np
import dataset_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score


# Esecuzione del Protocollo Subject Dependent
def subject_dependent(df):
    accuracy_scores = []
    clfSD = HistGradientBoostingClassifier(loss="log_loss", l2_regularization=1.5)
    for i in range(1, 4):
        X_trainSD, X_testSD, y_trainSD, y_testSD = dataset_split.subject_dependent_split(df, i)
        clfSD.fit(X_trainSD, y_trainSD)
        accuracy_scores.append(clfSD.score(X_testSD, y_testSD))
    print("ACCURACY:", np.array(accuracy_scores).mean())


# Esecuzione del Protocollo Subject Indipendent
def subject_indipendent(df):
    logo = LeaveOneGroupOut()
    scores_test = []
    gbc_SI = HistGradientBoostingClassifier(loss="log_loss")
    for i in range(1, 4):
        X_SI, y_SI, groups = dataset_split.subject_independent_split(df, i)
        scores_gbc = cross_val_score(gbc_SI, X_SI, y_SI, cv=logo, verbose=1, groups=groups, n_jobs=-1)
        scores_test.append(scores_gbc.mean())
    print("TEST:", np.array(scores_test).mean())


# Esecuzione del Protocollo Subject Biased
def subject_biased(df):
    X_trainSB, X_testSB, y_trainSB, y_testSB = train_test_split(df.drop('emotion', axis=1), df.emotion, test_size=0.20, random_state=22)
    clf = HistGradientBoostingClassifier(loss="log_loss").fit(X_trainSB, y_trainSB)
    print("ACCURACY", clf.score(X_testSB, y_testSB))


# Scelta Dataset su cui si vuole effettuare il training
# df = pd.read_csv("CSV/dataset_de_LDS_SEEDIV.csv")
# df = pd.read_csv("CSV/dataset_eeg_de_stand.csv")
df = pd.read_csv("CSV/dataset_eeg_nseg.csv")

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df = df.set_index(['id_user', 'session', 'video'])

# Esecuzione dei Protocolli
subject_dependent(df)
subject_indipendent(df)
subject_biased(df)
