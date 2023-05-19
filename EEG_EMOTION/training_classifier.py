import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score


def split_train_test_sd(df):
    df_sd = df.reset_index()
    # generiamo 8 video su cui eseguiamo i test
    test_videos = np.random.randint(1, 25, size=8)
    print(test_videos)
    X_testSD = df_sd[df_sd['video'].isin(test_videos)]
    print(X_testSD.emotion.unique()) #check se sono rappresentate tutte le emozioni
    X_testSD.set_index(['id_user','session','video'], inplace=True)
    y_testSD = X_testSD.emotion
    X_testSD = X_testSD.drop(['emotion'],axis=1)
    X_trainSD = df_sd[~df_sd['video'].isin(test_videos)]
    X_trainSD.set_index(['id_user','session','video'], inplace=True)
    y_trainSD = X_trainSD.emotion
    X_trainSD = X_trainSD.drop(['emotion'],axis=1)
    return X_trainSD,y_trainSD,X_testSD,y_testSD

def SubjectDependent_GBC(df):
    X_trainSD, y_trainSD,X_testSD,y_testSD = split_train_test_sd(df)
    clfSD = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_trainSD, y_trainSD)
    print(clfSD.score(X_testSD, y_testSD))


def SubjectIndipendent_GBC(df):
    df_SI = df.reset_index()
    groups = df_SI['id_user']
    X_SI = df_SI.drop(['emotion','id_user','session','video'],axis=1)
    y_SI = df_SI.emotion
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X_SI, y_SI, groups)
    scores = cross_val_score(model, X_SI, y_SI, cv=logo, verbose=1 , groups = groups, n_jobs = -1)
    print(scores.mean())


def SubjectBiasedExperiment(df):
    X_trainSB, X_testSB, y_trainSB, y_testSB = train_test_split(df.drop('emotion',axis=1), df.emotion, test_size=0.20, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_trainSB, y_trainSB)
    print(clf.score(X_testSB, y_testSB))


df = pd.read_csv("dataset_eeg.csv")
df.drop(['Unnamed: 0'],axis=1, inplace=True)
df = df.set_index(['id_user','session','video'])
SubjectDependent_GBC(df)
SubjectIndipendent_GBC(df)
SubjectBiasedExperiment(df)