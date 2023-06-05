import pandas
import numpy as np

def subject_dependent_split(df,session):
    df = df.reset_index()
    np.random.seed(75)
    test_videos = np.random.choice(np.arange(1, 25), replace=False, size=(8))
    df_sess = df.loc[df['session'] == session]
    X_test = df_sess[df_sess['video'].isin(test_videos)].set_index(['id_user','session','video']).drop('emotion',axis=1)
    y_test = df_sess[df_sess['video'].isin(test_videos)].set_index(['id_user','session','video']).emotion
    X_train = df_sess[~df_sess['video'].isin(test_videos)].set_index(['id_user','session','video']).drop('emotion',axis=1)
    y_train = df_sess[~df_sess['video'].isin(test_videos)].set_index(['id_user','session','video']).emotion
    return X_train, X_test, y_train, y_test

def subject_independent_split(df,session):
    df = df.reset_index()
    df_sess = df.loc[df['session'] == session]
    groups = df_sess['id_user']
    X = df_sess.set_index(['id_user','session','video']).drop('emotion',axis=1)
    y = df_sess.set_index(['id_user','session','video']).emotion
    return X, y, groups