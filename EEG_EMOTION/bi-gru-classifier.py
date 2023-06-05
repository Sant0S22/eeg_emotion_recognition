from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Masking, Bidirectional
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import dataset_split


def preprocessing_for_gru(X_train, X_test, y_train, y_test):
    X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train = to_categorical(y_train)
    X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test


def subject_dependent(df):
    accuracy_test = []
    model = Sequential()
    for i in range(1, 4):
        X_trainSD, X_testSD, y_trainSD, y_testSD = dataset_split.subject_dependent_split(df, i)
        X_trainSD, X_testSD, y_trainSD, y_testSD = preprocessing_for_gru(X_trainSD, X_testSD, y_trainSD, y_testSD)
        del model
        model = Sequential()
        # model.add(Masking(mask_value=-10,input_shape=(X_trainSD.shape[1], 1)))
        model.add(Bidirectional(GRU(units=128, input_shape=(X_trainSD.shape[1], 1))))
        model.add(Dense(units=4, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_trainSD, y_trainSD, epochs=300, batch_size=64)
        loss, accuracy = model.evaluate(X_testSD, y_testSD)
        accuracy_test.append(accuracy)
    print("Accuracy: ", np.array(accuracy_test).mean())


def subject_biased(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('emotion', axis=1), df.emotion, test_size=0.20, random_state=44)
    X_train, X_test, y_train, y_test = preprocessing_for_gru(X_train, X_test, y_train, y_test)
    model = Sequential()
    # model.add(Masking(mask_value=-10,input_shape=(X_train.shape[1], 1)))
    model.add(Bidirectional(GRU(units=32, input_shape=(X_train.shape[1], 1))))
    model.add(Dense(units=4, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200, batch_size=16)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy", accuracy)


def subject_indipendent(df):
    X1, y1, groups1 = dataset_split.subject_independent_split(df, 1)
    X2, y2, groups2 = dataset_split.subject_independent_split(df, 2)
    X3, y3, groups3 = dataset_split.subject_independent_split(df, 3)
    train = [X1, X2, X3]
    targets = [y1, y2, y3]
    groups = [groups1, groups2, groups3]
    logo = LeaveOneGroupOut()
    accuracy_all = []
    for n in range(0, 3):
        X = train[n]
        y = targets[n]
        group = groups[n]
        for i, (train_index, test_index) in enumerate(logo.split(X, y, group)):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            X_train, X_test, y_train, y_test = preprocessing_for_gru(X_train, X_test, y_train, y_test)

            model = Sequential()
            # model.add(Masking(mask_value=-10,input_shape=(X_train.shape[1], 1)))
            model.add(Bidirectional(GRU(units=32, input_shape=(X_train.shape[1], 1))))
            model.add(Dense(units=4, activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=200, batch_size=64)
            loss, accuracy = model.evaluate(X_test, y_test)
            accuracy_all.append(accuracy)
    print("Accuracy: ",  np.array(accuracy).mean() )



#df = pd.read_csv("CSV/dataset_de_LDS_SEEDIV.csv")
#df = pd.read_csv("CSV/dataset_eeg_de_stand.csv")
df = pd.read_csv("CSV/dataset_eeg_nseg.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df = df.set_index(['id_user', 'session', 'video'])
df[df.isnull() == True] = -10
subject_dependent(df)
subject_indipendent(df)
subject_biased(df)
