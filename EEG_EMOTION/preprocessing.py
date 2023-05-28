from scipy import signal


# Metodo utilizzato per eliminare i dati relativi a sensori non interessanti per emotion recognition
def drop_channels(df, channel_order):
    df = df.set_index(channel_order[0])
    df = df.transpose()
    df = df[['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8']]
    df = df.transpose()
    return df


# Metodo che effettua downsampling del segnale per eliminare rumore
def down_sampling(data):
    sfreq = 1000
    new_sfreq = 200

    data = signal.resample(data, int(len(data) * new_sfreq / sfreq))
    return data


# Metodo che applica un un filtro passa banda sul segnale per filtrare le informazioni
def band_pass_filter(data):
    sfreq = 200
    nyquist_frequency = sfreq / 2
    lowcut = 0.3
    highcut = 50
    order = 4

    b, a = signal.butter(order, [lowcut / nyquist_frequency, highcut / nyquist_frequency], btype='band')
    data = signal.filtfilt(b, a, data, padlen=0)
    return data
