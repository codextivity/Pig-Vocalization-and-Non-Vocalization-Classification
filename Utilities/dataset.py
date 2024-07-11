import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from Utilities.audio_augmentation import pitching, nosing, shifting, stretching


def load_dataset(dataset_dir):
    data_x, data_y = [], []
    for count, file_name in enumerate(os.listdir(dataset_dir)):
        labels = file_name.split('-')[1]
        file_path = dataset_dir+file_name
        y, sample = librosa.load(file_path)
        data_x.append(y)
        data_y.append(labels)
    print('Features are extracted successfully.')

    return np.array(data_x), np.array(data_y)


def extract_features(feature_type, y, sr):
    if feature_type == 1:
        mel_features = np.array(librosa.feature.mfcc(y=y, sr=sr))
    elif feature_type == 2:
        mel_features = np.array(librosa.feature.melspectrogram(y=y, sr=sr))
    elif feature_type == 3:
        mel_features = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
    elif feature_type == 4:
        mel_features = np.array(librosa.feature.tonnetz(y=y, sr=sr))
    elif feature_type == 5:
        mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
        melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr))
        chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
        tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=sr))

        mel_features = np.concatenate((mfcc, melspectrogram, chroma, tonnetz))
    else:
        print("Enter number from 1-5 to produce feature type")
    return mel_features


'''
These are feature type
# => feature_type param receives int value as below
1: MFCC
2: MelSpectrogram
3: Chroma
4: Tonnetz
5: MMCT

# => Use default sample_rate value in librosa
# Default sample_rate = 22050
'''


def dataset_augmentation(X_input, y_input, feature_type=2, aug=True):
    data_values, data_list, data_labels = [], [], []
    minmax_scaler = preprocessing.MinMaxScaler()
    sample_rate = 22050
    for i in tqdm(range(len(X_input))):
        data_feature = extract_features(feature_type=feature_type,
                                        y=X_input[i], sr=sample_rate)
        data_list.append(data_feature)
        data_labels.append(y_input[i])
        if aug is True:
            # Data augmentation
            audio_pitching = pitching(X_input[i])
            audio_pitching = extract_features(feature_type=feature_type,
                                              y=audio_pitching, sr=sample_rate)
            data_list.append(audio_pitching)
            data_labels.append(y_input[i])

            audio_noising = nosing(X_input[i])
            audio_noising = extract_features(feature_type=feature_type,
                                             y=audio_noising, sr=sample_rate)
            data_list.append(audio_noising)
            data_labels.append(y_input[i])

            audio_time_shift = shifting(X_input[i])
            audio_time_shift = extract_features(feature_type=feature_type,
                                                y=audio_time_shift, sr=sample_rate)
            data_list.append(audio_time_shift)
            data_labels.append(y_input[i])

            audio_time_stretching = stretching(X_input[i])
            audio_time_stretching = extract_features(feature_type=feature_type,
                                                     y=audio_time_stretching, sr=sample_rate)

            data_list.append(audio_time_stretching)
            data_labels.append(y_input[i])

    print('Finished data augmentation and feature scaling of training set.')
    data_list_to_db = librosa.power_to_db(np.array(data_list), ref=np.max)
    # Scale feature into smaller value
    for index in tqdm(range(0, data_list_to_db.shape[0])):
        scale_feature = minmax_scaler.fit_transform(data_list_to_db[index, :, :])
        data_values.append(scale_feature)
    print('Finished feature scaling of training set')
    data_X, data_Y = (np.array(data_values), np.array(data_labels))
    # Prepare dataset shape and for model
    data_X = np.array(data_X)
    data_X = data_X.reshape(data_X.shape[0], data_X.shape[1], data_X.shape[2], 1)
    data_Y = np.squeeze(data_Y, axis=1)
    data_Y = pd.get_dummies(data_Y).to_numpy(dtype='long')

    return data_X, data_Y




