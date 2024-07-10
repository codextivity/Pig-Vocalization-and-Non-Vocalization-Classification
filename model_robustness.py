import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Utilities.dataset import load_dataset, extract_features

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPU')
    except RuntimeError as e:
        print(e)


# Dataset directory
dataset_dir = 'Datasets/pig_vocal_nonvocal/'
# Load dataset
features_X, features_Y = load_dataset(dataset_dir)
# Reshape the label feature
features_Y = features_Y.reshape(features_Y.shape[0], 1)

test_inputs, test_list, test_labels = [], [], []
feature_type = 2
sample_rate = 22050
for i in tqdm(range(len(features_X))):
    data_feature = extract_features(feature_type=feature_type,
                                    y=features_X[i], sr=sample_rate)
    test_list.append(data_feature)
    test_labels.append(features_Y[i])
print('Finished feature extraction.')

test_list_to_db = librosa.power_to_db(np.array(test_list), ref=np.max)
for index in tqdm(range(0, test_list.shape[0])):
    feature_scale = min
