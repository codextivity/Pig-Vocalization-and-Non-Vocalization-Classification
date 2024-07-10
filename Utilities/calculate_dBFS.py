import os
import numpy as np
from pydub import AudioSegment, effects


# This function is used to calculate dBFS with a single file
def single_file(file):
    raw_audio = AudioSegment.from_file(file)
    audio_dbfs = raw_audio.dBFS
    return audio_dbfs


# This function is used to calculate dBFS with multiple files
def multiple_files(file_dir):
    j = 0
    list_dbfs = []
    for count, file_name in enumerate(os.listdir(file_dir)):
        path = file_dir+file_name
        raw_audio = AudioSegment.from_file(path, format='wav')
        audio_dbfs = raw_audio.dBFS
        list_dbfs.append(audio_dbfs)
    return list_dbfs


# This function is used to calculate min, max, and mean of dBFS in the list
def cal_min_max_mean(dbfs_list):
    to_arr_numpy = np.array(dbfs_list)
    min_dbfs = np.min(to_arr_numpy)
    max_dbfs = np.max(to_arr_numpy)
    mean_dbfs = np.mean(to_arr_numpy)
    return min_dbfs, max_dbfs, mean_dbfs


file_path = '../Datasets/TestFile/'
audio_abfs = multiple_files(file_path)
min_dbfs, max_dbfs, mean_dbfs = cal_min_max_mean(audio_abfs)


