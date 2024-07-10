import librosa
import numpy as np


def pitching(sample):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(sample.astype('float64'), sr=22050, n_steps=pitch_change, bins_per_octave=bins_per_octave)
    return y_pitch


def nosing(sample):
    nosing_amp = 0.5*np.random.uniform()*np.amax(sample)
    y_nose = sample.astype('float64') + nosing_amp * np.random.normal(size=sample.shape[0])
    return y_nose


def shifting(sample):
    time_shift_fac = 0.2 * 2 * (np.random.uniform()-0.5)
    start = int(sample.shape[0] * time_shift_fac)

    if start > 0:
        y_shift = np.pad(sample, (start, 0), mode='constant')[0:sample.shape[0]]
    else:
        y_shift = np.pad(sample, (0, -start), mode='constant')[0:sample.shape[0]]

    return y_shift


def stretching(sample):
    input_length = len(sample)
    y_stretch = librosa.effects.time_stretch(sample.astype('float'), rate=1.1)
    if len(y_stretch) > input_length:
        y_stretch = y_stretch[:input_length]
    else:
        y_stretch = np.pad(y_stretch, (0, max(0, input_length - len(y_stretch))), 'constant')

    return y_stretch
