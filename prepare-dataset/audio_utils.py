import librosa
import numpy as np

from utils import downsample_with_max_pooling

NUM_MFCC = 128


def load_wav(filename: str, sample_rate: int):
    return librosa.load(filename, sample_rate)


def add_missing_padding(audio, sr, duration):
    signal_length = duration * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio


def split_audio(signal, sr, split_duration):
    length = split_duration * sr

    if length < len(signal):
        frames = librosa.util.frame(signal, frame_length=length, hop_length=length).T
        return frames
    else:
        audio = add_missing_padding(signal, sr, split_duration)
        frames = [audio]
        return np.array(frames)


def spectrogram_from_audio_file(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    audio_segment = split_audio(wav, sr, duration)

    # Use values from Sun H
    stft = np.abs(librosa.stft(audio_segment[0], n_fft=278, hop_length=229))
    downsampled = downsample_with_max_pooling(stft)

    # # get Fast Fourier Transformation
    # # Sun H. et al EmotionNAS, 2022
    # hamming_window_size = int((25 / 1000) * sr)  # 25ms Window length
    # hop_length = int((25 - 14) / 1000 * sr)  # 14ms overlap
    #
    # frame = librosa.util.frame(audio_segment[0], hamming_window_size, hop_length)
    # hamming_w = librosa.filters.get_window("hamming", hamming_window_size)
    # wind_frames = hamming_w.reshape(-1, 1) * frame
    # sp = np.real(fft2(wind_frames, (128, 128)))
    return downsampled

    # return librosa.feature.melspectrogram(y=frames[0], sr=sr, hop_length=506)


def mfcc_from_audio_file(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    audio_segment = split_audio(wav, sr, duration)

    mfcc = librosa.feature.mfcc(y=audio_segment[0], sr=sr, n_mfcc=NUM_MFCC)

    downsampled = downsample_with_max_pooling(mfcc, (1, 4))
    return downsampled
