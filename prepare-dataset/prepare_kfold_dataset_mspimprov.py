import glob
import os
import shutil

import numpy as np
import tqdm

from audio_utils import mfcc_from_audio_file

k_folds_exclusions = {
    1: ['session1'],
    2: ['session2'],
    3: ['session3'],
    4: ['session5']
}

EMOTIONS = ['hap', 'sad', 'ang', 'neu']

emo_map = {
    'S': 'sad',
    'H': 'hap',
    'A': 'ang',
    'N': 'neu'
}


def get_wav_files(root_folder):
    return glob.glob(f"{root_folder}/*/*/R/*.wav")


def get_session(fname):
    sections = fname.split('/')
    session = sections[-4]
    return session


def get_audio_files_per_session(wav_files):
    audio_files = {}
    for f in wav_files:
        session = get_session(f)
        try:
            audio_files[session]
        except KeyError:
            audio_files[session] = []

        audio_files[session].append(f)

    return audio_files


def get_training_files_per_fold(fold, audio_files_per_session):
    files = []
    for s in audio_files_per_session:
        if s not in k_folds_exclusions[fold]:
            files.extend(audio_files_per_session[s])

    return files


def get_val_files_per_fold(fold, audio_files_per_session):
    files = []
    for s in audio_files_per_session:
        if s in k_folds_exclusions[fold]:
            files.extend(audio_files_per_session[s])

    return files


def get_class(fname):
    sections = fname.split('/')
    e = sections[-1].split('-')[-4][-1]
    emo_i = emo_map[e]
    if emo_i in EMOTIONS:
        return emo_i
    else:
        return None


def copy_file(src_file_path, destination_base_path):
    c = get_class(src_file_path)
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("/")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy(src_file_path, d)


def save_mfcc(src_file_path, destination_base_path, sample_rate, utterance_duration):
    spectrogram = mfcc_from_audio_file(src_file_path, sample_rate, utterance_duration)

    c = get_class(src_file_path)
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("/")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    with open(f"{d}.npy", "wb") as f:
        np.save(f, spectrogram)


def main(source_path, destination):
    wav_files = get_wav_files(source_path)
    files_per_speaker = get_audio_files_per_session(wav_files)
    for fold in k_folds_exclusions:
        destination_base_path = f"{destination}/{fold}"
        print(f"Processing fold {fold}")
        training_files = get_training_files_per_fold(fold, files_per_speaker)
        for f in tqdm.tqdm(training_files):
            c = get_class(f)
            if c in EMOTIONS:
                copy_file(f, destination_base_path + "/raw/train")
                save_mfcc(f, destination_base_path + "/mfcc/train", 32750, 8)

        val_files = get_val_files_per_fold(fold, files_per_speaker)
        for f in tqdm.tqdm(val_files):
            c = get_class(f)
            if c in EMOTIONS:
                copy_file(f, destination_base_path + "/raw/val")
                save_mfcc(f, destination_base_path + "/mfcc/val", 32750, 8)


if __name__ == "__main__":
    SOURCE_PATH = "[MSP-IMPROV_PATH]/MSP-IMPROV/all_sessions"
    DESTINATION_PATH = "[MSP-IMPROV_PATH]/MSP-IMPROV/min/processed_mfcc_k_fold"
    main(SOURCE_PATH, DESTINATION_PATH)
