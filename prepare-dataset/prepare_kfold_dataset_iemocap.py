import glob
import os
import re
import shutil

import numpy as np
import tqdm

from audio_utils import mfcc_from_audio_file

k_folds_exclusions = {
    1: ['01_M', '02_F'],
    2: ['02_M', '03_F'],
    3: ['03_M', '04_F'],
    4: ['04_M', '05_F'],
    5: ['05_M', '01_F']
}

EMOTIONS = ['hap', 'sad', 'ang', 'neu']


def get_impro_wav_folders(iemocap_path):
    wav_folders = glob.glob(iemocap_path + '/Session*/sentences/wav/*')
    impro_wav_folders = [f for f in wav_folders if "impro" in f]
    processing_ids = []
    processing_wav_folders = []

    for f in impro_wav_folders:
        folder_id = f.split("/")[-1]
        result = re.search("(Ses\d{2})([F|M])_(impro\d{2})", folder_id)
        if result is not None:
            i = f"{result.group(1)}_{result.group(3)}"
            if i not in processing_ids:
                processing_ids.append(i)
                processing_wav_folders.append(f)

    return processing_wav_folders


def get_wav_files(wav_folders):
    wav_files = []
    for f in wav_folders:
        wav = glob.glob(f'{f}/*')
        wav_files.extend(wav)

    return wav_files


def get_speaker(fname):
    sections = fname.split('/')
    sentence_id = sections[-1].split(".")[0]
    result = re.search("Ses(\d{2}).*([F|M])\d{3}", sentence_id)
    return f"{result.group(1)}_{result.group(2)}"


def get_audio_files_per_speaker(wav_files):
    audio_files = {}
    for f in wav_files:
        speaker = get_speaker(f)
        try:
            audio_files[speaker]
        except KeyError:
            audio_files[speaker] = []

        audio_files[speaker].append(f)

    return audio_files


def get_training_files_per_fold(fold, audio_files_per_speaker):
    files = []
    for s in audio_files_per_speaker:
        if s not in k_folds_exclusions[fold]:
            files.extend(audio_files_per_speaker[s])

    return files


def get_val_files_per_fold(fold, audio_files_per_speaker):
    files = []
    for s in audio_files_per_speaker:
        if s in k_folds_exclusions[fold]:
            files.extend(audio_files_per_speaker[s])

    return files


def get_class(fname):
    sections = fname.split('/')
    session_id = sections[-5]
    dialog_id = sections[-2]
    sentence_id = sections[-1].split(".")[0]

    dataset_base_path = str.join("/", fname.split("/")[0:-5])

    emo_evaluation_file = dataset_base_path + '/' + session_id + '/dialog/EmoEvaluation/' + dialog_id + '.txt'
    with open(emo_evaluation_file, 'r') as f:
        targets = [line for line in f if sentence_id in line]
        emo = targets[0].split('\t')[2]
        return emo


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


def main(iemocap_path, destination):
    wav_folders = get_impro_wav_folders(iemocap_path)
    wav_files = get_wav_files(wav_folders)
    files_per_speaker = get_audio_files_per_speaker(wav_files)
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
    IEMOCAP_PATH = "[IEMOCAP_PATH]/IEMOCAP_full_release"
    DESTINATION_PATH = "[IEMOCAP_PATH]/iemocap/processed_mfcc_k_fold"
    main(IEMOCAP_PATH, DESTINATION_PATH)
