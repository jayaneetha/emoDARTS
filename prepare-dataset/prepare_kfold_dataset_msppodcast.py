import os
import shutil
from multiprocessing import Process

import numpy as np
import pandas as pd
import tqdm

from audio_utils import mfcc_from_audio_file

k_folds_exclusions = {
    1: [4],
    2: [3],
    3: [2],
    4: [1],
    5: [0]
}

EMOTIONS = ['hap', 'sad', 'ang', 'neu']

emo_map = {
    'S': 'sad',
    'H': 'hap',
    'A': 'ang',
    'N': 'neu'
}


def get_training_files_per_fold(fold, groups_speakers, df):
    files = list()
    for f in groups_speakers:
        if f not in k_folds_exclusions[fold]:
            fls = list(df[df['SpkrID'].isin(groups_speakers[f])]['FileName'])
            files = files + fls

    return files


def get_val_files_per_fold(fold, groups_speakers, df):
    files = list()
    for f in groups_speakers:
        if f in k_folds_exclusions[fold]:
            fls = list(df[df['SpkrID'].isin(groups_speakers[f])]['FileName'])
            files = files + fls

    return files


def get_class(fname, df):
    f = fname.split("/")[-1]
    e = df[df['FileName'] == f]['EmoClass'].values[0]
    emo_i = emo_map[e]
    if emo_i in EMOTIONS:
        return emo_i
    else:
        return None


def copy_file(src_file_path, destination_base_path, c):
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("/")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy(src_file_path, d)


def save_mfcc(src_file_path, destination_base_path, sample_rate, utterance_duration, c):
    spectrogram = mfcc_from_audio_file(src_file_path, sample_rate, utterance_duration)
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("/")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    with open(f"{d}.npy", "wb") as f:
        np.save(f, spectrogram)


def main(source_path, destination):
    consensus_df = pd.read_csv(f"{source_path}/labels_consensus.csv")

    ashn_df = consensus_df[consensus_df['EmoClass'].isin(emo_map.keys())]
    ashn_df['Split_Set'] = np.where(ashn_df['Split_Set'] == "Test1", "Train", ashn_df['Split_Set'])
    ashn_df['Split_Set'] = np.where(ashn_df['Split_Set'] == "Test2", "Train", ashn_df['Split_Set'])
    ashn_df = ashn_df[ashn_df['SpkrID'] != 'Unknown']

    SPLIT = "Train"

    df = ashn_df[ashn_df['Split_Set'] == SPLIT]
    df = df.sort_values(by='SpkrID')
    grouped = df.groupby('SpkrID')
    num_groups = 5
    group_size = len(df) // num_groups

    groups_speakers = {}
    groups_count = {}
    g = 0
    for key, group in grouped:
        try:
            c = groups_count[g]
        except KeyError:
            c = 0
            groups_speakers[g] = []

        c = c + len(group)
        groups_count[g] = c
        groups_speakers[g].append(key)

        if c > group_size:
            g = g + 1

    processes = {}
    for fold in k_folds_exclusions:
        processes[fold] = Process(target=process_fold, args=(destination, df, fold, groups_speakers, source_path))
        processes[fold].start()
        # process_fold(destination, df, fold, groups_speakers, source_path)

    for fold in k_folds_exclusions:
        processes[fold].join()


def process_fold(destination, df, fold, groups_speakers, source_path):
    destination_base_path = f"{destination}/{fold}"
    print(f"Processing fold {fold}")
    training_files = get_training_files_per_fold(fold, groups_speakers, df)
    for f in tqdm.tqdm(training_files):
        f = f"{source_path}/Audio/{f}"
        c = get_class(f, df)
        if c in EMOTIONS:
            try:
                copy_file(f, destination_base_path + "/raw/train", c)
                save_mfcc(f, destination_base_path + "/mfcc/train", 32750, 8, c)
            except FileNotFoundError as e:
                print(str(e))
    val_files = get_val_files_per_fold(fold, groups_speakers, df)
    for f in tqdm.tqdm(val_files):
        f = f"{source_path}/Audio/{f}"
        c = get_class(f, df)
        if c in EMOTIONS:
            try:
                copy_file(f, destination_base_path + "/raw/val", c)
                save_mfcc(f, destination_base_path + "/mfcc/val", 32750, 8, c)
            except FileNotFoundError as e:
                print(str(e))


if __name__ == "__main__":
    SOURCE_PATH = "[MSP-Podcast_PATH]/MSP-Podcast"
    DESTINATION_PATH = "[MSP-Podcast_PATH]/MSP-Podcast/processed_mfcc_k_fold"
    main(SOURCE_PATH, DESTINATION_PATH)
