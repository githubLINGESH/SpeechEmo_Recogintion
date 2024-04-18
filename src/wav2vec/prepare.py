import os
import json
import random
import logging
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_data(
    data_original,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
    seed=12,
):
    # Setting seeds for reproducible code.
    random.seed(seed)

    # Check if data preparation has already been done (skip if files exist)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # Collect audio files and labels
    wav_list = []
    labels = os.listdir(data_original)

    for label in labels:
        label_dir = os.path.join(data_original, label)
        if os.path.isdir(label_dir):
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    wav_file = os.path.join(label_dir, audio_file)
                    wav_list.append((wav_file, label))

    # Shuffle and split the data
    random.shuffle(wav_list)
    n_total = len(wav_list)
    n_train = n_total * split_ratio[0] // 100
    n_valid = n_total * split_ratio[1] // 100

    train_set = wav_list[:n_train]
    valid_set = wav_list[n_train:n_train + n_valid]
    test_set = wav_list[n_train + n_valid:]

    # Create JSON files for train, valid, and test sets
    create_json(train_set, save_json_train)
    create_json(valid_set, save_json_valid)
    create_json(test_set, save_json_test)

    logger.info(
        f"Created {save_json_train}, {save_json_valid}, and {save_json_test}"
    )


def create_json(wav_list, json_file):
    json_dict = {}
    for wav_file, label in wav_list:
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE
        uttid = os.path.splitext(os.path.basename(wav_file))[0]

        json_dict[uttid] = {
            "wav": wav_file,
            "length": duration,
            "label": label,
        }

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"Created {json_file}")


def skip(*filenames):
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True
