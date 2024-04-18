import os
import sys
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import json
import random
import torch
from sklearn.preprocessing import LabelEncoder

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

def prepare_data(data_original, save_json_train, save_json_valid, save_json_test, split_ratio=[80, 10, 10], seed=12):
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
                    if os.path.isfile(wav_file):
                        wav_list.append((wav_file, label))
                    else:
                        logger.warning(f"Skipping invalid audio file: {wav_file}")

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

    logger.info(f"Created {save_json_train}, {save_json_valid}, and {save_json_test}")


def create_json(wav_list, json_file):
    json_dict = {}
    for wav_file, label in wav_list:
        signal = sb.dataio.dataio.read_audio(wav_file)
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


class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + emotion classifier."""
        wavs, lengths = batch.sig, batch.length
        outputs = self.modules.wav2vec2(wavs, lengths)
        outputs = self.hparams.avg_pool(outputs, lengths)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using emotion ID as label."""
        emo_id = batch.emo_encoded
        loss = self.hparams.compute_cost(predictions, emo_id)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emo_id)
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        self.loss_metric = sb.utils.metric_stats.MetricStats(metric=sb.nnet.losses.nll_loss)
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }
            if stage == sb.Stage.VALID:
                old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch, "lr": old_lr},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stats,
                )
                self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error_rate"])
            if stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=stats,
                )

    def init_optimizers(self):
        """Initializes the optimizer."""
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
        self.optimizers_dict = {"model_optimizer": self.optimizer}


label_encoder = LabelEncoder()


def dataio_prep(hparams):
    """Prepares the datasets to be used in the brain class."""
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.transform([emo])[0]
        yield emo_encoded

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal from a WAV file."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_original"]},
            output_keys=["id", "wav", "label"],
        )

    all_labels = [data_item["label"] for dataset in datasets.values() for data_item in datasets[dataset]]
    label_encoder.fit(all_labels)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    with open(lab_enc_file, "w") as f:
        json.dump({"classes": label_encoder.classes_.tolist()}, f)

    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    try:
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        data_original = hparams.get("data_original")
        if data_original is not None:
            data_original = os.path.normpath(data_original)
            if not os.path.exists(data_original):
                raise ValueError(f"data_original path '{data_original}' does not exist.")
        else:
            raise ValueError("data_original path is not specified in the YAML configuration.")

    except Exception as e:
        print("Error occurred", e)
        sys.exit(1)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not hparams["skip_prep"]:
        prepare_kwargs = {
            "data_original": hparams["data_original"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
        }
        sb.utils.distributed.run_on_main(prepare_data, kwargs=prepare_kwargs)

    datasets = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to(device=run_opts["device"])
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()

    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
