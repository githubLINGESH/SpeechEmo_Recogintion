import os
import sys
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from prepare import prepare_data
import json

logger = logging.getLogger(__name__)


class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.wav2vec2(wavs, lens)

        # last dim will be used for StatisticsPooling
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)

        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using emotion ID as label."""
        emo_id, _ = batch.emo_encoded

        # to meet the input form of nll loss
        emo_id = emo_id.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emo_id)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emo_id)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Store the train loss until the validation stage
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

            # At the end of validation
            if stage == sb.Stage.VALID:
                old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

                # Logging training and validation stats
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch, "lr": old_lr},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stats,
                )

                # Save the current checkpoint and delete previous checkpoints
                self.checkpointer.save_and_keep_only(
                    meta=stats, min_keys=["error_rate"]
                )

            # Logging test data stats
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


def dataio_prep(hparams):
    """Prepares the datasets to be used in the brain class."""
    try:
        # Ensure dataset preparation is completed
        prepare_kwargs = {
            "data_original": hparams["data_original"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": hparams["split_ratio"],
            "seed": hparams["seed"],
        }
        sb.utils.distributed.run_on_main(prepare_data, kwargs=prepare_kwargs)

        # Load or create label encoder
        lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
        label_encoder = {}

        # Check if the label encoder file exists and is non-empty
        if os.path.isfile(lab_enc_file) and os.path.getsize(lab_enc_file) > 0:
            # Load the existing label encoder
            with open(lab_enc_file, "r") as f:
                label_encoder = json.load(f)
        else:
            # Create a new label encoder from the training dataset
            train_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["train_annotation"],
                replacements={"data_root": hparams["data_original"]},
                dynamic_items=[],
                output_keys=["id", "label"],
            )

            # Extract unique labels from the "label" column
            labels = get_unique_labels(train_dataset, "label")
            label_encoder = {label: idx for idx, label in enumerate(labels)}

            # Save label encoder to file
            with open(lab_enc_file, "w") as f:
                json.dump(label_encoder, f)

        # Define data processing pipelines
        @sb.utils.data_pipeline.takes("wav")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav):
            sig = sb.dataio.dataio.read_audio(wav)
            return sig

        @sb.utils.data_pipeline.takes("label")
        @sb.utils.data_pipeline.provides("label", "emo_encoded")
        def label_pipeline(label):
            if isinstance(label, str) and label in label_encoder:
                yield label
                emo_encoded = label_encoder[label]
                yield emo_encoded
            else:
                logger.warning(f"Unexpected 'label' value: {label}")
                yield None, None

        # Create dataset objects
        data_info = {
            "train": hparams["train_annotation"],
            "valid": hparams["valid_annotation"],
            "test": hparams["test_annotation"],
        }
        datasets = {}
        for dataset_name, json_path in data_info.items():
            datasets[dataset_name] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=json_path,
                replacements={"data_root": hparams["data_original"]},
                dynamic_items=[audio_pipeline, label_pipeline],
                output_keys=["id", "sig", "emo_encoded", "label"],
            )

        return datasets

    except Exception as e:
        # Handle any exceptions and log the error
        logger.error(f"Error during dataset preparation: {e}", exc_info=True)
        raise

    
def get_unique_labels(dataset, column_name):
    """Extracts unique labels from the specified column in the dataset."""
    unique_labels = set()
    
    for sample in dataset:
        # Extract the label directly from the dataset sample
        label = sample[column_name]
        
        if label is not None:
            unique_labels.add(label)
        else:
            logger.warning(f"'{column_name}' not found in data_point: {sample}")
            # Log additional information about the sample for debugging
            logger.debug(f"Sample content: {sample}")
    
    return list(unique_labels)


if __name__ == "__main__":
    # Parse command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize distributed training
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    data_original = hparams.get("data_original")
    print("Data_path",data_original)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
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

    # Data loading and preprocessing
    datasets = dataio_prep(hparams)

    # Initialize brain object on CPU
    def initialize_brain():
        brain = EmoIdBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        # Move the brain to CPU device
        brain.to('cpu')
        return brain

    with sb.utils.distributed.run_on_main(initialize_brain):
        emo_id_brain = initialize_brain()

    # Training loop
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Evaluate on test set
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
