import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset
from FullyEncodedPerAntenna import FullyEndToEnd as FullField
from field_training import check_preprocessing, evaluate_checkpoint, run_cli


def load_encoder_from_checkpoint(model_path, model, dataset=None):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    mode = checkpoint.get("waveform_mode", "legacy_raw")
    transform = checkpoint.get("preprocessing_id")
    check_preprocessing(dataset, mode, transform)
    state = checkpoint.get("state_dict", checkpoint)
    encoder = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
    model.encoder.load_state_dict(encoder)
    model.checkpoint_waveform_mode = mode
    model.checkpoint_preprocessing_id = transform
    model.checkpoint_training_protocol = checkpoint.get("training_protocol")
    return model


def freeze_encoder(model):
    model.encoder.requires_grad_(False)


class FullyEndToEnd(FullField):
    """Field decoder trained with a frozen, pretrained antenna encoder."""

    def on_fit_start(self):
        super().on_fit_start()
        freeze_encoder(self)


def normal_train(data_path, encoder_checkpoint):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )

    metric = data_module.checkpoint_metric()
    n_antennas, n_directions, n_timesteps = data_module.dataset.sensor_shape
    model = FullyEndToEnd(
        n_antennas=n_antennas,
        n_directions=n_directions,
        n_timesteps=n_timesteps,
        dim_position=3,
        dim_output=3,
        learning_rate=1e-3,
        internal_batch_size_training=64,
        internal_batch_size_validation=64,
    )

    load_encoder_from_checkpoint(encoder_checkpoint, model, data_module.dataset)

    PATH = "FullyDecoder_logs"

    early_stop_callback = EarlyStopping(monitor=metric, min_delta=0.00, patience=25, verbose=True, mode="min")
    logger = TensorBoardLogger(save_dir=PATH, version=f"beamforming_{datetime.datetime.now()}")
    checkpoint_callback = ModelCheckpoint(dirpath=f"{PATH}/{logger.version}", save_top_k=1,
                                          monitor=metric)

    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    logger.log_graph(model)

    trainer.fit(model, datamodule=data_module)


def render(data_path, checkpoint_path, *, visualize=True):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )
    return evaluate_checkpoint(FullyEndToEnd, data_module, checkpoint_path, visualize=visualize, show_permittivity=True)


if __name__ == "__main__":
    run_cli(normal_train, render, pretrained_encoder=True)
