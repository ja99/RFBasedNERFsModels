import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset
from FullySpereatedEncodedPerDir import FullyEndToEnd as FullField
from field_training import DEFAULT_DATA_PATH, evaluate_checkpoint, run_cli


class FullyEndToEnd(FullField):
    target_channel = 1


def normal_train(data_path=DEFAULT_DATA_PATH):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=False,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=True,
    )

    metric = data_module.checkpoint_metric()
    n_antennas, n_directions, n_timesteps = data_module.dataset.sensor_shape
    model = FullyEndToEnd(
        n_antennas=n_antennas,
        n_directions=n_directions,
        n_timesteps=n_timesteps,
        dim_position=3,
        dim_output=1,
        learning_rate=1e-6,
        internal_batch_size_training=64,
        internal_batch_size_validation=64,
    )

    PATH = "FullySimplifiedSpereatedEncodedPerDir_logs"

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
        load_normal_grids=False,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=True,
    )
    return evaluate_checkpoint(FullyEndToEnd, data_module, checkpoint_path, visualize=visualize, show_permittivity=False)


if __name__ == "__main__":
    run_cli(normal_train, render)
