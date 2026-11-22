import datetime
from pathlib import Path
from typing import List
import warnings

import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import CustomDataset
from field_training import DEFAULT_DATA_PATH, PointQueryModel, evaluate_checkpoint, run_cli


class AntennaEncoderModule(nn.Module):
    def __init__(self,
                 sensor_samples_size: int = 1000,  # n_antenna*n_sample
                 dims_hidden: List[int] = [256],
                 activation=nn.GELU,
                 ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(sensor_samples_size, dims_hidden[0]))
        layers.append(activation())
        for i in range(1, len(dims_hidden)):
            layers.append(nn.Linear(dims_hidden[i - 1], dims_hidden[i]))
            layers.append(activation())

        self.encode = nn.Sequential(*layers)

    def forward(self,
                inputs: torch.Tensor,  # [B, n_antennas*n_samples]
                ) -> torch.Tensor:  # [B, dims_hidden[-1]]
        latent = self.encode(inputs)
        return latent


class FullyNetwork(PointQueryModel):
    def __init__(self,
                 n_antennas: int | None = None,
                 n_timesteps: int | None = None,
                 encoder_dims_hidden: List[int] = [256],
                 position_dim_hidden: int = 256,
                 own_dims_hidden: List[int] = [256],
                 activation=nn.GELU,
                 learning_rate: float = 1e-6,
                 internal_batch_size: int = 100,
                 internal_batch_size_validation: int | None = None,
                 output_activation: bool = False,
                 ):
        super().__init__()

        # Store the name so ordinary torch checkpoints contain only basic types.
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.save_hyperparameters(ignore=["activation"])
        self.hparams.activation = activation.__name__ if getattr(nn, activation.__name__, None) is activation else activation
        self._activation = activation
        self.antenna_encoder = None
        if n_antennas is not None and n_timesteps is not None:
            self._build_layers()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = learning_rate
        self.internal_batch_size_training = internal_batch_size
        self.internal_batch_size_validation = (internal_batch_size if internal_batch_size_validation is None
                                               else internal_batch_size_validation)

    def _build_layers(self):
        hp = self.hparams
        self.antenna_encoder = AntennaEncoderModule(
            sensor_samples_size=hp.n_antennas * hp.n_timesteps,
            dims_hidden=hp.encoder_dims_hidden,
            activation=self._activation,
        )
        self.position_encoder = nn.Sequential(nn.Linear(3, hp.position_dim_hidden), self._activation())
        width = hp.encoder_dims_hidden[-1] + hp.position_dim_hidden
        layers = []
        for hidden in hp.own_dims_hidden:
            layers.extend((nn.Linear(width, hidden), self._activation()))
            width = hidden
        layers.extend((nn.Linear(width, 3), self._activation() if hp.output_activation else nn.Identity()))
        self.decode = nn.Sequential(*layers)

    def encode_scene(self, voltages):
        if self.antenna_encoder is None:
            raise ValueError("Supply n_antennas and n_timesteps for a new model, or load an existing checkpoint")
        return self.antenna_encoder(voltages)

    def decode_queries(self, positions, scene):
        scene = scene.unsqueeze(1).expand(-1, positions.size(1), -1)
        coords = self.position_encoder(positions)
        return self.decode(torch.cat((scene, coords), dim=-1))

    def forward(self, inputs, position=None):
        if position is None:
            return super().forward(inputs)
        if position.ndim == 2:
            # Existing callers provide one coordinate beside each voltage row.
            return self.predict_queries(position.unsqueeze(1), inputs).squeeze(1)
        return self.predict_queries(position, inputs)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        saved = checkpoint.get("hyper_parameters", {})
        if "output_activation" not in saved:
            self.hparams.output_activation = True
            if self.antenna_encoder is not None:
                self.decode[-1] = self._activation()
        if "state_dict" not in checkpoint or all(key in saved for key in (
                "n_antennas", "n_timesteps", "encoder_dims_hidden", "position_dim_hidden", "own_dims_hidden")):
            return

        # The original FullyINR never called save_hyperparameters. Recover the linear layers before loading weights.
        state = checkpoint["state_dict"]
        def linear_shapes(prefix):
            matrices = [(int(key.split(".")[-2]), weight.shape) for key, weight in state.items()
                        if key.startswith(prefix + ".") and key.endswith(".weight") and weight.ndim == 2]
            return [shape for _, shape in sorted(matrices)]

        encoder = linear_shapes("antenna_encoder.encode")
        decoder = linear_shapes("decode")
        width = encoder[0][1]
        # Only the product of antenna rows and time samples enters this architecture.
        antennas = self.hparams.n_antennas or (width // self.hparams.n_timesteps if self.hparams.n_timesteps else 1)
        timesteps = self.hparams.n_timesteps or width // antennas
        if antennas * timesteps != width:
            raise ValueError(f"Archival RF encoder requires {width} flattened input values")
        self.hparams.update(n_antennas=antennas, n_timesteps=timesteps,
                            encoder_dims_hidden=[s[0] for s in encoder], own_dims_hidden=[s[0] for s in decoder[:-1]],
                            position_dim_hidden=state["position_encoder.0.weight"].size(0))
        if "activation" not in saved:
            warnings.warn(f"Archived FullyINR checkpoint omits its activation. Using {self._activation.__name__}. "
                          "The thesis entry point used GELU. Override with activation=... or --activation.",
                          stacklevel=2)
        current = self.state_dict()
        if any(key not in current or current[key].shape != tensor.shape for key, tensor in state.items()):
            self._build_layers()
            self.to(device=self.device, dtype=self.dtype)


def normal_train(data_path=DEFAULT_DATA_PATH, *, compact=False, max_epochs=None, output_dir=Path("FullyINR_logs")):
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_sensor_samples=True,
        load_subsampled_grids=False,
        flatten_sensor_samples=True
    )
    metric = data_module.checkpoint_metric()
    n_antennas, n_directions, n_timesteps = data_module.dataset.sensor_shape
    model = FullyNetwork(
        n_antennas=n_antennas * n_directions,
        n_timesteps=n_timesteps,
        encoder_dims_hidden=[128, 64] if compact else [16384, 8192, 4096, 2048],
        position_dim_hidden=32 if compact else 256,
        own_dims_hidden=[128, 64] if compact else [2048, 1024, 512, 256],
        activation=nn.GELU,
        learning_rate=1e-3 if compact else 1e-6,
        internal_batch_size=256 if compact else 100,
        internal_batch_size_validation=1024 if compact else 100,
    )

    early_stop_callback = EarlyStopping(monitor=metric, min_delta=0.00, patience=25, verbose=True, mode="min")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logger = TensorBoardLogger(save_dir=output_dir, name="compact" if compact else "beamforming", version=stamp)
    checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, filename="best" if compact else None,
                                          save_top_k=1 if compact else 2, enable_version_counter=not compact,
                                          monitor=metric)

    trainer = pl.Trainer(
        max_epochs=max_epochs if max_epochs is not None else (100 if compact else 1000),
        deterministic=compact,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)
    print(f"Best checkpoint ({metric}): {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def render(data_path, checkpoint_path, *, visualize=True, activation=None):
    data_module = CustomDataset.SensorDataModule(
        data_path, batch_size=1, load_positions=True, load_normal_grids=True,
        load_subsampled_grids=False, flatten_sensor_samples=True,
    )
    overrides = {"activation": activation} if activation else {}
    return evaluate_checkpoint(FullyNetwork, data_module, checkpoint_path, visualize=visualize,
                               checkpoint_options=overrides)


if __name__ == "__main__":
    def configure_train(parser):
        parser.add_argument("--compact", action="store_true", help="Use the small RF encoder and coordinate decoder")
        parser.add_argument("--max-epochs", type=int, help="Override the training epoch limit")
        parser.add_argument("--output-dir", type=Path, default=Path("FullyINR_logs"),
                            help="Parent directory for TensorBoard logs and best.ckpt")

    def configure_evaluation(parser):
        parser.add_argument("--activation", help="Override the torch.nn activation (historical default GELU)")

    run_cli(normal_train, render, configure_train=configure_train, configure_evaluation=configure_evaluation)
