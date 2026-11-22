import datetime
import math
from enum import Enum

import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset
from field_training import DEFAULT_DATA_PATH, PointQueryModel, evaluate_checkpoint, run_cli


class PositionalEncodingType(Enum):
    sinusoidal = 1
    learned = 2


class TransformerEncoder(pl.LightningModule):
    def __init__(self,
                 n_antennas: int,
                 n_timesteps: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 position_encoding: PositionalEncodingType = PositionalEncodingType.learned
                 ):
        super().__init__()

        self.encoder_embedding = nn.Linear(n_timesteps, d_model)

        self.set_position_encoding(position_encoding, n_antennas, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def set_position_encoding(self, mode, n_tokens, d_model):
        if hasattr(self, "positional_encoding"):
            del self.positional_encoding
        self.position_encoding = mode
        if mode == PositionalEncodingType.sinusoidal:
            encoding = self.create_sinusoidal_positional_encoding(n_tokens, d_model)
            self.register_buffer("positional_encoding", encoding, persistent=False)
        else:
            self.positional_encoding = nn.Embedding(n_tokens, d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        a = src.size(1)
        src = self.encoder_embedding(src)

        # Add positional encoding
        if self.position_encoding == PositionalEncodingType.sinusoidal:
            src = src + self.positional_encoding[:a, :]
        elif self.position_encoding == PositionalEncodingType.learned:
            src = src + self.positional_encoding(torch.arange(a, device=src.device))

        memory = self.transformer_encoder(src)

        return memory

    def create_sinusoidal_positional_encoding(self, n_timesteps: int, d_model: int) -> torch.Tensor:
        position = torch.arange(n_timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        sinusoidal_encoding = torch.zeros(n_timesteps, d_model)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        return sinusoidal_encoding


class IndependentQueryDecoderLayer(nn.TransformerDecoderLayer):
    """Run each coordinate through self-attention alone, sharing cross-attention memory across coordinates."""

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        if attn_mask is not None or key_padding_mask is not None or is_causal:
            return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)
        singleton = x.reshape(-1, 1, x.size(-1))
        return super()._sa_block(singleton, None, None).reshape_as(x)


class CustomTransformer(PointQueryModel):
    def __init__(self, n_antennas: int, n_timesteps: int, dim_position: int, dim_output: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1,
                 learning_rate: float = 1e-3, internal_batch_size_training: int = 16,
                 internal_batch_size_validation: int = 16,
                 position_encoding: PositionalEncodingType = PositionalEncodingType.learned
                 ):
        super().__init__()
        if isinstance(position_encoding, str):
            position_encoding = PositionalEncodingType[position_encoding]
        self.encoder = TransformerEncoder(n_antennas, n_timesteps, d_model, nhead, num_encoder_layers,
                                          dim_feedforward, dropout, position_encoding)

        # Decoder part
        self.decoder_embedding = nn.Linear(dim_position, d_model)
        decoder_layer = IndependentQueryDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(d_model, dim_output)

        self.lr = learning_rate
        self.internal_batch_size_training = internal_batch_size_training
        self.internal_batch_size_validation = internal_batch_size_validation
        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters(ignore=["position_encoding"])
        self.hparams.position_encoding = position_encoding.name
        self.position_encoding = position_encoding

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if "position_encoding" not in checkpoint.get("hyper_parameters", {}):
            # The archived model saved weights for learned embeddings only.
            has_embedding = "encoder.positional_encoding.weight" in checkpoint["state_dict"]
            mode = PositionalEncodingType.learned if has_embedding else PositionalEncodingType.sinusoidal
            if mode != self.position_encoding:
                self.encoder.set_position_encoding(mode, self.hparams.n_antennas, self.hparams.d_model)
                self.encoder.to(self.decoder_embedding.weight)
            self.position_encoding = mode
            self.hparams.position_encoding = mode.name

    def encode_scene(self, voltages):
        if voltages.ndim == 4:
            voltages = voltages.flatten(1, 2)
        return self.encoder(voltages)

    def decode_queries(self, positions, memory):
        queries = self.decoder_embedding(positions)
        return self.output_layer(self.transformer_decoder(queries, memory))


def normal_train(data_path=DEFAULT_DATA_PATH):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_sensor_samples=True,
        load_subsampled_grids=False,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
    )

    metric = data_module.checkpoint_metric()
    n_antennas, n_directions, n_timesteps = data_module.dataset.sensor_shape
    model = CustomTransformer(n_antennas=n_antennas * n_directions,
                              n_timesteps=n_timesteps,
                              dim_position=3,
                              dim_output=3,
                              d_model=1024,
                              nhead=8,
                              num_encoder_layers=2,
                              num_decoder_layers=4,
                              dim_feedforward=2048,
                              dropout=0.1,
                              learning_rate=1e-6,
                              internal_batch_size_training=64,
                              internal_batch_size_validation=64,
                              position_encoding=PositionalEncodingType.sinusoidal
                              )

    sub_dir = "Sinusoidal" if model.position_encoding == PositionalEncodingType.sinusoidal else "Learned"

    PATH = "TransformerPerAntennaAndDir_logs" + f"/{sub_dir}"

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
        load_sensor_samples=True,
        load_subsampled_grids=False,
        split_sensor_samples=True,
        flatten_sensor_samples=False
    )
    return evaluate_checkpoint(CustomTransformer, data_module, checkpoint_path, visualize=visualize, show_permittivity=True)


if __name__ == "__main__":
    run_cli(normal_train, render)
