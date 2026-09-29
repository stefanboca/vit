from typing import cast, override

import timm
import torch
from lightning import LightningModule
from timm.models import Eva
from tokenizers import Tokenizer
from torch import Tensor, nn

from vit.utils.state_dict import StateDictExcludeModulesMixin


class OcrVit(LightningModule, StateDictExcludeModulesMixin):
    def __init__(
        self,
        backbone: str = "vit_small_plus_patch16_dinov3",
        n_head=6,
        n_blocks=6,
    ):
        super().__init__()

        self.backbone = cast(Eva, timm.create_model(backbone, pretrained=True))
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.tokenizer = Tokenizer.from_pretrained("google/flan-t5-base")
        self.embedding = nn.Embedding(
            self.tokenizer.get_vocab_size(with_added_tokens=True),
            self.backbone.embed_dim,
            max_norm=1.0,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.backbone.embed_dim,
                nhead=n_head,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=n_blocks,
            norm=nn.LayerNorm(self.backbone.embed_dim),
        )

        self._exclude_modules_from_state_dict("backbone")

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @override
    def training_step(self, batch, _batch_idx):
        imgs, labels = batch
        with torch.no_grad():
            memory = self.backbone.forward(imgs)
        tgt, labels = self._tokenize_batch(labels)
        # TODO: handle tgts to tensor, and mismatching seq lengths
        tgt = self.embedding.forward(tgt)
        features = self.decoder.forward(tgt, memory, tgt_is_causal=True)
        out = nn.functional.linear(features, self.embedding.weight)
        # TODO: cross-entropy loss between out and labels

    @override
    def validation_step(self, batch, _batch_idx):
        pass

    def _tokenize_batch(
        self, text: list[str]
    ) -> tuple[list[list[int]], list[list[int]]]:
        encoded = self.tokenizer.encode_batch_fast(text, add_special_tokens=True)
        labels = [e.ids for e in encoded]
        pad_id = self.tokenizer.token_to_id("<pad>")
        assert pad_id is not None
        tgts = [[pad_id] + label[:-1] for label in labels]
        return tgts, labels
