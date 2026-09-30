from typing import cast, override

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from timm.models import Eva
from tokenizers import Tokenizer
from torch import Tensor, nn

from vit.nn import RotaryTransformerDecoder
from vit.utils.state_dict import StateDictExcludeModulesMixin


class OcrVit(LightningModule, StateDictExcludeModulesMixin):
    def __init__(
        self,
        n_head: int = 6,
        n_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = cast(
            Eva, timm.create_model("vit_small_plus_patch16_dinov3", pretrained=True)
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.tokenizer = Tokenizer.from_pretrained("google/flan-t5-base")
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("The tokenizer must define <pad>")
        self.pad_id = pad_id

        d_model = self.backbone.embed_dim
        self.embedding = nn.Embedding(
            self.tokenizer.get_vocab_size(with_added_tokens=True), d_model
        )
        self.decoder = RotaryTransformerDecoder(
            d_model=d_model,
            n_head=n_head,
            n_layers=n_blocks,
            dropout=dropout,
        )

        self._exclude_modules_from_state_dict("backbone")

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _prepare_targets(
        self, texts: list[str], device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        label_ids = [item.ids for item in encoded]

        lengths = torch.tensor([len(ids) for ids in label_ids], device=device)
        max_len = int(lengths.max().item())
        decoder_input_ids = torch.full(
            (len(label_ids), max_len), self.pad_id, dtype=torch.long, device=device
        )
        labels = torch.full(
            (len(label_ids), max_len), -100, dtype=torch.long, device=device
        )
        positions = torch.arange(max_len, device=device)
        tgt_valid = positions[None, :] < lengths[:, None]

        for row, ids in enumerate(label_ids):
            labels[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            decoder_input_ids[row, 0] = self.pad_id
            if len(ids) > 1:
                decoder_input_ids[row, 1 : len(ids)] = torch.tensor(
                    ids[:-1], dtype=torch.long, device=device
                )

        # Position zero is the active decoder-start <pad>, so tgt_valid comes
        # from sequence lengths instead of comparing token IDs with pad_id.
        return decoder_input_ids, tgt_valid, labels

    def _forward_inner(
        self, imgs: Tensor, decoder_input_ids: Tensor, tgt_valid: Tensor
    ) -> Tensor:
        with torch.no_grad():
            memory = self.backbone.forward_features(imgs)

        tgt = self.embedding(decoder_input_ids)
        features = self.decoder(tgt, memory, tgt_valid)
        # Tied output projection: [B, T, D] @ [D, V] -> [B, T, V].
        return F.linear(features, self.embedding.weight)

    @override
    # @torch.compile
    def forward(
        self, imgs: Tensor, decoder_input_ids: Tensor, tgt_valid: Tensor
    ) -> Tensor:
        return self._forward_inner(imgs, decoder_input_ids, tgt_valid)

    # @torch.compile
    def forward_with_loss(
        self, imgs: Tensor, decoder_input_ids: Tensor, tgt_valid: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = self._forward_inner(imgs, decoder_input_ids, tgt_valid)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    def _step(self, batch, step: str) -> Tensor:
        imgs, texts = batch
        decoder_input_ids, tgt_valid, labels = self._prepare_targets(texts, imgs.device)
        _, loss = self.forward_with_loss(imgs, decoder_input_ids, tgt_valid, labels)
        self.log(f"{step}/loss", loss, prog_bar=True, batch_size=imgs.shape[0])
        return loss

    @override
    def training_step(self, batch, _batch_idx):
        return self._step(batch, "train")

    @override
    def validation_step(self, batch, _batch_idx):
        self._step(batch, "val")
