from typing import cast, override

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from timm.models import Eva
from tokenizers import Tokenizer
from torch import Tensor, nn
from torchmetrics import MeanMetric
from torchmetrics.text import CharErrorRate, Perplexity, WordErrorRate

from vit.nn import RotaryTransformerDecoder
from vit.utils.state_dict import StateDictExcludeModulesMixin


class OcrVit(LightningModule, StateDictExcludeModulesMixin):
    def __init__(
        self,
        n_head: int = 6,
        n_blocks: int = 6,
        dropout: float = 0.1,
        max_generation_length: int = 128,
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
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.max_generation_length = max_generation_length

        self.val_perplexity = Perplexity(ignore_index=-100)
        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()
        self.val_exact_match = MeanMetric()

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

    def forward(
        self,
        imgs: Tensor,
        decoder_input_ids: Tensor,
        tgt_valid: Tensor,
        memory: Tensor | None = None,
    ) -> Tensor:
        if memory is None:
            with torch.no_grad():
                memory = self.backbone.forward_features(imgs)

        tgt = self.embedding(decoder_input_ids)
        features = self.decoder(tgt, memory, tgt_valid)
        # Tied output projection: [B, T, D] @ [D, V] -> [B, T, V].
        return F.linear(features, self.embedding.weight)

    def forward_with_loss(
        self,
        imgs: Tensor,
        decoder_input_ids: Tensor,
        tgt_valid: Tensor,
        labels: Tensor,
        memory: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = self.forward(imgs, decoder_input_ids, tgt_valid, memory)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    def _step(
        self, batch, step: str, memory: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        imgs, texts = batch
        decoder_input_ids, tgt_valid, labels = self._prepare_targets(texts, imgs.device)
        logits, loss = self.forward_with_loss(
            imgs, decoder_input_ids, tgt_valid, labels, memory
        )
        self.log(f"{step}/loss", loss, prog_bar=True, batch_size=imgs.shape[0])
        return logits, loss, labels

    def _generate(self, imgs: Tensor, memory: Tensor) -> list[str]:
        decoder_input_ids = torch.full(
            (imgs.shape[0], 1), self.pad_id, dtype=torch.long, device=imgs.device
        )
        finished = torch.zeros(imgs.shape[0], dtype=torch.bool, device=imgs.device)
        generated = []

        for _ in range(self.max_generation_length):
            tgt_valid = torch.ones_like(decoder_input_ids, dtype=torch.bool)
            features = self.decoder(
                self.embedding(decoder_input_ids), memory, tgt_valid
            )
            next_ids = F.linear(features[:, -1], self.embedding.weight).argmax(-1)
            if self.eos_id is not None:
                next_ids = torch.where(
                    finished, torch.full_like(next_ids, self.pad_id), next_ids
                )
                finished |= next_ids == self.eos_id
            generated.append(next_ids)
            decoder_input_ids = torch.cat((decoder_input_ids, next_ids[:, None]), dim=1)
            if self.eos_id is not None and finished.all():
                break

        token_ids = torch.stack(generated, dim=1).tolist()
        return self.tokenizer.decode_batch(token_ids, skip_special_tokens=True)

    @override
    def training_step(self, batch, _batch_idx):
        _, loss, _ = self._step(batch, "train")
        return loss

    @override
    def validation_step(self, batch, _batch_idx):
        imgs, texts = batch
        with torch.no_grad():
            memory = self.backbone.forward_features(imgs)
        logits, _, labels = self._step(batch, "val", memory)
        predictions = self._generate(imgs, memory)

        self.val_perplexity.update(logits.float(), labels)
        self.val_cer.update(predictions, texts)
        self.val_wer.update(predictions, texts)
        self.val_exact_match.update(
            torch.tensor(
                [
                    prediction == target
                    for prediction, target in zip(predictions, texts)
                ],
                dtype=torch.float32,
                device=imgs.device,
            )
        )

        self.log("val/perplexity", self.val_perplexity, on_step=False, on_epoch=True)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True)
        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True)
        self.log("val/exact_match", self.val_exact_match, on_step=False, on_epoch=True)
