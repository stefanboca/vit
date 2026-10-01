from typing import cast, override

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from timm.models import Eva
from tokenizers import Tokenizer
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from torchmetrics import MeanMetric
from torchmetrics.text import CharErrorRate, Perplexity, WordErrorRate

from vit.nn import RotaryTransformerDecoder
from vit.utils.beam_search import BeamSearch
from vit.utils.state_dict import StateDictExcludeModulesMixin


class OcrVit(LightningModule, StateDictExcludeModulesMixin):
    def __init__(
        self,
        n_head: int = 6,
        n_blocks: int = 6,
        dropout: float = 0.1,
        max_generation_length: int = 128,
        beam_size: int = 4,
        length_penalty: float = 0.6,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        total_steps: int = 800_000,
        min_learning_rate: float = 5e-6,
        cached_features: bool = False,
    ):
        super().__init__()
        self.max_generation_length = max_generation_length
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_learning_rate = min_learning_rate

        backbone = cast(
            Eva,
            timm.create_model(
                "vit_small_plus_patch16_dinov3", pretrained=not cached_features
            ),
        )
        d_model = backbone.embed_dim
        self.backbone: Eva | None = None
        if not cached_features:
            self.backbone = backbone
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.tokenizer = Tokenizer.from_pretrained("google/flan-t5-base")
        pad_id = self.tokenizer.token_to_id("<pad>")
        eos_id = self.tokenizer.token_to_id("</s>")
        if pad_id is None or eos_id is None:
            raise ValueError("The tokenizer must define <pad> and </s>")
        self.pad_id = pad_id
        self.eos_id = eos_id

        self.embedding = nn.Embedding(
            self.tokenizer.get_vocab_size(with_added_tokens=True), d_model
        )
        self.decoder = RotaryTransformerDecoder(
            d_model=d_model,
            n_head=n_head,
            n_layers=n_blocks,
            dropout=dropout,
        )

        self.val_perplexity = Perplexity(ignore_index=-100)
        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()
        self.val_exact_match = MeanMetric()

        self._exclude_modules_from_state_dict("backbone")

    @override
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.total_steps - self.warmup_steps,
            eta_min=self.min_learning_rate,
        )
        floor = LambdaLR(
            optimizer,
            lr_lambda=lambda _: self.min_learning_rate / self.learning_rate,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine, floor],
            milestones=[self.warmup_steps, self.total_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
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
        memory: Tensor,
        decoder_input_ids: Tensor,
        tgt_valid: Tensor,
        *,
        memory_valid: Tensor | None = None,
    ) -> Tensor:
        tgt = self.embedding(decoder_input_ids)
        features = self.decoder(tgt, memory, tgt_valid, memory_valid)
        # Tied output projection: [B, T, D] @ [D, V] -> [B, T, V].
        return F.linear(features, self.embedding.weight)

    def _encode_images(
        self, imgs: Tensor | list[Tensor]
    ) -> tuple[Tensor, Tensor | None]:
        if self.backbone is None:
            if not isinstance(imgs, Tensor):
                raise TypeError("Cached features must be a padded tensor")
            return imgs, None

        if isinstance(imgs, Tensor):
            with torch.no_grad():
                return self.backbone.forward_features(imgs), None

        groups: dict[tuple[int, ...], list[int]] = {}
        for index, image in enumerate(imgs):
            groups.setdefault(tuple(image.shape), []).append(index)

        features: list[Tensor | None] = [None] * len(imgs)
        with torch.no_grad():
            for indices in groups.values():
                group = torch.stack([imgs[index] for index in indices])
                encoded = self.backbone.forward_features(group)
                for index, feature in zip(indices, encoded, strict=True):
                    features[index] = feature

        assert all(feature is not None for feature in features)
        ordered = [cast(Tensor, feature) for feature in features]
        lengths = torch.tensor(
            [feature.shape[0] for feature in ordered], device=ordered[0].device
        )
        if bool((lengths == lengths[0]).all()):
            return torch.stack(ordered), None
        padded = pad_sequence(ordered, batch_first=True)
        valid = (
            torch.arange(padded.shape[1], device=padded.device)[None, :]
            < lengths[:, None]
        )
        return padded, valid

    def forward_with_loss(
        self,
        memory: Tensor,
        decoder_input_ids: Tensor,
        tgt_valid: Tensor,
        labels: Tensor,
        *,
        memory_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = self.forward(
            memory, decoder_input_ids, tgt_valid, memory_valid=memory_valid
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss

    def _step(
        self,
        batch,
        step: str,
        memory: Tensor | None = None,
        memory_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        imgs, texts, *masks = batch
        if memory is None:
            memory, encoded_valid = self._encode_images(imgs)
            if memory_valid is None:
                memory_valid = encoded_valid
        if memory_valid is None:
            memory_valid = masks[0] if masks else None
        decoder_input_ids, tgt_valid, labels = self._prepare_targets(
            texts, memory.device
        )
        logits, loss = self.forward_with_loss(
            memory,
            decoder_input_ids,
            tgt_valid,
            labels,
            memory_valid=memory_valid,
        )
        self.log(
            f"{step}/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(imgs),
        )
        return logits, loss, labels

    def _generate(
        self,
        memory: Tensor,
        memory_valid: Tensor | None = None,
    ) -> list[str]:
        state = (self.decoder.init_cache(memory), 0, memory_valid)

        def expand_state(state, count: int):
            cache, position, valid = state
            expanded_cache = [
                tuple(
                    value.repeat_interleave(count, dim=0) if value is not None else None
                    for value in layer_cache
                )
                for layer_cache in cache
            ]
            if valid is not None:
                valid = valid.repeat_interleave(count, dim=0)
            return expanded_cache, position, valid

        def reorder_state(state, indices: Tensor):
            cache, position, valid = state
            reordered_cache = [
                tuple(
                    value.index_select(0, indices) if value is not None else None
                    for value in layer_cache
                )
                for layer_cache in cache
            ]
            if valid is not None:
                valid = valid.index_select(0, indices)
            return reordered_cache, position, valid

        def step(token_ids: Tensor, state):
            cache, position, valid = state
            features, cache = self.decoder.decode_step(
                self.embedding(token_ids[:, None]), cache, position, valid
            )
            logits = F.linear(features[:, 0], self.embedding.weight)
            return logits, (cache, position + 1, valid)

        start_tokens = torch.full(
            (memory.shape[0],), self.pad_id, dtype=torch.long, device=memory.device
        )
        beam_search = BeamSearch(
            beam_size=self.beam_size,
            max_length=self.max_generation_length,
            length_penalty=self.length_penalty,
        )
        token_ids = beam_search.search(
            start_tokens,
            step,
            state,
            self.eos_id,
            expand_state,
            reorder_state,
        )
        token_ids = token_ids.tolist()
        return self.tokenizer.decode_batch(token_ids, skip_special_tokens=True)

    @override
    def training_step(self, batch, _batch_idx):
        _, loss, _ = self._step(batch, "train")
        return loss

    @override
    @torch.no_grad
    def validation_step(self, batch, _batch_idx):
        imgs, texts, *masks = batch
        memory, encoded_valid = self._encode_images(imgs)
        memory_valid = masks[0] if masks and masks[0] is not None else encoded_valid
        logits, _, labels = self._step(batch, "val", memory, memory_valid)
        predictions = self._generate(memory, memory_valid)

        self.val_perplexity(logits.float(), labels)
        self.val_cer(predictions, texts)
        self.val_wer(predictions, texts)
        self.val_exact_match(
            torch.tensor(
                [
                    prediction == target
                    for prediction, target in zip(predictions, texts)
                ],
                dtype=torch.float32,
                device=memory.device,
            )
        )

        self.log("val/perplexity", self.val_perplexity, on_step=True, on_epoch=True)
        self.log("val/cer", self.val_cer, on_step=True, on_epoch=True)
        self.log("val/wer", self.val_wer, on_step=True, on_epoch=True)
        self.log("val/exact_match", self.val_exact_match, on_step=True, on_epoch=True)

    @override
    @torch.no_grad
    def predict_step(self, batch, _batch_idx, _dataloader_idx=0) -> list[str]:
        imgs, _texts, *masks = batch
        memory, encoded_valid = self._encode_images(imgs)
        memory_valid = masks[0] if masks and masks[0] is not None else encoded_valid
        return self._generate(memory, memory_valid)
