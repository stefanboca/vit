from collections.abc import Callable

import torch
from torch import Tensor


class BeamSearch[State]:
    """Batched autoregressive beam search with caller-managed decoder state.

    ``step`` consumes the current token and returns next-token logits plus an
    updated state. The two state callbacks expand state across beams and select
    parent beams after each search step; this keeps the helper model-agnostic.
    """

    def __init__(
        self,
        beam_size: int = 4,
        max_length: int = 128,
        length_penalty: float = 0.0,
    ):
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty

    def search(
        self,
        start_tokens: Tensor,
        step: Callable[[Tensor, State], tuple[Tensor, State]],
        state: State,
        eos_token_id: int,
        expand_state: Callable[[State, int], State],
        reorder_state: Callable[[State, Tensor], State],
    ) -> Tensor:
        """Return best generated token IDs, including EOS when emitted."""
        batch_size = start_tokens.shape[0]
        beam_size = self.beam_size
        state = expand_state(state, beam_size)
        sequences = start_tokens[:, None, None].expand(batch_size, beam_size, 1)
        scores = torch.full(
            (batch_size, beam_size), -torch.inf, device=start_tokens.device
        )
        scores[:, 0] = 0
        lengths = torch.zeros(
            (batch_size, beam_size), dtype=torch.long, device=start_tokens.device
        )
        finished = torch.zeros_like(scores, dtype=torch.bool)

        for _ in range(self.max_length):
            logits, state = step(sequences[:, :, -1].reshape(-1), state)
            vocab_size = logits.shape[-1]
            log_probs = logits.log_softmax(dim=-1).view(
                batch_size, beam_size, vocab_size
            )
            if finished.any():
                log_probs = log_probs.masked_fill(finished[:, :, None], -torch.inf)
                log_probs[:, :, eos_token_id] = torch.where(
                    finished, torch.zeros_like(scores), log_probs[:, :, eos_token_id]
                )

            candidate_scores = scores[:, :, None] + log_probs
            scores, candidates = candidate_scores.view(batch_size, -1).topk(
                beam_size, dim=-1
            )
            parents = candidates // vocab_size
            tokens = candidates % vocab_size
            offsets = (
                torch.arange(batch_size, device=start_tokens.device)[:, None]
                * beam_size
            )
            parent_indices = (parents + offsets).reshape(-1)

            previous_finished = finished.gather(1, parents)
            lengths = lengths.gather(1, parents) + (~previous_finished).long()
            finished = previous_finished | (tokens == eos_token_id)
            sequences = sequences.gather(
                1, parents[:, :, None].expand(-1, -1, sequences.shape[-1])
            )
            sequences = torch.cat((sequences, tokens[:, :, None]), dim=-1)
            state = reorder_state(state, parent_indices)
            if finished.all():
                break

        if self.length_penalty:
            norm = ((5 + lengths.clamp_min(1)) / 6).pow(self.length_penalty)
            scores = scores / norm
        best = scores.argmax(dim=1)
        return sequences[torch.arange(batch_size, device=start_tokens.device), best, 1:]
