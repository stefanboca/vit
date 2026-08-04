import torch
from dinov3.hub.backbones import dinov3_vitl16plus
from dinov3.models.vision_transformer import DinoVisionTransformer
from torch import Tensor, nn

from ...heads.dpt import DPT, DPTSegHead


class HoleModel(nn.Module):
    def __init__(self, *, device: torch.device | None = None):
        super().__init__()
        self.backbone: DinoVisionTransformer = dinov3_vitl16plus(
            pretrained=False, device=device
        )
        self.dpt = DPT(d_model=self.backbone.embed_dim, batch_norm=True, device=device)
        self.seg_head = DPTSegHead(device=device)

    def forward(self, input: Tensor) -> Tensor:
        B, _, H, W = input.shape
        H //= self.backbone.patch_size
        W //= self.backbone.patch_size
        x = self.backbone.get_intermediate_layers(
            input, n=[5, 11, 17, 23], return_class_token=True
        )
        patch_tokens, cls_tokens = zip(*x)
        patch_tokens = [p.reshape(B, H, W, -1) for p in patch_tokens]
        x = self.dpt.forward(cls_tokens, patch_tokens)  # ty: ignore[invalid-argument-type]
        x = self.seg_head.forward(x)
        return x


if __name__ == "__main__":
    pass
