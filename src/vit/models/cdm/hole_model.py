from pathlib import Path

import lightning as L
import torch
import torchmetrics
from dinov3.hub.backbones import dinov3_vitl16plus
from dinov3.models.vision_transformer import DinoVisionTransformer
from icecream import ic
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.segmentation import MeanIoU
from torchvision.transforms import v2

from vit.data.byte_camera_depth import BCDSample, ByteCameraDepth

from ...heads.dpt import DPT, DPTSegHead


class HoleModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

        self.backbone: DinoVisionTransformer = dinov3_vitl16plus(
            pretrained=False, device=self.device
        )
        self.dpt = DPT(
            d_model=self.backbone.embed_dim, batch_norm=True, device=self.device
        )
        self.seg_head = DPTSegHead(device=self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_miou = MeanIoU(input_format="mixed")

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

    def training_step(self, batch: BCDSample, batch_idx):
        x, y = batch["color"], batch["hole_mask"]
        logits = self.forward(x)

        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch: BCDSample, batch_idx):
        x, y = batch["color"], batch["hole_mask"]
        logits = self.forward(x)

        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.val_miou(logits, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/mIoU", self.val_miou, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)


def main():
    torch.set_float32_matmul_precision("high")

    transform = v2.Compose(
        [
            v2.Resize(
                # (640, 400),
                (512, 320),
                antialias=True,
            ),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            v2.ToPureTensor(),
        ]
    )

    dataset = ByteCameraDepth(
        Path("/home/stefan/src/datasets/ByteCameraDepth/recorded_data"),
        cameras="d455",
        depth=False,
        hole_mask=True,
        transform=transform,
    )
    train_ds, val_ds = random_split(
        dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=8, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=8)

    checkpoint = ModelCheckpoint(
        dirpath="checkpoionts/cdm/hole",
        filename="{epoch}-{val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=5,
    )
    trainer = L.Trainer(max_epochs=200, callbacks=[checkpoint])

    model = HoleModel()
    model.backbone.requires_grad_(False)
    ic(model)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
