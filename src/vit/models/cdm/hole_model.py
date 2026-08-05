from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from dinov3.hub.backbones import dinov3_vitl16
from dinov3.models.vision_transformer import DinoVisionTransformer
from icecream import ic
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins import MixedPrecision
from segmentation_models_pytorch.losses import DiceLoss
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.segmentation import MeanIoU
from torchvision.transforms import v2
from torchvision.utils import make_grid

from vit.data.byte_camera_depth import BCDSample, ByteCameraDepth

from ...heads.dpt import DPT, DPTSegHead


class HoleModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

        self.backbone: DinoVisionTransformer = dinov3_vitl16(
            pretrained=False, device=self.device
        )
        self.dpt = DPT(
            d_model=self.backbone.embed_dim, batch_norm=True, device=self.device
        )
        self.seg_head = DPTSegHead(device=self.device, n_classes=1)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss("binary")

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_miou = MeanIoU(num_classes=2)

    @torch.compile
    def forward(self, input: Tensor) -> Tensor:
        B, _, H, W = input.shape
        H //= self.backbone.patch_size
        W //= self.backbone.patch_size
        with torch.no_grad():  # TODO: only if frozen
            x = self.backbone.get_intermediate_layers(
                input, n=[5, 11, 17, 23], return_class_token=True
            )
        patch_tokens, cls_tokens = zip(*x)
        patch_tokens = [p.reshape(B, H, W, -1) for p in patch_tokens]
        x = self.dpt.forward(cls_tokens, patch_tokens)  # ty: ignore[invalid-argument-type]
        x = self.seg_head.forward(x)
        return x

    @torch.compile
    def loss(self, preds: Tensor, target: Tensor) -> Tensor:
        target = target.to(torch.float32)
        return self.bce_loss.forward(preds, target) + self.dice_loss.forward(
            preds, target
        )

    def training_step(self, batch: BCDSample, batch_idx):
        x, y = batch["color"], batch["hole_mask"]
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.train_acc(logits, y)

        self.log_metrics(batch, logits, loss, batch_idx, "train")

        return loss

    def validation_step(self, batch: BCDSample, batch_idx):
        x, y = batch["color"], batch["hole_mask"]
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.val_acc(logits, y)
        self.val_miou(logits, y)

        self.log_metrics(batch, logits, loss, batch_idx, "val")

    def log_metrics(
        self, batch, logits, loss, batch_idx, kind: Literal["train", "val"]
    ):
        self.log(f"{kind}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            f"{kind}/acc",
            self.train_acc if kind == "val" else self.val_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if self.logger is not None and batch_idx % 50 == 0:
            B = logits.shape[0]
            N = min(B, 4)
            color_imgs = batch["color"][:N]
            gts = batch["hole_mask"][:N].expand_as(color_imgs)
            preds = F.sigmoid(logits[:N]).expand_as(color_imgs)
            imgs = torch.cat([color_imgs, gts, preds], dim=3)
            grid = make_grid(imgs, nrow=1, padding=0)
            self.logger.experiment.add_image(f"{kind}/images", grid, batch_idx)

        if kind == "val":
            self.log(
                "val/mIoU", self.val_miou, on_step=True, on_epoch=True, prog_bar=True
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)


def main():
    torch.set_float32_matmul_precision("high")

    transform = v2.Compose(
        [
            v2.Resize(
                (320, 512),
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
    train_loader = DataLoader(
        train_ds, batch_size=24, num_workers=16, pin_memory=True, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=24, num_workers=16, pin_memory=True)

    model = HoleModel()
    model.backbone.load_state_dict(
        torch.load(
            "/home/stefan/src/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            map_location=model.device,
            mmap=True,
            weights_only=True,
        )
    )
    model.backbone.requires_grad_(False)
    model.backbone.eval()

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/cdm/hole",
        filename="epoch={epoch}-val_loss={val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last="link",
        auto_insert_metric_name=False,
        save_on_exception=True,
    )
    amp = MixedPrecision(precision="bf16-mixed", device="cuda")
    logger = TensorBoardLogger("lightning_logs", name="cdm/hole_model")
    trainer = L.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint],
        plugins=[amp],
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path="last",
    )


if __name__ == "__main__":
    main()
