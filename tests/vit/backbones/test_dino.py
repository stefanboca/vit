import gc
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from vit.backbones.dino import (
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vits16,
    dinov3_vits16plus,
)


@pytest.fixture(scope="session")
def image_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


@pytest.fixture(scope="session")
def image(image_transform):
    image = Image.open("assets/cats_on_couch.jpg").convert("RGB")
    return image_transform(image).unsqueeze(0)


@pytest.fixture(scope="session")
def checkpoint_dir():
    return Path(os.environ["DINOV3_CHECKPOINTS_DIR"])


@pytest.mark.parametrize(
    "checkpoint_filename,model,expected_cls,expected_patch",
    [
        (
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            dinov3_vits16,
            [0.463561, -0.415609, 0.408236, -0.126613, -0.286636],
            [-0.038754, -0.250895, -0.016392, -0.455473, 0.571582],
        ),
        (
            "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            dinov3_vits16plus,
            [-0.471349, -1.365778, -0.317983, 0.377219, -0.769085],
            [0.144551, -0.388117, -0.393433, -0.157695, -0.600380],
        ),
        (
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            dinov3_vitb16,
            [1.034643, -0.180609, -0.341018, -0.066376, -0.011383],
            [-0.082523, -0.456272, -0.728029, -0.430680, -0.152880],
        ),
        (
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            dinov3_vitl16,
            [0.484527, -0.582214, 0.480636, 0.592040, 0.945166],
            [-0.211367, -0.490863, -0.257131, 0.101763, 0.154511],
        ),
        (
            "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
            dinov3_vith16plus,
            [-0.064575, -0.148866, -0.621524, 0.634878, 0.152695],
            [-0.093817, 0.287407, -0.050036, 0.428043, 0.094561],
        ),
        (
            "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            dinov3_vit7b16,
            [0.275439, -0.261353, 0.067772, 0.049936, -0.158747],
            [0.044442, -0.052542, 0.070777, -0.065111, -0.026546],
        ),
    ],
    ids=[
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vith16plus",
        "dinov3_vit7b16",
    ],
)
@torch.no_grad()
def test_loads_and_matches(
    checkpoint_filename,
    model,
    expected_cls,
    expected_patch,
    checkpoint_dir,
    image,
):
    # free up memory from previous runs
    gc.collect()

    model = model(device="cpu").eval()
    model.load_state_dict(
        torch.load(
            checkpoint_dir / checkpoint_filename,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    )

    with torch.inference_mode():
        out = model.forward(image)

    torch.testing.assert_close(
        out.cls_token[0, :5].tolist(), expected_cls, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        out.patches[0, 0, 0, :5].tolist(), expected_patch, rtol=1e-3, atol=1e-3
    )
