# vit

A monorepo for my computer vision experiments, including depth estimation,
instance segmentation, object detection, and optical character recognition
with PyTorch.

Much of the work explores adapting pretrained image representations to new
tasks while keeping the image encoder frozen and training task-specific
heads. The depth experiments also investigate generating synthetic depth
images with the missing measurements found in real depth-camera data.

## Experiments

- **[Depth prediction and synthesis](src/vit/models/cdm/synth/):** Separate
  hole and value models for predicting missing-depth regions and depth values.
  Includes a DINOv3 backbone with a trainable DPT head and adaptation of
  Depth Anything 3.
- **[Instance segmentation](src/vit/models/seg/):** A frozen DINOv3 encoder
  with a trainable DPT segmentation head for predicting individual object masks.
- **[RF-DETR](src/vit/models/rfdetr.py):** Fine-tuning for object detection
  and segmentation, with dataset preparation and model export.
- **[OCR](src/vit/models/ocr/):** Text recognition using DINOv3 image features
  and a trainable transformer decoder, with synthetic text images for training.

These are experiments at different stages of development.
