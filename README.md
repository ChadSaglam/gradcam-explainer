# Grad-CAM Explainer

A small ResNet50 demo that shows the top five ImageNet predictions and a Grad-CAM overlay for the highest-scoring class.

## Resources

- [Kaggle notebook](https://www.kaggle.com/code/saglamchad/gradcam-explainer-resnet50)
- [ResNet50 model and ImageNet-1K V2 weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [ImageNet-1K dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391)
- [Grad-CAM implementation](https://github.com/jacobgil/pytorch-grad-cam)

The demo uses pretrained torchvision weights. No training or ImageNet download is required.

## Run the notebook

Open the [Kaggle notebook](https://www.kaggle.com/code/saglamchad/gradcam-explainer-resnet50) and select **Copy & Edit**. Internet access must be enabled so the notebook can install `grad-cam`, download the ResNet50 weights and fetch the sample image. A GPU is optional.

The notebook downloads a sample photograph, prints the top five predictions and saves the side-by-side visualization as `gradcam.png`.

## Run the Gradio app locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio and upload an image. The first run also downloads the pretrained weights, which are about 98 MB.

## How to read the result

Grad-CAM uses gradients from the selected class at the last residual block of ResNet50. Warmer colors indicate regions that contributed more to that class score. The overlay is a visualization aid, not proof that the model made its decision for a human-interpretable reason.

The classifier is limited to the 1,000 ImageNet categories. Images outside those categories can still receive confident but misleading predictions.

## Files

- `app.py`: local Gradio interface
- `gradcam_kaggle.ipynb`: self-contained Kaggle notebook
- `requirements.txt`: dependencies for the local app
