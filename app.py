import gradio as gr
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = models.ResNet50_Weights.DEFAULT
PREPROCESS = WEIGHTS.transforms()
LABELS = WEIGHTS.meta["categories"]

model = models.resnet50(weights=WEIGHTS).eval().to(DEVICE)
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])


def explain(image):
    if image is None:
        raise gr.Error("Upload an image first.")

    input_tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = model(input_tensor).softmax(dim=1)[0]

    top_indices = probabilities.topk(5).indices
    predictions = {
        LABELS[int(index)]: float(probabilities[index]) for index in top_indices
    }

    heatmap = cam(input_tensor=input_tensor, targets=None)[0]
    mean = input_tensor.new_tensor(PREPROCESS.mean).view(3, 1, 1)
    std = input_tensor.new_tensor(PREPROCESS.std).view(3, 1, 1)
    base = (input_tensor[0] * std + mean).permute(1, 2, 0)
    base = base.detach().cpu().numpy().clip(0, 1)
    overlay = show_cam_on_image(base, heatmap, use_rgb=True)

    return predictions, Image.fromarray(overlay)


demo = gr.Interface(
    fn=explain,
    inputs=gr.Image(type="pil", label="Input image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top predictions"),
        gr.Image(label="Grad-CAM for the top prediction"),
    ],
    title="Grad-CAM Explainer",
    description=(
        "Upload an image to see the top five ResNet50 predictions and the regions "
        "that contributed to the highest-scoring class."
    ),
    article=(
        "Warmer colors indicate stronger contributions to the selected class score. "
        "This is a visualization aid, not a causal explanation.\n\n"
        "[ResNet50 weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) · "
        "[ImageNet-1K dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k) · "
        "[Kaggle notebook](https://www.kaggle.com/code/saglamchad/gradcam-explainer-resnet50)"
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
