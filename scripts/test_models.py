import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import os
from evaluate import evaluate_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():

    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)

    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1,1), stride=(1,1))
    aux_in = model.aux_classifier[4].in_channels
    model.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=(1,1), stride=(1,1))

    return model

def run_evaluation():

    experiments = [
        ("Baseline (ImageNet)", "experiments/DeepLabV3_MobileNetV3/best_model.pth"),
        ("COCO Weights", "experiments/DeepLabV3_MobileNetV3_COCO/best_model.pth"),
        ("COCO + ComboLoss", "experiments/DeepLabV3_MobileNetV3_COCO_ComboLoss/best_model.pth"),
        ("ImageNet + ComboLoss", "experiments/DeepLabV3_MobileNetV3_ComboLoss/best_model.pth")
    ]

    for name, path in experiments:
        if not os.path.exists(path):
            print(f"Skipping {name} as {path} not found")
            continue

        print(f"Evaluating {name}")

        model = get_model()

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)

        evaluate_model(model, model_name=name)

if __name__ == "__main__":
    run_evaluation()