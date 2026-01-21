import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.deeplabv3 import get_deeplab_model
from src.evaluation.evaluate import evaluate_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_evaluation():

    experiments = [
        ("Augmented + ImageNet + ComboLoss", "experiments/Exp5_DeepLab_MobileNet_ImageNet_CombinedLoss_Augmented/best_model.pth")
    ]

    for name, path in experiments:
        if not os.path.exists(path):
            print(f"Skipping {name} as {path} not found")
            continue

        print(f"Evaluating {name}")

        model = get_deeplab_model(device, model_weights="None")

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)

        evaluate_model(model, model_name=name)

if __name__ == "__main__":
    run_evaluation()