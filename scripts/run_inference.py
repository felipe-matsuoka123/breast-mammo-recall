import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from model import MammoModel

ensemble_model_paths = [
    '/home/felipe-matsuoka/Desktop/breast-screening/models/SPR_v0_CV_Ensemble_fold1_best_model.pth',
    '/home/felipe-matsuoka/Desktop/breast-screening/models/SPR_v0_CV_Ensemble_fold2_best_model.pth',
    '/home/felipe-matsuoka/Desktop/breast-screening/models/SPR_v0_CV_Ensemble_fold3_best_model.pth',
    '/home/felipe-matsuoka/Desktop/breast-screening/models/SPR_v0_CV_Ensemble_fold4_best_model.pth',
    '/home/felipe-matsuoka/Desktop/breast-screening/models/SPR_v0_CV_Ensemble_fold5_best_model.pth'
]

test_csv_path = '/home/felipe-matsuoka/Desktop/breast-screening/sample_submissionA.csv'
base_path = '/home/felipe-matsuoka/Documents/spr-mammo-recall/spr-mmg-merged'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_df = pd.read_csv(test_csv_path)
accession_numbers = [str(x).zfill(6) for x in test_df['AccessionNumber']]

ensemble_models = []
for model_path in ensemble_model_paths:
    model = MammoModel(dropout_rate=0.0)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        model = checkpoint
    model.to(device)
    model.eval()
    ensemble_models.append(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

results = []
for acc_num in tqdm(accession_numbers, desc="Processing patients"):
    folder_path = os.path.join(base_path, acc_num)
    if not os.path.isdir(folder_path):
        print(f"Folder for AccessionNumber {acc_num} not found, skipping.")
        continue

    image_predictions = []
    image_files = [file for file in os.listdir(folder_path)
                   if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in tqdm(image_files, desc=f"Processing images for {acc_num}", leave=False):
        image_path = os.path.join(folder_path, file)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        input_tensor = transform(image).unsqueeze(0).to(device)

        model_probs = []
        with torch.no_grad():
            for model in ensemble_models:
                output = model(input_tensor)
                probability = torch.sigmoid(output).item()
                model_probs.append(probability)
        avg_image_prob = np.mean(model_probs)
        image_predictions.append(avg_image_prob)
    
    if image_predictions:
        # Média das predições de todas as imagens do mesmo AccessionNumber
        avg_prediction = np.mean(image_predictions)
        results.append({'AccessionNumber': acc_num, 'target': avg_prediction})
    else:
        print(f"No valid images found for AccessionNumber {acc_num}")

results_df = pd.DataFrame(results)
results_df.to_csv('ensemble_inference_results.csv', index=False)
print("Inference completed. Results saved to ensemble_inference_results.csv")
