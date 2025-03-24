import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

model_path = 'path_to_your_model.pth'
test_csv_path = 'path_to_test_csv.csv'
base_path = 'base_path/spr_mammo_mammo_recall'  # Base folder where AcessionNumber folders are stored

test_df = pd.read_csv(test_csv_path)
accession_numbers = test_df['AcessionNumber'].astype(str).tolist()

model = torch.load(model_path)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Example size; change if needed.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

results = []
for acc_num in accession_numbers:
    folder_path = os.path.join(base_path, acc_num)
    if not os.path.isdir(folder_path):
        print(f"Folder for AcessionNumber {acc_num} not found, skipping.")
        continue

    predictions = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.item()
                predictions.append(prediction)
    
    if predictions:
        avg_prediction = sum(predictions) / len(predictions)
        results.append({'AcessionNumber': acc_num, 'target': avg_prediction})
    else:
        print(f"No valid images found for AcessionNumber {acc_num}")

results_df = pd.DataFrame(results)
results_df.to_csv('inference_results.csv', index=False)
print("Inference completed. Results saved to inference_results.csv")
