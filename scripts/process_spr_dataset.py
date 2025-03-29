import os
import glob
import pandas as pd

# Load the CSV with accession-level labels and additional metadata
df = pd.read_csv("/home/felipe-matsuoka/Desktop/breast-screening/scripts/train (2).csv")

# List to collect image-level data
image_data = []

# Base directory where the PNG folders are stored
base_dir = "/home/felipe-matsuoka/Documents/spr-mammo-recall/spr-mmg-merged"

# Iterate over each row in the CSV
for idx, row in df.iterrows():
    accession = str(row['AccessionNumber'])
    # Uncomment the next line if you need to enforce 6 digits
    accession = accession.zfill(6)
    patient_id = row['PatientID']
    laterality = row['Laterality']
    target = row['target']
    
    # Build the path to the folder with this accession number
    folder_path = os.path.join(base_dir, accession)
    print(f"Checking folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        continue
    
    # Get all PNG files in the folder
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    #print(f"Found {len(png_files)} PNG files in {folder_path}")
    
    # For each image file, add a record with all metadata
    for png_file in png_files:
        image_data.append({
            "AccessionNumber": accession,
            "PatientID": patient_id,
            "Laterality": laterality,
            "target": target,
            "image_path": png_file
        })

# Create a DataFrame with image-level labels and metadata
df_images = pd.DataFrame(image_data)
print("Resulting dataframe preview:")
print(df_images.head())

# Save the image-level CSV
df_images.to_csv("spr_train_image_level.csv", index=False)
