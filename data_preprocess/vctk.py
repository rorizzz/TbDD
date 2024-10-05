import os
import csv

root_dir = '/data/xuanru/VCTK/VCTK_16k'
output_csv = 'VCTK.csv'

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['audio_path', 'label'])

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        if os.path.isdir(folder_path):
           
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                
                    wav_path = os.path.join(folder_path, filename)
                    txt_path = wav_path.replace('.wav', '.txt')
                    
                
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as txt_file:
                            text = txt_file.read().strip()
                        
                        writer.writerow([wav_path, text])

print(f"Dataset CSV has been successfully created at {output_csv}.")
