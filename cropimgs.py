import cv2
import os
import random

# --- CONFIGURATION ---
folder = '/Users/alex/Documents/handwriting_data'
output_base = os.path.join(folder, "dataset")
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
target_size = (32, 32)  # Change to (28, 28) if you want MNIST style
split_ratio = 0.5 
random.seed(42)

# Grid coordinates optimized for your sheets
cords=[]
start_x, start_y = 207, 461
width, height = 345, 245

for row in range(5):
    for col in range(6):
        x1 = start_x + (col * width)
        y1 = start_y + (row * height)
        x2 = x1 + width
        y2 = y1 + height
        cords.append([(x1, y1), (x2, y2)])

# --- PROCESSING ---
for filename in os.listdir(folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(os.path.join(folder, filename))
        if img is None: continue

        for i, (p1, p2) in enumerate(cords):
            if i >= len(alphabet): break
            letter = alphabet[i]

            # 1. Select subset
            subset = "train" if random.random() < split_ratio else "val"
            save_dir = os.path.join(output_base, subset, letter)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 2. Crop
            crop = img[p1[1]:p2[1], p1[0]:p2[0]]
            
            # 3. Resize (using INTER_AREA which is best for shrinking)
            resized_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            
            # 4. Save
            base_name = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_{i}.png"), resized_crop)

print(f"Dataset complete! Images resized to {target_size}.")