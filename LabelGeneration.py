import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

input_dir = r"C:\Users\dalki\Desktop\RBC Images"  # Raw Smear Images
output_dir = r"C:\Users\dalki\Downloads\OutputRBCs"
min_cell_area = 1000
max_cell_area = 10000
display_size = (256, 256)

# Create output folders
os.makedirs(f"{output_dir}/acanthocyteTest", exist_ok=True)
os.makedirs(f"{output_dir}/not_acanthocyteTest", exist_ok=True)

def segment_rbc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_cell_area < area < max_cell_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cell_img = image[y:y+h, x:x+w]
            cells.append(cell_img)
    return cells

def overlay_text(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    background_color = (0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    padding = 4

    # Draw background rectangle
    cv2.rectangle(image, (5, 5), (10 + text_w, 10 + text_h + padding), background_color, -1)
    # Put text
    cv2.putText(image, text, (8, 8 + text_h), font, font_scale, color, thickness)
    return image

def label_cells():
    img_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    cell_count = 0
    all_cells = []

    # Pre-process all images to collect all RBCs
    for file in img_files:
        img = cv2.imread(os.path.join(input_dir, file))
        cells = segment_rbc(img)
        for cell in cells:
            all_cells.append(cell)

    print(f"Total RBCs to label: {len(all_cells)}")

    # Progress bar for labeling
    for i, cell in enumerate(tqdm(all_cells, desc="Labeling RBCs")):
        cell_resized = cv2.resize(cell, display_size)
        label_text = "1 = Acanthocyte | 2 = Not Acanthocyte | s = Skip"
        labeled_img = overlay_text(cell_resized.copy(), label_text)
        cv2.imshow("Label RBC", labeled_img)
        key = cv2.waitKey(0)

        if key == ord('1'):
            label = "acanthocyteTest"
        elif key == ord('2'):
            label = "not_acanthocyteTest"
        elif key == ord('s'):
            continue  # skip
        else:
            print("Invalid key. Press 1, 2, or s.")
            continue

        save_path = os.path.join(output_dir, label, f"{label}_{cell_count}.jpg")
        cv2.imwrite(save_path, cell_resized)
        cell_count += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    label_cells()


