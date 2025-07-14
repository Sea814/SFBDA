import json
import os
from pycocotools.coco import COCO
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import random

sam_checkpoint = "/home/WZH/weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

expand_ratio = 5
coco_annotation_folder = "/home/WZH/project/DeFRCN-IRHIT/datasets/cocosplit/seed0-备份"
image_dir = "/home/WZH/project/DeFRCN-IRHIT/datasets/coco/trainval2014"
output_dir = "output_masks_rgba"

train_annotation_path = "/home/WZH/project/DeFRCN-IRHIT/datasets/cocosplit/datasplit-备份/trainvalno5k.json"
out_put_new_samples_images = f"HITIR_new_samples_images_sam*{expand_ratio}"
out_put_new_samples_annotations = f"HITIR_new_samples_annotations_sam*{expand_ratio}"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(out_put_new_samples_images, exist_ok=True)

def load_existing_coco(file_path):
    try:
        with open(file_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "category_name"}]
        }
    return coco_data

def add_new_coco_data(coco_data, new_data):
    coco_data['images'].extend(new_data['images'])
    coco_data['annotations'].extend(new_data['annotations'])
    
def save_coco_data(coco_data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
def get_random_files(directory, num_files):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory)]
    
    if not all_files:
        raise ValueError(f"The directory {directory} is empty.")
    
    selected_files = random.sample(all_files, num_files)
    
    return selected_files

def generate_coco_annotation(image_id, image_filename, width, height, category_id, bbox, ann_id):
    image_info = {
        "id": image_id,
        "file_name": image_filename,
        "width": width,
        "height": height
    }

    annotation_info = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0
    }

    coco_data = {
        "images": [image_info],
        "annotations": [annotation_info],
    }
    
    return coco_data

def overlay_images(background, mask, category_id, image_id, ann_id):
    h, w = background.shape[:2]
    png_height, png_width = mask.shape[:2]
    if (h < png_height or w < png_width):
        return None
    x_offset = random.randint(0, w - png_width)
    y_offset = random.randint(0, h - png_height)
    overlay = background.copy()
    alpha_channel = mask[:, :, 3]
    rgba_png_image = mask[:, :, :3]

    if x_offset + png_width > w or y_offset + png_height > h:
        raise ValueError("The PNG image exceeds the boundaries of the JPG image.")
    
    for c in range(0, 3):
        overlay[y_offset:y_offset+png_height, x_offset:x_offset+png_width, c] = \
            (alpha_channel / 255.0 * rgba_png_image[:, :, c] + 
             (1 - alpha_channel / 255.0) * background[y_offset:y_offset+png_height, x_offset:x_offset+png_width, c])
    
    overlay_path = os.path.join(out_put_new_samples_images,f"new_sample_{image_id}.jpg")
    cv2.imwrite(overlay_path, overlay)
    new_coco_data = generate_coco_annotation(
        image_id=image_id,
        image_filename=f"new_sample_{image_id}.jpg",
        width=w,
        height=h,
        category_id=category_id,
        bbox=[x_offset, y_offset, png_width, png_height],
        ann_id=ann_id
    )
    return new_coco_data

def process_one_annotation_file(annotation_file):
    
    global train_coco
    global new_image_id
    global new_ann_id
    
    coco = COCO(annotation_file)
    current_coco = load_existing_coco(annotation_file)
    
    for ann_id in coco.getAnnIds():
        annotation = coco.loadAnns([ann_id])[0]
        image_id = annotation['image_id']
        image_info = coco.loadImgs([image_id])[0]
        image_path = os.path.join(image_dir, image_info['file_name'])

        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue

        predictor.set_image(image)

        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        x, y, w, h = map(int, bbox)
        input_box = np.array([x, y, x + w, y + h])

        masks_predict, _, _ = predictor.predict(box=input_box, multimask_output=False)

        mask = masks_predict[0]
        mask = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba_image[:, :, 3] = binary_mask
        
        cropped_mask = rgba_image[y:y+h, x:x+w]
        background_images_paths = get_random_files(image_dir, expand_ratio)
        for background_image_path in background_images_paths:
            background_image = cv2.imread(background_image_path)
            new_coco = overlay_images(background_image, cropped_mask, category_id, new_image_id, new_ann_id)
            if new_coco is None:
                print("there is no new coco data")
            else:
                new_ann_id = new_ann_id + 1
                new_image_id = new_image_id + 1
                add_new_coco_data(current_coco, new_coco)
                add_new_coco_data(train_coco, new_coco)
    output_annotation_path = os.path.join(
        out_put_new_samples_annotations, 
        '/'.join(annotation_file.split('/')[-2:])
    )
    save_coco_data(current_coco, output_annotation_path)

if __name__ == "__main__":
    train_coco = load_existing_coco(train_annotation_path)
    new_image_id = 100000
    new_ann_id = 130000
    for json_file in os.listdir(coco_annotation_folder):
        if json_file.endswith('.json'):
            coco_annotation_file = os.path.join(coco_annotation_folder, json_file)
            process_one_annotation_file(coco_annotation_file)
    output_train_coco_path = os.path.join(out_put_new_samples_annotations, train_annotation_path.split('/')[-1])
    save_coco_data(train_coco, output_train_coco_path)