import numpy as np
import os
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image
from torch import Tensor

def save_masks_to_textfile(masks, imagename):
    output_dir = 'examples/output/cam0'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'masks.txt')
    with open(file_path, 'a') as file:  # Open file in append mode
        for mask in masks:
            file.write(f"('{imagename}', {Tensor.numpy(mask)})\n")  # ISSUE: Data is truncated here. how do I retain everything?


# Fonction pour traiter les sorties d'inférence du modèle
def process_inference(model_output, image, imagename: str):

    # Extraire les masques, les scores, et les labels de la sortie du modèle
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']
    masks_retenus = []

    # Convertir l'image en tableau numpy
    img_np = np.array(image)

    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
    
        # Appliquer le seuil et vérifier si le label correspond à une personne
        if score > THRESHOLD and label == PERSON_LABEL:
            
            # Convertir le masque en tableau numpy et l'appliquer à l'image            
            mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255)  
            # print(mask)
            masks_retenus.append(mask)
            for c in range(3):
                img_np[:, :, c] = np.where(mask_np, 
                                        (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                        img_np[:, :, c])
    print(f"{len(masks_retenus)} personnes dans cette image, donc il aura {len(masks_retenus)} masks")
    save_masks_to_textfile(masks_retenus, imagename)
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))