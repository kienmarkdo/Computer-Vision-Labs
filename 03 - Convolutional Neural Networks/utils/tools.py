import numpy as np
import os
from utils.config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image
from torch import Tensor

# Fonction pour traiter les sorties d'inférence du modèle
def process_inference(model_output, image, imagename: str):
    np_masks = []
    # Extraire les masques, les scores, et les labels de la sortie du modèle
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']

    # Convertir l'image en tableau numpy
    img_np = np.array(image)

    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
    
        # Appliquer le seuil et vérifier si le label correspond à une personne
        if score > THRESHOLD and label == PERSON_LABEL:
            
            # Convertir le masque en tableau numpy et l'appliquer à l'image            
            mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255)  
            np_masks.append(mask_np)   

            for c in range(3):
                img_np[:, :, c] = np.where(mask_np, 
                                        (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                        img_np[:, :, c])
    # print(f"{len(masks_retenus)} personnes dans cette image, donc il aura {len(masks_retenus)} masks")
    with open(f'examples/output/cam0_saved_masks/{imagename}.npy', 'wb') as f:
        np.save(f,np_masks)
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))


def apply_saved_mask(image, imagename):

    # Convertir l'image en tableau numpy
    img_np = np.array(image)
    masks = np.load(f'examples/output/cam0_saved_masks/{imagename}.npy')
    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, mask in enumerate(masks):  
        for c in range(3):
            img_np[:, :, c] = np.where(mask, 
                                    (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                    img_np[:, :, c])
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer            
    return Image.fromarray(img_np.astype(np.uint8))