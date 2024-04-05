import os
from PIL import Image
from utils import model, tools
import torch

def remove_png_extension(filename):
    if filename.endswith('.png'):
        return filename[:-4]  # Remove last 4 characters (i.e., '.png')
    else:
        return filename

# Point d'entrée principal du script
if __name__ == "__main__":

    # Définir les répertoires source et de sortie, et le nom de l'image
    source_path_dir = "images/cam0"
    output_path_dir = "examples/output/cam0"
    # image_name = "1637433772063999700.png"

    counter = 0
    
    for file in os.listdir(source_path_dir):
        if file.endswith(".png"):
            counter += 1

            image_name = file
            print(f"===========================================================")
            print(f"Segmenting {counter}: {image_name}")

            # Charger le modèle et appliquer les transformations à l'image
            seg_model, transforms = model.get_model()

            # Ouvrir l'image et appliquer les transformations
            image_path = os.path.join(source_path_dir, image_name)
            image = Image.open(image_path)
            transformed_img = transforms(image)
            

            # Effectuer l'inférence sur l'image transformée sans calculer les gradients
            with torch.no_grad():
                output = seg_model([transformed_img])
                # print(output)

            imagename_no_ext = remove_png_extension(image_name)

            # Traiter le résultat de l'inférence
            result = tools.process_inference(output,image,imagename_no_ext)

            result = tools.apply_saved_mask(image, imagename_no_ext)

            result.save(os.path.join(output_path_dir, image_name))
            
            # result.show()
        else:
            break
            