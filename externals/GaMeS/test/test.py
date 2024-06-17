import os
print(os.environ['CONDA_DEFAULT_ENV'])
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def load_and_display_image(image_path_person, image_path_bg, to_tensor=False):
    # Ładowanie obrazka za pomocą PIL
    person = Image.open(image_path_person)
    bg = Image.open(image_path_bg)

    # Konwersja do tablicy NumPy
    person_array = np.array(person)
    bg_array = np.array(bg)
    #image = person_array - bg_array
    image = bg_array - person_array
    print(image)
    threshold = 40
    image = np.abs(image)
    mask = np.max(image, axis=-1) > threshold
    mask = mask[:, :, np.newaxis]
    
    mask = mask.astype(np.uint8)
    image_array = person_array * mask

    if to_tensor:
        # Konwersja do tensora PyTorch
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # Zamiana wymiarów do formatu CxHxW
        # Wyświetlanie obrazu z tensora
        plt.imshow(image_tensor.permute(1, 2, 0).numpy())
        plt.title("Image displayed as PyTorch tensor")
    else:
        # Wyświetlanie obrazu z tablicy NumPy
        plt.imshow(image_array)
        plt.title("Image displayed as NumPy array")

    plt.axis('off')  # Usunięcie osi
    plt.show()

# Przykład użycia
person_path = 'f_0020.png'
bg_path = '00050.png'
load_and_display_image(person_path, bg_path, to_tensor=False)  # Zmiana na to_tensor=True jeśli chcesz tensor PyTorch