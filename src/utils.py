import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_flow(flow):
    """
    Visualise le flot optique en HSV (teinte=direction, saturation=magnitude).
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    
    # Teinte (Direction)
    hsv[..., 0] = angle * 180 / np.pi / 2
    # Saturation au maximum
    hsv[..., 1] = 255
    # Valeur (Magnitude) normalisée entre 0 et 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Conversion en RGB pour l'affichage avec matplotlib ou cv2.imshow
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def test_alpha_influence(im1_path, im2_path, horn_schunck_func):
    """
    Teste différentes valeurs de alpha et affiche les résultats côte à côte.
    Takes the horn_schunck function as an argument to avoid circular imports
    if this function is moved to another file.
    """
    # Chargement des images en niveaux de gris
    im1 = cv2.imread(im1_path, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2_path, cv2.IMREAD_GRAYSCALE)
    
    if im1 is None or im2 is None:
        print("Erreur: Impossible de charger les images.")
        return
    
    # Redimensionnement pour accélérer les calculs 
    im1 = cv2.resize(im1, (640, 480))
    im2 = cv2.resize(im2, (640, 480))

    # Valeurs de alpha à tester
    alphas = [0.1, 1.0, 10.0, 50.0]
    
    plt.figure(figsize=(15, 8))
    
    # Affichage de l'image de référence
    plt.subplot(2, 3, 1)
    plt.imshow(im1, cmap='gray')
    plt.title("Image 1 (Référence)")
    plt.axis('off')

    # Calcul et affichage pour chaque alpha
    for i, alpha in enumerate(alphas):
        print(f"Calcul pour alpha = {alpha}...")
        u, v = horn_schunck_func(im1, im2, alpha=alpha, num_iter=150)
        
        # Formatage du flot pour la fonction de visualisation
        flow = np.stack((u, v), axis=-1)
        flow_rgb = visualize_flow(flow)
        
        plt.subplot(2, 3, i + 2)
        plt.imshow(flow_rgb)
        plt.title(f"Flot (alpha = {alpha})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()