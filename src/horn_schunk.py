import numpy as np
import cv2
from scipy.ndimage import convolve

def compute_derivatives(im1, im2):
    """
    Calcule les gradients spatio-temporels (Ix, Iy, It) en utilisant
    les masques de convolution classiques de Horn-Schunck.
    """
    # Noyaux de Horn-Schunck pour approximer les dérivées
    kernel_X = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernel_Y = np.array([[-1, -1], [1, 1]]) * 0.25
    kernel_T = np.ones((2, 2)) * 0.25

    # Application des filtres de convolution
    Ix = convolve(im1, kernel_X) + convolve(im2, kernel_X)
    Iy = convolve(im1, kernel_Y) + convolve(im2, kernel_Y)
    It = convolve(im2, kernel_T) - convolve(im1, kernel_T)
    
    return Ix, Iy, It

def horn_schunck(im1, im2, alpha=1.0, num_iter=100):
    """
    Estime le flot optique entre deux images par la méthode de Horn-Schunck
    en utilisant un schéma itératif de Jacobi.
    
    Paramètres:
    - im1, im2 : Images en niveaux de gris (tableaux numpy 2D)
    - alpha : Paramètre de régularisation (poids du lissage)
    - num_iter : Nombre d'itérations du schéma de Jacobi
    
    Retourne:
    - u, v : Composantes horizontale et verticale du flot optique
    """
    # Conversion en flottants pour la précision des calculs
    im1 = im1.astype(np.float32) / 255.0
    im2 = im2.astype(np.float32) / 255.0

    # Calcul des gradients
    Ix, Iy, It = compute_derivatives(im1, im2)

    # Initialisation des composantes du flot (u et v) à zéro
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    # Noyau pour calculer la moyenne locale (approximation du Laplacien)
    # Les pixels diagonaux ont un poids de 1/12, les adjacents de 1/6
    kernel_avg = np.array([[1/12, 1/6, 1/12],
                           [1/6,    0, 1/6],
                           [1/12, 1/6, 1/12]], dtype=np.float32)

    # Boucle itérative de Jacobi
    for _ in range(num_iter):
        # Calcul des moyennes locales u_bar et v_bar
        u_avg = convolve(u, kernel_avg)
        v_avg = convolve(v, kernel_avg)

        # Calcul du terme de mise à jour (fraction dans l'équation)
        numerator = (Ix * u_avg + Iy * v_avg + It)
        denominator = (alpha + Ix**2 + Iy**2)
        
        update_term = numerator / denominator

        # Mise à jour de u et v
        u = u_avg - Ix * update_term
        v = v_avg - Iy * update_term

    return u, v

def warp_image(img, u, v):
    """
    Déforme (warp) l'image 'img' en utilisant le champ de flot (u, v).
    """
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    map_x = (x + u).astype(np.float32)
    map_y = (y + v).astype(np.float32)
    
    warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped_img

def multiscale_horn_schunck(im1, im2, num_levels=3, alpha=1.0, num_iter=50):
    """
    Implémente la méthode multi-échelle coarse-to-fine.
    """
    # Construction des pyramides
    pyramid1 = [im1.astype(np.float32) / 255.0]
    pyramid2 = [im2.astype(np.float32) / 255.0]
    
    for i in range(1, num_levels):
        pyramid1.append(cv2.pyrDown(pyramid1[-1]))
        pyramid2.append(cv2.pyrDown(pyramid2[-1]))

    # Initialisation du flot a zero au sommet de la pyramide
    u = np.zeros(pyramid1[-1].shape, dtype=np.float32)
    v = np.zeros(pyramid2[-1].shape, dtype=np.float32)

    # Descente de la pyramide
    for level in range(num_levels - 1, -1, -1):
        img1_lvl = pyramid1[level]
        img2_lvl = pyramid2[level]

        if level != num_levels - 1:
            # Sur-echantillonnage du flot et multiplication par 2
            u = cv2.pyrUp(u) * 2.0
            v = cv2.pyrUp(v) * 2.0
            
            # Ajustement strict des dimensions (largeur, hauteur)
            u = cv2.resize(u, (img1_lvl.shape[1], img1_lvl.shape[0]))
            v = cv2.resize(v, (img1_lvl.shape[1], img1_lvl.shape[0]))

        # Recalage de l'image 2
        img2_warped = warp_image(img2_lvl, u, v)

        # Calcul du residu
        du, dv = horn_schunck(img1_lvl * 255, img2_warped * 255, alpha=alpha, num_iter=num_iter)

        # Mise a jour
        u += du
        v += dv

    return u, v