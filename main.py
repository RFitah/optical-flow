import argparse
import cv2
import numpy as np

# Importation de modules depuis src
from src.tv_l1 import create_tvl1_algorithm, calculate_tvl1
from src.horn_schunck import horn_schunck, multiscale_horn_schunck
from src.lucas_kanade import calculate_lucas_kanade, visualize_lucas_kanade
from src.utils import visualize_flow

def main():

    # 1. Configuration des arguments du terminal
    parser = argparse.ArgumentParser(description="Projet OPTICAL : Estimation du flot optique sur vidéos réelles.")
    parser.add_argument('--input', type=str, required=True, 
                        help="Chemin vers le fichier vidéo (ex: data/raw/cafe.mp4)")
    parser.add_argument('--algo', type=str, 
                        choices=['lucas_kanade', 'horn_schunck', 'multiscale', 'tvl1'], 
                        default='tvl1', 
                        help="L'algorithme de flot optique à utiliser (défaut: tvl1)")
    parser.add_argument('--width', type=int, default=320, 
                        help="Largeur de redimensionnement pour accélérer les calculs (défaut: 320)")
    
    args = parser.parse_args()

    # 2. Initialisation de la capture vidéo
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Erreur : Impossible de lire la vidéo '{args.input}'. Vérifiez le chemin.")
        return

    ret, frame1 = cap.read()
    if not ret:
        print("Erreur : La vidéo est vide ou illisible.")
        return

    # Redimensionnement de la première frame pour les performances
    h, w = frame1.shape[:2]
    new_dim = (args.width, int(h * (args.width / float(w))))
    prvs = cv2.cvtColor(cv2.resize(frame1, new_dim), cv2.COLOR_BGR2GRAY)

    # 3. Initialisations spécifiques selon l'algorithme choisi
    p0 = None
    if args.algo == 'lucas_kanade':
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p0 = cv2.goodFeaturesToTrack(prvs, mask=None, **feature_params)
        mask_lk = np.zeros_like(cv2.resize(frame1, new_dim))
        color = np.random.randint(0, 255, (100, 3))
    elif args.algo == 'tvl1':
        tvl1_algo = create_tvl1_algorithm()

    print("\n--- Démarrage de l'analyse ---")
    print(f"Vidéo      : {args.input}")
    print(f"Algorithme : {args.algo}")
    print("------------------------------")
    print("Appuyez sur la touche 'q' pour quitter la visualisation.\n")

    # 4. Boucle de traitement image par image
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
            
        frame2_resized = cv2.resize(frame2, new_dim)
        next_frame = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
        vis_frame = frame2_resized.copy()

        # Calcul du flot selon l'algorithme
        if args.algo == 'tvl1':
            flow = calculate_tvl1(tvl1_algo, prvs, next_frame)
            display_img = visualize_flow(flow)

        elif args.algo == 'multiscale':
            u, v = multiscale_horn_schunck(prvs, next_frame, num_levels=3, alpha=1.0)
            display_img = visualize_flow(np.stack((u, v), axis=-1))

        elif args.algo == 'horn_schunck':
            # Méthode classique L2 Mono-échelle (avec un alpha par défaut)
            u, v = horn_schunck(prvs, next_frame, alpha=10.0, num_iter=50)
            display_img = visualize_flow(np.stack((u, v), axis=-1))

        elif args.algo == 'lucas_kanade':
            good_new, good_old, status = calculate_lucas_kanade(prvs, next_frame, p0, lk_params)
            display_img, p0, mask_lk = visualize_lucas_kanade(vis_frame, mask_lk, good_new, good_old, color)
            
            # Si l'algorithme perd trop de points, on les réinitialise pour éviter un crash
            if p0 is None or len(p0) < 10:
                p0 = cv2.goodFeaturesToTrack(prvs, mask=None, **feature_params)

        # Affichage (Vidéo d'origine à gauche, Flot optique à droite)
        combined = np.hstack((frame2_resized, display_img))
        cv2.imshow(f'Flot Optique - {args.algo}', combined)

        # Mise à jour pour la prochaine itération
        prvs = next_frame

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()
    print("Analyse terminée.")

if __name__ == "__main__":
    main()