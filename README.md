# Projet OPTICAL : Estimation du Flot Optique

> **Master 1 Mathématiques Appliquées (UM4MA266)**  
> **Auteur :** Fitahiry Rajaobelina  

Ce projet explore la modélisation mathématique, l'implémentation numérique et l'analyse critique de l'estimation du flot optique en vision par ordinateur. Il propose une progression depuis les modèles variationnels classiques avec régularisation de Tikhonov (norme $L^2$) jusqu'aux algorithmes d'optimisation non-lisse basés sur la Variation Totale (norme $L^1$).

## Algorithmes implémentés et étudiés

Le projet compare plusieurs approches mathématiques face aux défis du monde réel (problème d'ouverture, grands déplacements, discontinuités géométriques, dynamique des fluides) :

1. **Horn-Schunck Classique (Norme $L^2$)** : Approche globale et dense. Implémentation maison du schéma itératif de Jacobi par différences finies.
2. **Horn-Schunck Multi-échelle (Coarse-to-fine)** : Utilisation d'une pyramide gaussienne pour surmonter les limites de l'approximation de Taylor face aux grands déplacements physiques.
3. **Lucas-Kanade** : Approche locale et éparse avec suivi des points d'intérêt (algorithme d'OpenCV).
4. **Dual TV- $L^1$ (Variation Totale)** : Algorithme proximal avancé permettant de préserver les discontinuités nettes des objets (absence de sur-lissage) et d'agir comme un estimateur robuste face aux reflets spéculaires.

## Architecture du Projet

```text
.
├── data/                       # Séquences de test 
│   ├── frames/                 # Paires d'images pour tests unitaires (ex: pizzas)
│   └── raw/                    # Vidéos réelles (ballequiroule.mp4, cafe.mp4, robinet.mp4)
├── docs/                       # Documentation académique
│   ├── presentation/           # Slides de soutenance
│   └── rapport/                
│       └── RAJAOBELINA_Fitahiry_Projet_Optical_Rapport.pdf
├── src/                        # Code source modulaire
│   ├── horn_schunck.py         # Implémentation L2 (Standard et Multi-échelle)
│   ├── lucas_kanade.py         # Wrapper pour Lucas-Kanade
│   ├── tv_l1.py                # Wrapper pour l'algo Dual TV-L1 d'OpenCV
│   └── utils.py                # Outils de visualisation HSV
├── main.py                     # Script d'exécution en ligne de commande (CLI)
├── requirements.txt            # Dépendances Python
└── README.md
```

## Installation

Pour exécuter ce projet localement, clonez le dépôt et installez les dépendances requises via `pip`. Il est recommandé d'utiliser un environnement virtuel.

```bash
# Cloner le dépôt
git clone [https://github.com/votre-pseudo/optical-flow.git](https://github.com/votre-pseudo/optical-flow.git)
cd optical-flow

# Installer les dépendances (NumPy, OpenCV, SciPy, Matplotlib)
pip install -r requirements.txt
```

## Utilisation (CLI)

Le point d'entrée principal du projet est le script `main.py`. Il vous permet de tester les différents algorithmes sur les vidéos fournies dans le dossier `data/raw/` ou sur vos propres vidéos.

**Syntaxe de base :**
```bash
python main.py --input <chemin_vers_la_video> --algo <nom_de_l_algorithme>
```

### Exemples d'exécution :

* **1. Analyser la préservation des bords nets sur un fluide (Variation Totale) :**
  ```bash
  python main.py --input data/raw/cafe.mp4 --algo tvl1
  ```

* **2. Tester la résilience aux grands déplacements (Horn-Schunck Multi-échelle) :**
  ```bash
  python main.py --input data/raw/ballequiroule.mp4 --algo multiscale
  ```

* **3. Extraire des trajectoires éparses (Lucas-Kanade) :**
  ```bash
  python main.py --input data/raw/robinet.mp4 --algo lucas_kanade
  ```

*Note : Pendant la lecture de la vidéo, appuyez sur la touche `q` pour quitter la fenêtre de visualisation.*

## Documentation

L'analyse mathématique complète du problème d'ouverture, la discrétisation de l'opérateur Laplacien, l'influence du paramètre de régularisation $\alpha$ ainsi que les interprétations cinématiques des résultats sont disponibles dans le rapport détaillé :

👉 [Consulter le rapport complet (PDF)](docs/rapport/RAJAOBELINA_Fitahiry_Projet_Optical_Rapport.pdf)

---
*Projet réalisé en 2026 dans le cadre du Master 1 Mathématiques Appliquées*