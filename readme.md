# Documentation: Application Web CBIR avec Authentification

## Introduction

Cette application web combine un système de recherche d'images basée sur le contenu (CBIR) avec une authentification sécurisée intégrant la reconnaissance faciale. Elle permet aux utilisateurs de s'inscrire, de se connecter et de rechercher des images similaires à partir d'un dataset selon différentes méthodes d'extraction de caractéristiques et de mesures de distance.

## Aperçu du système

L'application est composée de deux modules principaux:

1. **Module d'authentification** - Gère l'inscription et la connexion des utilisateurs avec reconnaissance faciale
2. **Module CBIR** - Permet la recherche d'images similaires basée sur le contenu

## Choix techniques

### Technologies utilisées

- **Framework Web**: Streamlit (interface utilisateur simple et rapide à développer)
- **Base de données**: SQLite (légère et sans serveur, idéale pour cette application)
- **Traitement d'images**: OpenCV, face_recognition
- **Extraction de caractéristiques**: GLCM, Haralick, BiT
- **Calcul de similarité**: Distances euclidienne, Manhattan, Tchebychev, Canberra

### Structure du projet

```
├── auth.py            # Fonctions d'authentification
├── cbir.py            # Fonctions de recherche d'images
├── db.py              # Gestion de la base de données
├── descripteurs.py    # Calcul des descripteurs d'images
├── main.py            # Point d'entrée de l'application
├── utils.py           # Fonctions utilitaires
├── dataSet/           # Dossier contenant les images
└── users.db           # Base de données SQLite
```

## Étapes de développement

### 1. Mise en place de la base de données

La base de données SQLite a été créée pour stocker les informations utilisateur:
- Nom d'utilisateur
- Email
- Mot de passe (hashé)
- Descripteur facial (stocké en BLOB)

Le schéma de table permet de stocker efficacement les vecteurs d'encodage facial pour l'authentification biométrique.

### 2. Développement du module d'authentification

#### Inscription

Le processus d'inscription inclut:
1. Collecte des informations utilisateur (nom, email, mot de passe)
2. Capture et prétraitement d'une image du visage
3. Extraction des encodages faciaux
4. Hashage du mot de passe
5. Stockage des données dans la base SQLite

#### Connexion

Deux étapes d'authentification sont implémentées:
1. Vérification de l'email/mot de passe
2. Reconnaissance faciale via la webcam

### 3. Développement du module CBIR

#### Extraction des caractéristiques

Quatre méthodes d'extraction sont implémentées:
- **GLCM** (Gray-Level Co-occurrence Matrix) - Textures
- **Haralick** - Caractéristiques statistiques
- **BiT** (Bio-Inspired Texture descriptor) - Descripteur biomimétique
- **Concat** - Concaténation des trois descripteurs ci-dessus

Les signatures peuvent être calculées sur l'image en niveaux de gris ou en RGB.

#### Calcul de similarité

Quatre mesures de distance sont implémentées:
- Distance euclidienne
- Distance de Manhattan
- Distance de Tchebychev
- Distance de Canberra

#### Interface de recherche

L'interface de recherche permet à l'utilisateur de:
1. Téléverser une image requête
2. Sélectionner la méthode d'extraction des caractéristiques
3. Choisir d'utiliser l'analyse RGB ou en niveaux de gris
4. Sélectionner la mesure de distance
5. Définir le nombre de résultats à afficher (k)

### 4. Intégration et Interface utilisateur

L'interface utilisateur a été développée avec Streamlit et offre:
- Navigation entre inscription, connexion et recherche d'images
- Gestion de session pour maintenir l'état connecté/déconnecté
- Interface responsive pour l'affichage des résultats de recherche

## Instructions d'installation et d'exécution

### Prérequis

```bash
# Installation des dépendances
pip install streamlit opencv-python numpy pandas scikit-image mahotas scipy face_recognition
```

### Préparation

1. Créez un dossier `dataSet` à la racine du projet et ajoutez-y les images pour la recherche
2. Assurez-vous que tous les fichiers Python sont présents

### Exécution

```bash
# Lancez l'application
streamlit run main.py
```

### Utilisation

1. **Inscription**: Accédez à l'onglet "Inscription", remplissez le formulaire et téléversez une image de votre visage
2. **Connexion**: Fournissez votre email/mot de passe et effectuez la vérification faciale
3. **Recherche d'images**: Téléversez une image requête, sélectionnez les paramètres de recherche et visualisez les résultats

## Détails techniques

### Authentification

- **Sécurité des mots de passe**: Les mots de passe sont hachés avec SHA-256 pour éviter le stockage en clair
- **Tolérance de reconnaissance faciale**: Configurée à 0.7 pour un bon équilibre entre sécurité et convivialité
- **Prétraitement des images**: Redimensionnement et conversion des espaces colorimétriques pour une meilleure reconnaissance

### Recherche CBIR

- **Génération automatique de signatures**: Si les fichiers de signatures n'existent pas, ils sont générés à la volée
- **Gestion des chemins**: Utilisation de chemins relatifs pour une meilleure portabilité
- **Stockage optimisé**: Les signatures sont stockées en format .npy et .csv

## Points d'amélioration possibles

1. Implémentation de l'authentification via les réseaux sociaux (Google, Facebook)
2. Optimisation des performances pour les grandes collections d'images
3. Ajout d'une interface d'administration pour gérer les utilisateurs
4. Mise en place d'une méthode de récupération de mot de passe
5. Amélioration de l'interface utilisateur et des visualisations des résultats

## Conclusion

Cette application démontre l'intégration réussie de techniques d'authentification avancées avec un système CBIR fonctionnel. Elle offre une solution complète pour la recherche d'images basée sur le contenu avec une couche de sécurité efficace.