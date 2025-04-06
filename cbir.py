import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra
from BiT import bio_taxo
import numpy as np
from descripteurs import glcm, haralick_feat, bitdesk_feat, concat, glcm_rgb, haralick_feat_rgb, bitdesk_feat_rgb, concat_rgb
import os

# Fonction pour choisir le descripteur en fonction des paramètres
def extraire_caracteristiques(image, methode="glcm", rgb=False):
    try:
        if rgb:
            if methode == "glcm":
                return glcm_rgb(image)
            elif methode == "haralick":
                return haralick_feat_rgb(image)
            elif methode == "bit":
                return bitdesk_feat_rgb(image)
            elif methode == "concat":
                return concat_rgb(image)
        else:
            if methode == "glcm":
                return glcm(image)
            elif methode == "haralick":
                return haralick_feat(image)
            elif methode == "bit":
                return bitdesk_feat(image)
            elif methode == "concat":
                return concat(image)
    except Exception as e:
        print(f"Erreur d'extraction pour {image}: {e}")
        return None

# Extraction des signatures pour toutes les images du dossier
def extraction_signatures(chemin_dossier, methode="glcm", rgb=False):

    print(f"Extraction des signatures {methode}{'_rgb' if rgb else ''} depuis {chemin_dossier}")
    liste_carac = []
    
    # Déterminer le nom du fichier de sortie
    suffix = "_rgb" if rgb else ""
    fichier_sortie = f"Signatures{methode.capitalize()}{suffix}"
    
    for root, dirs, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    # Utiliser un chemin relatif comme dans ExtractionSignatures
                    path_relative = os.path.relpath(os.path.join(root, file), chemin_dossier)
                    path = os.path.join(root, file)
                    
                    if not os.path.exists(path):
                        print(f"Chemin invalide : {path}")
                        continue
                    
                    carac = extraire_caracteristiques(path, methode, rgb)
                    if carac is not None:
                        liste_carac.append(carac + [path_relative])  
                        print(f"Signature extraite pour: {path_relative}")
                    else:
                        print(f"Impossible d'extraire les caractéristiques pour: {path_relative}")
                except Exception as e:
                    print(f"Erreur pour le fichier {file}: {e}")
    
    if not liste_carac:
        print("Aucune signature n'a pu être extraite!")
        return None
    
    signatures = np.array(liste_carac, dtype=object)
    np.save(f'{fichier_sortie}.npy', signatures)
    print(f"Fichier de signatures créé: {fichier_sortie}.npy avec {len(signatures)} signatures")
    
    # Créer aussi un CSV pour la visualisation
    try:
        df = pd.DataFrame(signatures)
        df.to_csv(f'{fichier_sortie}.csv', index=False)
        print(f"Fichier CSV créé: {fichier_sortie}.csv")
    except Exception as e:
        print(f"Erreur lors de la création du CSV: {e}")
    
    return f'{fichier_sortie}.npy'

# Recherche d'image
def rechercher_image(image_query, fichier_signatures, distance="euclidienne", k=5):

    try:
        signatures = np.load(fichier_signatures, allow_pickle=True)
        print(f"Fichier de signatures chargé: {fichier_signatures} avec {len(signatures)} signatures")
    except Exception as e:
        print(f"Erreur lors du chargement des signatures: {e}")
        return []
    
    # Déterminer quelle méthode et RGB/non-RGB a été utilisée
    methode = "glcm"
    rgb = False
    
    if "Haralick" in fichier_signatures:
        methode = "haralick"
    elif "Bit" in fichier_signatures:
        methode = "bit"
    elif "Concat" in fichier_signatures:
        methode = "concat"
    
    if "rgb" in fichier_signatures.lower():
        rgb = True
    
    print(f"Utilisation de la méthode: {methode}, RGB: {rgb}")
    
    # Extraire les caractéristiques de l'image requête
    query_features = extraire_caracteristiques(image_query, methode, rgb)
    if query_features is None:
        print(f"Impossible d'extraire les caractéristiques de l'image requête: {image_query}")
        return []
    
    resultats = []
    
    for signature in signatures:
        try:
            features = signature[:-1]  
            chemin_image = signature[-1]  
            
            if len(query_features) != len(features):
                print(f"Incompatibilité de dimensions: Query={len(query_features)}, Stockée={len(features)}")
                continue
            
            dist = calculer_distance(query_features, features, methode=distance)
            resultats.append((chemin_image, dist))
        except Exception as e:
            print(f"Erreur lors de la comparaison avec {chemin_image if 'chemin_image' in locals() else 'une image'}: {e}")
    
    resultats.sort(key=lambda x: x[1])  
    print(f"{len(resultats)} résultats trouvés, retour des {min(k, len(resultats))} premiers")
    return resultats[:k]  

# Calcul des distances
def calculer_distance(feature1, feature2, methode="euclidienne"):

    if methode == "euclidienne":
        return euclidean(feature1, feature2)
    elif methode == "manhattan":
        return cityblock(feature1, feature2)
    elif methode == "tchebychev":
        return chebyshev(feature1, feature2)
    elif methode == "canberra":
        return canberra(feature1, feature2)
    else:
        raise ValueError("Méthode de distance non reconnue")