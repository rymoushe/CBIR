from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo
import cv2
import numpy as np
 
def glcm(chemin):
    data = cv2.imread(chemin, 0)
    co_matrice = graycomatrix(data, [1], [0], None, symmetric=False, normed=False)
    contrast = float(graycoprops(co_matrice, 'contrast')[0, 0])
    dissimilarity = float(graycoprops(co_matrice, 'dissimilarity')[0, 0])
    correlation = float(graycoprops(co_matrice, 'correlation')[0, 0])
    homogeneity = float(graycoprops(co_matrice, 'homogeneity')[0, 0])
    ASM = float(graycoprops(co_matrice, 'ASM')[0, 0])
    energy = float(graycoprops(co_matrice, 'energy')[0, 0])
    return [contrast, dissimilarity, correlation, homogeneity, ASM, energy]
 
 
def haralick_feat(chemin):
    data = cv2.imread(chemin, 0)
    features = haralick(data).mean(0).tolist()
    features = [float(x) for x in features]
    return features
 
def bitdesk_feat(chemin):
    data = cv2.imread(chemin, 0)
    features = bio_taxo(data)
    features = [float(x) for x in features]
    return features
 
# Concatenation des trois--------------
 
def concat(chemin):
    return glcm(chemin) + haralick_feat(chemin) + bitdesk_feat(chemin)
   
#---------------------RGB-------------------------------
 
def glcm_rgb(chemin):
    data = cv2.imread(chemin)
    img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    canals = ['R', 'G', 'B']
    carac = []
    for i, canal_name in enumerate(canals):
        canal = img_rgb[:, :, i]
        co_matrice = graycomatrix(canal, [1], [0], None, symmetric=False, normed=False)
        contrast = float(graycoprops(co_matrice, 'contrast')[0, 0])
        dissimilarity = float(graycoprops(co_matrice, 'dissimilarity')[0, 0])
        correlation = float(graycoprops(co_matrice, 'correlation')[0, 0])
        homogeneity = float(graycoprops(co_matrice, 'homogeneity')[0, 0])
        ASM = float(graycoprops(co_matrice, 'ASM')[0, 0])
        energy = float(graycoprops(co_matrice, 'energy')[0, 0])
        features = [contrast, dissimilarity, correlation, homogeneity, ASM, energy]
        carac.extend(features)      
    return carac
 
def haralick_feat_rgb(chemin):
    data = cv2.imread(chemin)
    img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    canals = ['R', 'G', 'B']
    carac = []
    for i, _ in enumerate(canals):
        canal = img_rgb[:, :, i]
        features = haralick(canal).mean(0).tolist()
        features = [float(x) for x in features]
        carac.extend(features)      
    return carac
 
def bitdesk_feat_rgb(chemin):
    data = cv2.imread(chemin)
    img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    canals = ['R', 'G', 'B']
    carac = []
    for i, canal_name in enumerate(canals):
        canal = img_rgb[:, :, i]
        features = bio_taxo(canal)
        features = [float(x) for x in features]
        carac.extend(features)      
    return carac
 
# Concatenation des trois--------------
 
def concat_rgb(chemin):
    return glcm_rgb(chemin) + haralick_feat_rgb(chemin) + bitdesk_feat_rgb(chemin)