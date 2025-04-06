import sqlite3
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from auth import enregistrer_utilisateur, authentification_par_facial, authentifier_utilisateur
from cbir import extraction_signatures, rechercher_image
from db import creer_base_donnees, verifier_structure_projet
from utils import preprocess_image_for_face_recognition
import os
import tempfile

# Fonction pour initialiser la session
def init_session():
    if "connected" not in st.session_state:
        st.session_state["connected"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None

def interface_inscription():
    st.title("Inscription")
    nom_utilisateur = st.text_input("Nom d'utilisateur")
    email = st.text_input("Email")
    mot_de_passe = st.text_input("Mot de passe", type="password")
    image_upload = st.file_uploader("Téléversez une image pour l'authentification faciale", type=["jpg", "png"])
    
    if st.button("S'inscrire"):
        if not nom_utilisateur.strip():
            st.error("Le champ 'Nom d'utilisateur' est vide.")
        elif not email.strip():
            st.error("Le champ 'Email' est vide.")
        elif not mot_de_passe.strip():
            st.error("Le champ 'Mot de passe' est vide.")
        elif not image_upload:
            st.error("Veuillez téléverser une image.")
        else:
            try:
                image = cv2.imdecode(np.frombuffer(image_upload.read(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    st.error("Impossible de lire l'image téléversée.")
                    return
                
                enregistrer_utilisateur(nom_utilisateur, email, mot_de_passe, image)
                st.success("Inscription réussie ! Vous pouvez maintenant vous connecter.")
            except Exception as e:
                st.error(f"Erreur lors de l'inscription : {e}")

def interface_connexion():
    if "connected" in st.session_state and st.session_state["connected"]:
        st.success(f"Vous êtes déjà connecté en tant que {st.session_state['user']}.")
        if st.button("Se déconnecter"):
            del st.session_state["connected"]
            del st.session_state["user"]
            st.experimental_rerun()
        return

    st.title("Connexion")
    email = st.text_input("Email")
    mot_de_passe = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if email and mot_de_passe:
            if not authentifier_utilisateur(email, mot_de_passe):
                st.error("Email ou mot de passe incorrect.")
                return

            st.success("Email et mot de passe vérifiés.")
            st.info("Veuillez vous placer devant la caméra pour vérification faciale.")

            try:
                capture = cv2.VideoCapture(0)
                if not capture.isOpened():
                    st.error("Impossible d'accéder à la caméra.")
                    return

                ret, frame = capture.read()
                capture.release()

                if not ret:
                    st.error("Erreur lors de la capture d'image.")
                    return
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                utilisateur = authentification_par_facial(rgb_frame)
                if utilisateur and utilisateur == email:
                    st.session_state["connected"] = True
                    st.session_state["user"] = email
                    st.success(f"Connexion réussie ! Bienvenue {utilisateur}.")
                    st.experimental_rerun()
                else:
                    st.error("Reconnaissance faciale échouée ou utilisateur non correspondant.")
            except Exception as e:
                st.error(f"Erreur lors de la reconnaissance faciale : {e}")
        else:
            st.error("Veuillez fournir un email et un mot de passe.")

def inspecter_utilisateurs():
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT nom_utilisateur, email, descripteur_facial FROM utilisateurs")
        utilisateurs = cursor.fetchall()
        print(f"Nombre d'utilisateurs: {len(utilisateurs)}")
        for utilisateur in utilisateurs:
            print(f"Nom: {utilisateur[0]}, Email: {utilisateur[1]}, Descripteur: {len(utilisateur[2]) if utilisateur[2] else 'Non défini'}")
        conn.close()
    except Exception as e:
        print(f"Erreur lors de l'inspection des utilisateurs: {e}")


def interface_application():
    if not st.session_state["connected"]:
        st.warning("Veuillez vous connecter pour accéder à la recherche d'images.")
        return
    
    st.title("Recherche d'images basée sur le contenu")
    
    # Vérifier si le dossier dataSet existe
    if not os.path.exists("./dataSet"):
        st.error("Le dossier 'dataSet' n'existe pas. Veuillez le créer et y ajouter des images.")
        return
    
    # Interface pour le choix des méthodes d'extraction
    col1, col2 = st.columns(2)
    
    with col1:
        # Choix de la signature
        signature_type = st.radio(
            "Type de signature",
            ["GLCM", "Haralick", "BIT", "Concat (tous)"],
            index=0
        )
        
        # Utiliser RGB ou non
        use_rgb = st.checkbox("Utiliser RGB", value=True)
    
    with col2:
        # Choix de la méthode de distance
        distance = st.selectbox(
            "Méthode de distance", 
            ["euclidienne", "manhattan", "tchebychev", "canberra"],
            index=0
        )
        
        # Nombre de résultats à retourner
        k_results = st.slider("Nombre de résultats (k)", 1, 20, 5)
    
    # Convertir le choix en valeur pour la fonction
    methode_map = {
        "GLCM": "glcm", 
        "Haralick": "haralick", 
        "BIT": "bit", 
        "Concat (tous)": "concat"
    }
    methode = methode_map[signature_type]
    
    # Nom du fichier de signatures attendu
    suffix = "_rgb" if use_rgb else ""
    signatures_file = f"Signatures{signature_type.capitalize().replace(' (tous)', '')}{suffix}.npy"
    
    # Information sur les signatures disponibles
    st.info(f"Utilisation du descripteur: {signature_type}{' (RGB)' if use_rgb else ''}")
    
    # Téléchargement de l'image de requête
    image_query = st.file_uploader("Téléversez une image pour la recherche", type=["jpg", "png", "jpeg", "bmp"])

    if image_query:
        try:
            # Lire l'image téléchargée
            image = cv2.imdecode(np.frombuffer(image_query.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("Impossible de charger l'image.")
                return
            
            # Afficher l'image de requête
            st.image(image, caption="Image de requête")
            
            # Sauvegarder temporairement l'image pour le traitement
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)
            
            # Vérifier si le fichier de signatures existe sinon le générer
            if not os.path.exists(signatures_file):
                st.warning(f"Base de signatures '{signatures_file}' non trouvée. Génération en cours...")
                with st.spinner("Extraction des caractéristiques..."):
                    try:
                        signatures_file = extraction_signatures("./dataSet", methode, use_rgb)
                        if signatures_file:
                            st.success(f"Signatures générées avec succès!")
                        else:
                            st.error("Échec de la génération des signatures.")
                            return
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des signatures : {e}")
                        return
            
            # Recherche d'images
            with st.spinner("Recherche en cours..."):
                resultats = rechercher_image(temp_path, signatures_file, distance, k_results)
            
            if not resultats:
                st.warning("Aucun résultat trouvé.")
            else:
                st.subheader(f"Top {len(resultats)} résultats")
                
                # Déterminer le nombre de colonnes
                num_cols = min(5, len(resultats))
                cols = st.columns(num_cols)
                
                for i, (chemin, score) in enumerate(resultats):
                    col_index = i % num_cols
                    with cols[col_index]:
                        try:
                            img_path = os.path.join("./dataSet", chemin)
                            if os.path.exists(img_path):
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    st.image(img, caption=f"Distance: {score:.2f}")
                                    st.text(os.path.basename(chemin))
                                else:
                                    st.error(f"Impossible de charger {os.path.basename(chemin)}")
                            else:
                                st.error(f"Fichier non trouvé: {os.path.basename(chemin)}")
                        except Exception as e:
                            st.error(f"Erreur: {e}")
        except Exception as e:
            st.error(f"Erreur lors de la recherche : {e}")
        finally:
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                except:
                    pass

if __name__ == "__main__":
    # Initialisation de la base de données
    creer_base_donnees()
    verifier_structure_projet()
    
    # Interface Streamlit
    init_session()
    st.sidebar.title("Navigation")
    choix = st.sidebar.radio("Choisissez une option", ["Inscription", "Connexion", "Recherche d'images"])
    
    if choix == "Inscription":
        interface_inscription()
    elif choix == "Connexion":
        interface_connexion()
    elif choix == "Recherche d'images":
        interface_application()