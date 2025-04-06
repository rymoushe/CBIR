import sqlite3
import os
from utils import create_directory_if_not_exists

def creer_base_donnees():
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
       
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS utilisateurs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom_utilisateur TEXT UNIQUE,
            email TEXT UNIQUE,
            mot_de_passe TEXT,
            descripteur_facial BLOB,
            date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()

def verifier_structure_projet():
    dataset_path = create_directory_if_not_exists("./dataSet")
    
    if not os.listdir(dataset_path):
        print("Attention: Le dossier dataSet est vide. Veuillez y ajouter des images pour la recherche.")
    
    return True