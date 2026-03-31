import tensorflow as tf  # Importer TensorFlow pour charger et exécuter le modèle IA
import numpy as np  # Importer NumPy pour manipuler les tableaux et les pixels
from PIL import Image, ImageTk  # Importer PIL pour ouvrir les images et les convertir pour Tkinter
import tkinter as tk  # Importer Tkinter pour construire l'interface graphique
from tkinter import filedialog  # Importer filedialog pour ouvrir l'explorateur de fichiers

# ══════════════════════════════
# CHARGER L'IA
# ══════════════════════════════

model = tf.keras.models.load_model("model/solar_defect_model.keras")  # Charger le modèle de deep learning pré-entraîné depuis le disque

# CLASSES associe chaque index de classe à un nom de défaut, un niveau de danger et une couleur d'affichage
CLASSES = {
    0: ("Bird Drop",    "Moyen",    "#f39c12"),  # Classe 0 : fientes d'oiseaux, danger moyen, orange
    1: ("Snow Covered", "Faible",   "#2980b9"),  # Classe 1 : panneau enneigé, danger faible, bleu
    2: ("Crack",        "Sérieux",  "#e67e22"),  # Classe 2 : fissure, danger sérieux, orange foncé
    3: ("Dirty",        "Moyen",    "#d4ac0d"),  # Classe 3 : panneau sale, danger moyen, jaune
    4: ("Hotspot",      "Critique", "#e74c3c"),  # Classe 4 : point chaud, danger critique, rouge
    5: ("Normal",       "Aucun",    "#27ae60"),  # Classe 5 : panneau normal, aucun danger, vert
}


# ══════════════════════════════
# FENÊTRE
# ══════════════════════════════

root = tk.Tk()  # Créer la fenêtre principale de l'application
root.title("Solar Panel Inspector — AI Detection")  # Définir le titre de la fenêtre
root.configure(bg="white")  # Définir le fond en blanc
root.configure(bg="#fafafa")  # Remplacer le fond par un blanc-gris très clair

screen_width = root.winfo_screenwidth()  # Récupérer la largeur de l'écran en pixels
screen_height = root.winfo_screenheight()  # Récupérer la hauteur de l'écran en pixels

width = int(screen_width * 0.8)  # Définir la largeur de la fenêtre à 80% de l'écran
height = int(screen_height * 0.9)  # Définir la hauteur de la fenêtre à 90% de l'écran

x = (screen_width - width) // 2  # Calculer la position horizontale pour centrer la fenêtre
y = (screen_height - height) // 2  # Calculer la position verticale pour centrer la fenêtre

root.geometry(f"{width}x{height}+{x}+{y}")  # Appliquer la taille et la position à la fenêtre

# ══════════════════════════════
# HEADER
# ══════════════════════════════

header = tk.Frame(root, bg="#02385A", pady=15)  # Créer le cadre d'en-tête avec un fond bleu foncé
header.pack(fill="x")  # Étirer l'en-tête sur toute la largeur de la fenêtre

tk.Label(header,  # Créer un label dans l'en-tête
         text="☀  Solar Panel Inspector",  # Texte du titre principal avec icône soleil
         font=("Arial", 20, "bold"),  # Grande police en gras
         bg="#02385A", fg="white").pack()  # Fond identique à l'en-tête, texte blanc, et placement

tk.Label(header,  # Créer un label de sous-titre
         text="Inspection intelligente par Drone & Intelligence Artificielle",  # Texte du sous-titre
         font=("Arial", 11),  # Police plus petite et normale
         bg="#02385A", fg="white").pack(pady=(3, 0))  # Petit espacement en haut, aucun en bas

# ══════════════════════════════
# BOUTON
# ══════════════════════════════

btn_frame = tk.Frame(root, bg="white", pady=5)  # Créer un cadre pour contenir le bouton avec fond blanc
btn_frame.pack(fill="x")  # Étirer le cadre du bouton sur toute la largeur

# ══════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════

main = tk.Frame(root, bg="white")  # Créer le cadre principal avec fond blanc
main.pack(fill="x", expand=False, padx=40, pady=(10, 0), anchor="n")  # Placer le cadre avec marges et alignement en haut

spacer = tk.Frame(root, bg="white")  # Créer un cadre vide servant d'espaceur
spacer.pack(fill="both", expand=True)  # Étendre l'espaceur pour pousser le contenu vers le haut

# COLONNE GAUCHE — affichage de l'image
left = tk.Frame(main, bg="white")  # Créer le cadre de la colonne gauche
left.pack(side="left", fill="both", expand=True)  # Le placer à gauche et lui permettre de s'étendre

tk.Label(left,  # Créer un label pour la section image
         text="Image thermique",  # Texte du titre de la section
         font=("Arial", 13, "bold"),  # Police moyenne en gras
         bg="white", fg="#1a1a2e").pack(anchor="w", pady=(0, 8))  # Aligné à gauche avec espacement en bas

canvas_image = tk.Canvas(left,  # Créer un canvas pour afficher l'image chargée
                          width=500, height=400,  # Taille fixe du canvas
                          bg="#f0f0f0",  # Fond gris clair avant le chargement de l'image
                          highlightthickness=1,  # Afficher une fine bordure
                          highlightbackground="#dddddd")  # Couleur gris clair pour la bordure
canvas_image.pack(anchor="w", pady=(0, 20))  # Aligner le canvas à gauche avec espacement en bas

centre_x = 500 // 2  # Calculer le centre horizontal du canvas (250px)
centre_y = 400 // 2  # Calculer le centre vertical du canvas (200px)

canvas_image.create_text(centre_x, centre_y,  # Placer le texte d'espace réservé au centre du canvas
                          text="[ Aucune image chargée ]",  # Message d'espace réservé
                          font=("Arial", 13),  # Taille de police moyenne
                          fill="#999999")  # Couleur grise pour le texte d'espace réservé

# COLONNE DROITE — résultats de l'analyse
right = tk.Frame(main, bg="white", padx=50)  # Créer le cadre de la colonne droite avec marge gauche
right.pack(side="right", fill="both", expand=True)  # Le placer à droite et lui permettre de s'étendre

tk.Label(right,  # Créer un label pour la section résultats
         text="Résultat de l'analyse",  # Texte du titre de la section
         font=("Arial", 14, "bold"),  # Police moyenne-grande en gras
         bg="white", fg="#1a1a2e").pack(anchor="w", pady=(0, 15))  # Aligné à gauche avec espacement en bas

label_resultat = tk.Label(right,  # Créer le label pour afficher le nom du défaut détecté
                           text="Aucune analyse",  # Texte par défaut avant toute analyse
                           font=("Arial", 22, "bold"),  # Grande police en gras
                           bg="white", fg="#cccccc")  # Couleur grise pour l'état par défaut
label_resultat.pack(anchor="w")  # Aligner le label à gauche

label_danger = tk.Label(right,  # Créer le label pour afficher le niveau de danger
                         text="",  # Vide par défaut
                         font=("Arial", 13),  # Police moyenne
                         bg="white", fg="#999999")  # Couleur grise pour l'état par défaut
label_danger.pack(anchor="w", pady=(8, 20))  # Aligné à gauche avec espacement vertical

tk.Label(right,  # Créer un label statique pour la section de confiance
         text="Confiance du modèle :",  # Texte du label
         font=("Arial", 12, "bold"),  # Police moyenne en gras
         bg="white", fg="#1a1a2e").pack(anchor="w")  # Aligné à gauche

canvas_bar = tk.Canvas(right, width=500, height=30,  # Créer un canvas pour la barre de progression de confiance
                        bg="white", highlightthickness=0)  # Pas de bordure sur le canvas de la barre
canvas_bar.pack(anchor="w", pady=(8, 10))  # Aligné à gauche avec espacement vertical

tk.Frame(right, bg="#eeeeee", height=1).pack(fill="x", pady=8)  # Dessiner une fine ligne séparatrice horizontale

tk.Label(right,  # Créer un label statique pour la section des scores
         text="Scores par classe :",  # Texte du label
         font=("Arial", 12, "bold"),  # Police moyenne en gras
         bg="white", fg="#1a1a2e").pack(anchor="w", pady=(5, 5))  # Aligné à gauche avec espacement vertical

labels_scores = {}  # Dictionnaire pour stocker les références des labels de score par classe
for i, (nom, danger, couleur) in CLASSES.items():  # Parcourir chaque classe
    lbl = tk.Label(right,  # Créer un label de score pour cette classe
                   text=f"{nom}: —",  # Texte par défaut montrant le nom de la classe sans score
                   font=("Arial", 11),  # Police normale de taille moyenne
                   bg="white", fg="#cccccc")  # Couleur grise pour l'état par défaut
    lbl.pack(anchor="w", pady=2)  # Aligné à gauche avec petit espacement vertical
    labels_scores[i] = lbl  # Stocker la référence du label par index de classe


# ══════════════════════════════
# FOOTER
# ══════════════════════════════

footer = tk.Frame(root, bg="#f5f5f5", pady=8)  # Créer le cadre du pied de page avec fond gris clair
footer.pack(fill="x", side="bottom")  # Étirer le pied de page en bas de la fenêtre

tk.Label(footer,  # Créer un label dans le pied de page
         text="Projet PFA 25/26 — Inspection photovoltaïque par Drone & IA ",  # Texte du pied de page
         font=("Arial", 8),  # Petite police
         bg="#f5f5f5", fg="#555555").pack()  # Fond identique au pied de page, texte gris foncé


# ══════════════════════════════
# FONCTION ANALYSER
# ══════════════════════════════

def analyser_image():  # Définir la fonction exécutée quand l'utilisateur clique sur le bouton
    path = filedialog.askopenfilename(  # Ouvrir une boîte de dialogue pour choisir une image
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]  # Filtrer uniquement les fichiers image
    )
    if not path:  # Si l'utilisateur a annulé la boîte de dialogue
        return  # Quitter la fonction sans rien faire

    img_display = Image.open(path).resize((620, 560), Image.LANCZOS)  # Ouvrir et redimensionner l'image pour l'affichage
    img_tk = ImageTk.PhotoImage(img_display)  # Convertir l'image en format compatible Tkinter
    canvas_image.delete("all")  # Effacer tout le contenu précédent du canvas
    canvas_image.create_image(0, 0, anchor="nw", image=img_tk)  # Dessiner la nouvelle image sur le canvas
    canvas_image.image = img_tk  # Garder une référence pour éviter la suppression par le garbage collector

    img = Image.open(path).resize((224, 224)).convert("RGB")  # Redimensionner l'image à la taille d'entrée du modèle et forcer le mode RGB
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normaliser les pixels entre 0 et 1 et ajouter la dimension de lot
    prediction = model.predict(arr)  # Passer l'image dans le modèle pour obtenir les probabilités de chaque classe
    class_idx = np.argmax(prediction)  # Récupérer l'index de la classe avec le score le plus élevé
    confidence = prediction[0][class_idx] * 100  # Convertir le score de la meilleure classe en pourcentage
    nom, danger, couleur = CLASSES[class_idx]  # Récupérer le nom, le niveau de danger et la couleur de la classe prédite

    canvas_bar.delete("all")  # Effacer la barre de confiance précédente
    canvas_bar.create_rectangle(0, 0, 500, 30, fill="#eeeeee", outline="")  # Dessiner la barre de fond grise
    canvas_bar.create_rectangle(0, 0, int(500 * confidence / 100), 30,  # Dessiner la partie remplie de la barre
                                 fill=couleur, outline="")  # Remplir avec la couleur de la classe, sans bordure
    canvas_bar.create_text(250, 15,  # Placer le texte du pourcentage au centre de la barre
                            text=f"{confidence:.1f}%",  # Formater avec une décimale
                            font=("Arial", 11, "bold"),  # Police en gras
                            fill="white")  # Texte blanc pour la lisibilité

    label_resultat.config(text=f"Défaut : {nom}", fg=couleur)  # Mettre à jour le label du défaut avec le résultat et la couleur
    label_danger.config(text=f"Niveau de danger : {danger}", fg=couleur)  # Mettre à jour le label de danger avec le résultat et la couleur

    for i, (cls_nom, _, cls_couleur) in CLASSES.items():  # Parcourir toutes les classes pour mettre à jour leurs labels de score
        score = prediction[0][i] * 100  # Convertir le score de la classe en pourcentage
        labels_scores[i].config(  # Mettre à jour le label de cette classe
            text=f"{cls_nom}: {score:.1f}%",  # Afficher le nom de la classe et son score
            fg=cls_couleur  # Appliquer la couleur de la classe
        )

# Créer le bouton après la fonction pour qu'il puisse référencer analyser_image
tk.Button(btn_frame,  # Placer le bouton dans le cadre du bouton
          text="📂   Charger une image et analyser",  # Texte du bouton avec icône dossier
          command=analyser_image,  # Appeler analyser_image au clic
          font=("Arial", 13, "bold"),  # Police moyenne en gras
          bg="#378ADD", fg="white",  # Fond bleu avec texte blanc
          padx=30, pady=10,  # Espacement intérieur horizontal et vertical
          relief="flat",  # Style plat sans bordure 3D
          cursor="hand2").pack()  # Afficher un curseur main au survol et placer le bouton

root.mainloop()  # Démarrer la boucle d'événements Tkinter et maintenir la fenêtre ouverte
