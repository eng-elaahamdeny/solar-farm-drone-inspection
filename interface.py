import tensorflow as tf
import numpy as np # Manipuler les tableaux de nombres (pixels)
from PIL import Image, ImageTk # Ouvrir et afficher des images
import tkinter as tk # Créer la fenêtre et l'interface graphique
from tkinter import filedialog # Ouvrir l'explorateur de fichiers

# ══════════════════════════════
# CHARGER L'IA
# ══════════════════════════════

# On charge le modèle de deep learning déjà entraîné (MobileNetV2) depuis le fichier.
model = tf.keras.models.load_model("model/solar_defect_model.keras")

# classes : dictionnaire qui associe à chaque numéro de classe un nom de défaut,
CLASSES = {
    0: ("Bird Drop",    "Moyen",    "#f39c12"),
    1: ("Snow Covered", "Faible",   "#2980b9"),
    2: ("Crack",        "Sérieux",  "#e67e22"),
    3: ("Dirty",        "Moyen",    "#d4ac0d"),
    4: ("Hotspot",      "Critique", "#e74c3c"),
    5: ("Normal",       "Aucun",    "#27ae60"),
}


# ══════════════════════════════
# FENÊTRE 
# ══════════════════════════════

root = tk.Tk() #tk.Tk() crée la fenêtre principale de l'application.
root.title("Solar Panel Inspector — AI Detection")
root.configure(bg="white")
root.configure(bg="#fafafa")

# Taille de l'écran
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Fenêtre à 80% de l'écran (même ratio)
width = int(screen_width * 0.8)
height = int(screen_height * 0.9)

# Centrer la fenêtre
x = (screen_width - width) // 2
y = (screen_height - height) // 2

root.geometry(f"{width}x{height}+{x}+{y}")

# ══════════════════════════════
# HEADER (en-tête)
# ══════════════════════════════
header = tk.Frame(root, bg="#02385A", pady=15)
header.pack(fill="x")# s'étirer sur toute la largeur de la fenêtre.

tk.Label(header,# Crée un texte dans l'en-tête
         text="☀  Solar Panel Inspector",
         font=("Arial", 20, "bold"),
         bg="#02385A", fg="white").pack()

tk.Label(header,
         text="Inspection intelligente par Drone & Intelligence Artificielle",
         font=("Arial", 11),
         bg="#02385A", fg="white").pack(pady=(3, 0))

# ══════════════════════════════
# BOUTON
# ══════════════════════════════
btn_frame = tk.Frame(root, bg="white", pady=5)
btn_frame.pack(fill="x") 

# ══════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════
main = tk.Frame(root, bg="white")  # Crée un cadre principal blanc
main.pack(fill="x", expand=False, padx=40, pady=(10, 0), anchor="n")  # Place le cadre avec marges et alignement haut

spacer = tk.Frame(root, bg="white")  # Crée un cadre vide pour pousser le contenu vers le haut
spacer.pack(fill="both", expand=True)  # Place le cadre vide pour occuper l'espace restant

# COLONNE GAUCHE — image
left = tk.Frame(main, bg="white")
left.pack(side="left", fill="both", expand=True)

tk.Label(left,
         text="Image thermique",
         font=("Arial", 13, "bold"),
         bg="white", fg="#1a1a2e").pack(anchor="w", pady=(0, 8))
#Le Canvas est une zone de dessin où tu peux afficher des images, dessiner des formes, écrire du texte, et créer des éléments graphiques
canvas_image = tk.Canvas(left,
                          width=500, height=400,
                          bg="#f0f0f0",
                          highlightthickness=1,
                          highlightbackground="#dddddd")
canvas_image.pack(anchor="w", pady=(0, 20))

# Calculer le centre du canvas
centre_x = 500 // 2  # 250 pixels
centre_y = 400 // 2  # 200 pixels

# Créer le texte centré
canvas_image.create_text(centre_x, centre_y,
                          text="[ Aucune image chargée ]",
                          font=("Arial", 13),
                          fill="#999999")

# COLONNE DROITE — résultats
right = tk.Frame(main, bg="white", padx=50)
right.pack(side="right", fill="both", expand=True)

tk.Label(right,
         text="Résultat de l'analyse",
         font=("Arial", 14, "bold"),
         bg="white", fg="#1a1a2e").pack(anchor="w", pady=(0, 15))

label_resultat = tk.Label(right,
                           text="Aucune analyse",
                           font=("Arial", 22, "bold"),
                           bg="white", fg="#cccccc")
label_resultat.pack(anchor="w")

label_danger = tk.Label(right,
                         text="",
                         font=("Arial", 13),
                         bg="white", fg="#999999")
label_danger.pack(anchor="w", pady=(8, 20))

tk.Label(right,
         text="Confiance du modèle :",
         font=("Arial", 12, "bold"),
         bg="white", fg="#1a1a2e").pack(anchor="w")

canvas_bar = tk.Canvas(right, width=500, height=30,
                        bg="white", highlightthickness=0)
canvas_bar.pack(anchor="w", pady=(8, 10))

tk.Frame(right, bg="#eeeeee", height=1).pack(fill="x", pady=8)

tk.Label(right,
         text="Scores par classe :",
         font=("Arial", 12, "bold"),
         bg="white", fg="#1a1a2e").pack(anchor="w", pady=(5, 5))


labels_scores = {}
for i, (nom, danger, couleur) in CLASSES.items():
    lbl = tk.Label(right,
                   text=f"{nom}: —",
                   font=("Arial", 11),
                   bg="white", fg="#cccccc")
    lbl.pack(anchor="w", pady=2)
    labels_scores[i] = lbl


# ══════════════════════════════
# FOOTER
# ══════════════════════════════
footer = tk.Frame(root, bg="#f5f5f5", pady=8)
footer.pack(fill="x", side="bottom")

tk.Label(footer,
         text="Projet PFA 25/26 — Inspection photovoltaïque par Drone & IA ",
         font=("Arial", 8),
         bg="#f5f5f5", fg="#555555").pack()


# ══════════════════════════════
# FONCTION ANALYSER — après tout
# ══════════════════════════════
def analyser_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return

    img_display = Image.open(path).resize((620, 560), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_display)
    canvas_image.delete("all")
    canvas_image.create_image(0, 0, anchor="nw", image=img_tk)
    canvas_image.image = img_tk

    img = Image.open(path).resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(arr)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100
    nom, danger, couleur = CLASSES[class_idx]

    canvas_bar.delete("all")
    canvas_bar.create_rectangle(0, 0, 500, 30, fill="#eeeeee", outline="")
    canvas_bar.create_rectangle(0, 0, int(500 * confidence / 100), 30,
                                 fill=couleur, outline="")
    canvas_bar.create_text(250, 15,
                            text=f"{confidence:.1f}%",
                            font=("Arial", 11, "bold"),
                            fill="white")

    label_resultat.config(text=f"Défaut : {nom}", fg=couleur)
    label_danger.config(text=f"Niveau de danger : {danger}", fg=couleur)

    for i, (cls_nom, _, cls_couleur) in CLASSES.items():
        score = prediction[0][i] * 100
        labels_scores[i].config(
            text=f"{cls_nom}: {score:.1f}%",
            fg=cls_couleur
        )

# Bouton créé après la fonction
tk.Button(btn_frame,
          text="📂   Charger une image et analyser",
          command=analyser_image,
          font=("Arial", 13, "bold"),
          bg="#378ADD", fg="white",
          padx=30, pady=10,
          relief="flat",
          cursor="hand2").pack()

root.mainloop()
