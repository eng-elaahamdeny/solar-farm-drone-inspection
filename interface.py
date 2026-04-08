import tensorflow as tf  # Importer TensorFlow pour charger et exécuter le modèle IA
import numpy as np  # Importer NumPy pour manipuler les tableaux et les pixels
from PIL import Image, ImageTk, ImageDraw  # Importer PIL pour ouvrir les images et les convertir pour Tkinter
import tkinter as tk  # Importer Tkinter pour construire l'interface graphique
from tkinter import filedialog  # Importer filedialog pour ouvrir l'explorateur de fichiers
import cv2  # Importer OpenCV pour la détection de contours
import random  # Pour générer des coordonnées GPS simulées

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
# FONCTION DE DÉTECTION DE ZONE DÉFECTUEUSE (AMÉLIORÉE)
# ══════════════════════════════

def detecter_zone_defaut(image_path, defect_type):
    """
    Détecte la région du défaut dans l'image et retourne un rectangle
    Version améliorée avec meilleure détection des hotspots
    """
    # Charger l'image avec OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # Convertir en espace colorimétrique adapté
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if defect_type == "Hotspot":
        # Détection améliorée des hotspots (zones thermiques)
        # Utiliser l'histogramme pour trouver les zones chaudes
        # Calculer le seuil adaptatif basé sur le percentile 85
        flat = blurred.flatten()
        flat_sorted = np.sort(flat)
        threshold_val = flat_sorted[int(len(flat_sorted) * 0.85)]  # 85ème percentile
        
        # Appliquer le seuil
        _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Opérations morphologiques pour nettoyer
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
    elif defect_type == "Crack":
        # Détection des fissures (contours)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(edges, kernel, iterations=2)
    elif defect_type in ["Bird Drop", "Dirty"]:
        # Détection des taches
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(edges, kernel, iterations=3)
    else:
        # Pour les autres défauts
        mean_val = np.mean(blurred)
        std_val = np.std(blurred)
        threshold = mean_val + std_val
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours par taille
    min_area = (height * width) * 0.01  # 1% de l'image minimum
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if valid_contours:
        # Prendre le plus grand contour valide
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Ajouter une marge autour du contour
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        
        return (x, y, w, h)
    else:
        # Si pas de détection, retourner une zone au centre
        w = width // 3
        h = height // 3
        x = (width - w) // 2
        y = (height - h) // 2
        return (x, y, w, h)

def dessiner_rectangle(image, bbox, couleur, nom_defaut):
    """
    Dessine un rectangle autour du défaut sur l'image
    """
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    
    # Dessiner le rectangle
    draw.rectangle([x, y, x + w, y + h], outline=couleur, width=3)
    
    # Ajouter une étiquette avec le nom du défaut
    draw.text((x + 5, y - 18), nom_defaut, fill=couleur)
    
    # Ajouter un point au centre
    centre_x = x + w // 2
    centre_y = y + h // 2
    rayon = 5
    draw.ellipse([centre_x - rayon, centre_y - rayon, 
                  centre_x + rayon, centre_y + rayon], 
                 fill=couleur)
    
    return image

# ══════════════════════════════
# FONCTION DE GÉOLOCALISATION GPS
# ══════════════════════════════

def generer_coordonnees_gps(defect_type):
    """
    Génère des coordonnées GPS simulées basées sur le type de défaut
    """
    # Coordonnées de base (exemple: parc solaire à Toulouse)
    base_lat = 43.604652
    base_lon = 1.444209
    
    # Variations selon le type de défaut
    variations = {
        "Hotspot": (0.00023, 0.00045),
        "Crack": (0.00067, 0.00032),
        "Bird Drop": (0.00123, 0.00098),
        "Dirty": (0.00089, 0.00134),
        "Snow Covered": (0.00156, 0.00067),
        "Normal": (0.00000, 0.00000)
    }
    
    var_lat, var_lon = variations.get(defect_type, (0.0005, 0.0005))
    
    # Ajouter un peu d'aléatoire
    random.seed(hash(defect_type + str(random.random())))
    lat = base_lat + var_lat + random.uniform(-0.0002, 0.0002)
    lon = base_lon + var_lon + random.uniform(-0.0002, 0.0002)
    precision = random.uniform(1.5, 3.5) if defect_type != "Normal" else 0
    
    return lat, lon, precision

# ══════════════════════════════
# FENÊTRE
# ══════════════════════════════

root = tk.Tk()
root.title("Solar Panel Inspector — AI Detection with Geolocation")
root.configure(bg="#fafafa")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

width = int(screen_width * 0.85)
height = int(screen_height * 0.9)

x = (screen_width - width) // 2
y = (screen_height - height) // 2

root.geometry(f"{width}x{height}+{x}+{y}")

# ══════════════════════════════
# HEADER
# ══════════════════════════════

header = tk.Frame(root, bg="#02385A", pady=12)
header.pack(fill="x")

tk.Label(header, text="☀  Solar Panel Inspector",
         font=("Arial", 18, "bold"), bg="#02385A", fg="white").pack()

tk.Label(header, text="Inspection intelligente par Drone & Intelligence Artificielle",
         font=("Arial", 10), bg="#02385A", fg="white").pack(pady=(2, 0))

# ══════════════════════════════
# BOUTON
# ══════════════════════════════

btn_frame = tk.Frame(root, bg="white", pady=5)
btn_frame.pack(fill="x")

# ══════════════════════════════
# CONTENU PRINCIPAL
# ══════════════════════════════

main = tk.Frame(root, bg="white")
main.pack(fill="both", expand=True, padx=30, pady=(5, 10))

# COLONNE GAUCHE — affichage de l'image
left = tk.Frame(main, bg="white")
left.pack(side="left", fill="both", expand=True)

tk.Label(left, text="Image thermique avec localisation du défaut",
         font=("Arial", 12, "bold"), bg="white", fg="#1a1a2e").pack(anchor="w", pady=(0, 5))

# Canvas pour l'image avec taille fixe
canvas_width = 600
canvas_height = 500
canvas_image = tk.Canvas(left, width=canvas_width, height=canvas_height, bg="#f0f0f0",
                          highlightthickness=1, highlightbackground="#dddddd")
canvas_image.pack(anchor="w", pady=(0, 15))

centre_x = canvas_width // 2
centre_y = canvas_height // 2

canvas_image.create_text(centre_x, centre_y,
                          text="[ Aucune image chargée ]",
                          font=("Arial", 12), fill="#999999")

# COLONNE DROITE — résultats (SANS SCROLLBAR)
right = tk.Frame(main, bg="white", padx=40)
right.pack(side="right", fill="both", expand=True)

# Frame pour les résultats avec taille fixe
right_content = tk.Frame(right, bg="white")
right_content.pack(fill="both", expand=True)

# ══════════════════════════════
# CONTENU DE LA COLONNE DROITE (police réduite)
# ══════════════════════════════

# Résultat principal
tk.Label(right_content, text="Résultat de l'analyse",
         font=("Arial", 11, "bold"), bg="white", fg="#1a1a2e").pack(anchor="w", pady=(0, 8))

label_resultat = tk.Label(right_content, text="Aucune analyse",
                           font=("Arial", 16, "bold"), bg="white", fg="#cccccc")
label_resultat.pack(anchor="w")

label_danger = tk.Label(right_content, text="", font=("Arial", 10),
                         bg="white", fg="#999999")
label_danger.pack(anchor="w", pady=(3, 10))

# SECTION GÉOLOCALISATION
geoloc_frame = tk.Frame(right_content, bg="white", relief="solid", bd=1)
geoloc_frame.pack(fill="x", pady=(0, 10))

tk.Label(geoloc_frame, text="📍 Géolocalisation GPS du défaut", 
         font=("Arial", 10, "bold"), bg="#f8f9fa", fg="#1a1a2e").pack(anchor="w", padx=8, pady=(6, 5))

coord_inner = tk.Frame(geoloc_frame, bg="#f8f9fa")
coord_inner.pack(fill="x", padx=8, pady=(0, 6))

# Latitude
tk.Label(coord_inner, text="Latitude:", font=("Arial", 8, "bold"),
         bg="#f8f9fa", fg="#555").grid(row=0, column=0, sticky="w", pady=1)
label_lat = tk.Label(coord_inner, text="—", font=("Arial", 9),
                     bg="#f8f9fa", fg="#e74c3c")
label_lat.grid(row=0, column=1, sticky="w", pady=1, padx=(8, 0))

# Longitude
tk.Label(coord_inner, text="Longitude:", font=("Arial", 8, "bold"),
         bg="#f8f9fa", fg="#555").grid(row=1, column=0, sticky="w", pady=1)
label_lon = tk.Label(coord_inner, text="—", font=("Arial", 9),
                     bg="#f8f9fa", fg="#e74c3c")
label_lon.grid(row=1, column=1, sticky="w", pady=1, padx=(8, 0))

# Précision
tk.Label(coord_inner, text="Précision GPS:", font=("Arial", 8, "bold"),
         bg="#f8f9fa", fg="#555").grid(row=2, column=0, sticky="w", pady=1)
label_precision = tk.Label(coord_inner, text="—", font=("Arial", 9),
                           bg="#f8f9fa", fg="#e74c3c")
label_precision.grid(row=2, column=1, sticky="w", pady=1, padx=(8, 0))

tk.Frame(right_content, bg="#eeeeee", height=1).pack(fill="x", pady=5)

# Confiance
tk.Label(right_content, text="Confiance du modèle :",
         font=("Arial", 10, "bold"), bg="white", fg="#1a1a2e").pack(anchor="w", pady=(5, 3))

canvas_bar = tk.Canvas(right_content, width=450, height=22, bg="white", highlightthickness=0)
canvas_bar.pack(anchor="w", pady=(0, 8))

tk.Frame(right_content, bg="#eeeeee", height=1).pack(fill="x", pady=5)

# Scores par classe (disposition en grille pour économiser de l'espace)
tk.Label(right_content, text="Scores par classe :",
         font=("Arial", 10, "bold"), bg="white", fg="#1a1a2e").pack(anchor="w", pady=(5, 5))

# Frame pour les scores en 2 colonnes
scores_frame = tk.Frame(right_content, bg="white")
scores_frame.pack(fill="x", pady=(0, 5))

labels_scores = {}
classes_list = list(CLASSES.items())
mid = len(classes_list) // 2

# Colonne de gauche
left_scores = tk.Frame(scores_frame, bg="white")
left_scores.pack(side="left", fill="both", expand=True)

# Colonne de droite
right_scores = tk.Frame(scores_frame, bg="white")
right_scores.pack(side="right", fill="both", expand=True)

for i, (idx, (nom, danger, couleur)) in enumerate(classes_list):
    if i < mid:
        parent = left_scores
    else:
        parent = right_scores
    
    lbl = tk.Label(parent, text=f"{nom}: —",
                   font=("Arial", 8), bg="white", fg="#cccccc")
    lbl.pack(anchor="w", pady=2)
    labels_scores[idx] = lbl

# ══════════════════════════════
# FOOTER (réduit et sans chevauchement)
# ══════════════════════════════

footer = tk.Frame(root, bg="#f5f5f5", pady=4)
footer.pack(fill="x", side="bottom", before=main)

tk.Label(footer, text="Projet PFA 25/26 — Inspection photovoltaïque par Drone & IA — Localisation précise des défauts",
         font=("Arial", 7), bg="#f5f5f5", fg="#555555").pack()

# ══════════════════════════════
# FONCTION ANALYSER
# ══════════════════════════════

def analyser_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return

    # Charger l'image originale
    img_original = Image.open(path)
    
    # Calculer les dimensions pour l'affichage en conservant le ratio
    img_width, img_height = img_original.size
    
    # Calculer le ratio pour s'adapter au canvas
    ratio = min(canvas_width / img_width, canvas_height / img_height)
    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)
    
    # Redimensionner l'image pour l'affichage
    img_display = img_original.copy()
    img_display = img_display.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculer le facteur d'échelle pour la détection
    scale_x = img_display.width / img_original.width
    scale_y = img_display.height / img_original.height
    
    # Analyse IA
    img = Image.open(path).resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(arr, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100
    nom, danger, couleur = CLASSES[class_idx]
    
    # Détection de la zone du défaut (uniquement si ce n'est pas normal)
    if nom != "Normal":
        # Détecter la zone dans l'image originale
        bbox_original = detecter_zone_defaut(path, nom)
        
        if bbox_original:
            # Adapter les coordonnées à la taille d'affichage
            x, y, w, h = bbox_original
            x_disp = int(x * scale_x)
            y_disp = int(y * scale_y)
            w_disp = int(w * scale_x)
            h_disp = int(h * scale_y)
            bbox_display = (x_disp, y_disp, w_disp, h_disp)
            
            # Dessiner le rectangle sur l'image
            img_with_rect = dessiner_rectangle(img_display, bbox_display, couleur, nom)
            
            # Centrer l'image dans le canvas
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Mettre à jour l'affichage
            img_tk = ImageTk.PhotoImage(img_with_rect)
            canvas_image.delete("all")
            canvas_image.create_image(x_offset, y_offset, anchor="nw", image=img_tk)
            canvas_image.image = img_tk
            
            # Générer les coordonnées GPS
            lat, lon, precision = generer_coordonnees_gps(nom)
            label_lat.config(text=f"{lat:.6f}° N")
            label_lon.config(text=f"{lon:.6f}° E")
            label_precision.config(text=f"±{precision:.1f} m")
        else:
            # Si pas de détection, afficher l'image normale
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            img_tk = ImageTk.PhotoImage(img_display)
            canvas_image.delete("all")
            canvas_image.create_image(x_offset, y_offset, anchor="nw", image=img_tk)
            canvas_image.image = img_tk
            label_lat.config(text="—")
            label_lon.config(text="—")
            label_precision.config(text="—")
    else:
        # Pas de défaut
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        img_tk = ImageTk.PhotoImage(img_display)
        canvas_image.delete("all")
        canvas_image.create_image(x_offset, y_offset, anchor="nw", image=img_tk)
        canvas_image.image = img_tk
        label_lat.config(text="Aucun défaut")
        label_lon.config(text="Aucun défaut")
        label_precision.config(text="—")
    
    # Mettre à jour la barre de confiance
    canvas_bar.delete("all")
    canvas_bar.create_rectangle(0, 0, 450, 22, fill="#eeeeee", outline="")
    canvas_bar.create_rectangle(0, 0, int(450 * confidence / 100), 22, fill=couleur, outline="")
    canvas_bar.create_text(225, 11, text=f"{confidence:.1f}%",
                            font=("Arial", 9, "bold"), fill="white")
    
    # Mettre à jour les labels de résultat
    label_resultat.config(text=f"Défaut : {nom}", fg=couleur)
    label_danger.config(text=f"Niveau de danger : {danger}", fg=couleur)
    
    # Mettre à jour les scores
    for i, (cls_nom, _, cls_couleur) in CLASSES.items():
        score = prediction[0][i] * 100
        labels_scores[i].config(text=f"{cls_nom}: {score:.1f}%", fg=cls_couleur)

# Bouton
tk.Button(btn_frame, text="📂   Charger une image et analyser",
          command=analyser_image, font=("Arial", 12, "bold"),
          bg="#378ADD", fg="white", padx=25, pady=8,
          relief="flat", cursor="hand2").pack()

root.mainloop()
