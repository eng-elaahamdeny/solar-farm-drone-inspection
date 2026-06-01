import tensorflow as tf  # Importer TensorFlow pour charger et exécuter le modèle IA
import numpy as np  # Importer NumPy pour manipuler les tableaux et les pixels
from PIL import Image, ImageTk, ImageDraw  # Importer PIL pour ouvrir les images et les convertir pour Tkinter
import tkinter as tk  # Importer Tkinter pour construire l'interface graphique
from tkinter import filedialog  # Importer filedialog pour ouvrir l'explorateur de fichiers
import cv2  # Importer OpenCV pour la détection de contours
import random  # Pour générer des coordonnées GPS simulées
from tensorflow.keras.models import Model

# ══════════════════════════════
# CHARGER L'IA
# ══════════════════════════════

model = tf.keras.models.load_model("model/solar_defect_model.keras")  # Charger le modèle de deep learning pré-entraîné depuis le disque

# CLASSES associe chaque index de classe à un nom de défaut, un niveau de danger et une couleur d'affichage
CLASSES = {
    0: ("Clean",             "Aucun",    "#27ae60"),
    1: ("Dusty",             "Faible",   "#2980b9"),
    2: ("Bird Drop",         "Moyen",    "#f39c12"),
    3: ("Electrical Damage", "Critique", "#e74c3c"),
    4: ("Physical Damage",   "Sérieux",  "#e67e22"),
    5: ("Snow Covered",      "Moyen",    "#d4ac0d"),
}
# ══════════════════════════════
# FONCTION DE DÉTECTION DE ZONE DÉFECTUEUSE (AMÉLIORÉE)
# ══════════════════════════════

def get_gradcam_bboxes(image_path, couleur):
    img_orig = cv2.imread(image_path)
    h, w = img_orig.shape[:2]
    hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    # Détecter selon la couleur du défaut
    if couleur == "#e74c3c":  # Electrical Damage → zones orange/brun/rouge
        mask1 = cv2.inRange(hsv, (0, 40, 80), (25, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 40, 80), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif couleur == "#f39c12":  # Bird Drop → zones blanches
        mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    elif couleur == "#2980b9":  # Dusty → zones grises claires
        mask = cv2.inRange(hsv, (0, 0, 120), (180, 40, 200))
    elif couleur == "#e67e22":  # Physical Damage → zones brillantes (fissures/éclats)
    # Fissures = zones très brillantes sur fond bleu
       _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # Zones sombres aussi (bords cassés)
       _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
       mask = cv2.bitwise_or(bright, dark)
    # Exclure les zones trop uniformément blanches (ciel, reflets)
       hsv_check = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
       not_sky = cv2.inRange(hsv_check, (90, 20, 20), (130, 255, 200))
       mask = cv2.bitwise_and(mask, not_sky)
    elif couleur == "#d4ac0d":  # Snow → zones très blanches
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    else:
        # Fallback générique
        mask = cv2.inRange(hsv, (0, 40, 80), (35, 255, 255))

    # Nettoyage
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((8, 😎, np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    min_area = (w * h) * 0.01
    max_area = (w * h) * 0.6

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        bw = min(w - x, bw + 2 * margin)
        bh = min(h - y, bh + 2 * margin)
        bboxes.append((x, y, bw, bh))

    # Trier par surface décroissante
    bboxes.sort(key=lambda b: -(b[2] * b[3]))
    bboxes = bboxes[:3]

    if not bboxes:
        # Fallback center
        bboxes = [(w//4, h//4, w//2, h//2)]

    return bboxes

def dessiner_rectangles(image, bboxes, couleur, nom_defaut):
    draw = ImageDraw.Draw(image)
    
    for i, (x, y, w, h) in enumerate(bboxes):
        # Rectangle pointillé simulé avec des segments
        dash = 8
        gap = 5
        # Haut
        cx = x
        while cx < x + w:
            draw.line([(cx, y), (min(cx + dash, x + w), y)], fill=couleur, width=2)
            cx += dash + gap
        # Bas
        cx = x
        while cx < x + w:
            draw.line([(cx, y + h), (min(cx + dash, x + w), y + h)], fill=couleur, width=2)
            cx += dash + gap
        # Gauche
        cy = y
        while cy < y + h:
            draw.line([(x, cy), (x, min(cy + dash, y + h))], fill=couleur, width=2)
            cy += dash + gap
        # Droite
        cy = y
        while cy < y + h:
            draw.line([(x + w, cy), (x + w, min(cy + dash, y + h))], fill=couleur, width=2)
            cy += dash + gap

        # Point rouge au centre exact du défaut
        centre_x = x + w // 2
        centre_y = y + h // 2
        rayon = 6
        draw.ellipse([centre_x - rayon, centre_y - rayon,
                      centre_x + rayon, centre_y + rayon],
                     fill=couleur, outline="white")

        # Étiquette
        label = nom_defaut if len(bboxes) == 1 else f"{nom_defaut} {i+1}"
        draw.rectangle([x + 2, y - 18, x + len(label) * 7 + 6, y - 2],
                       fill=couleur)
        draw.text((x + 4, y - 17), label, fill="white")

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
                   font=("Arial", 😎, bg="white", fg="#cccccc")
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
# FONCTIONS DE CORRECTION DES CONFUSIONS
# ══════════════════════════════

def detecter_ensoleillement(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = gray.size

    # 1. Zones sombres = défaut → pas ensoleillement
    dark = np.sum(gray < 50) / total
    if dark > 0.05:
        return False

    # 2. Zones orange/rouge = electrical damage → pas ensoleillement
    orange = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
    red = cv2.inRange(hsv, (0, 80, 80), (5, 255, 255))
    hot_ratio = (np.sum(orange > 0) + np.sum(red > 0)) / total
    if hot_ratio > 0.01:
        return False

    # 3. Couleur dominante bleue = panneau normal → ensoleillement possible
    blue_mask = cv2.inRange(hsv, (90, 20, 30), (140, 255, 200))
    blue_ratio = np.sum(blue_mask > 0) / total
    if blue_ratio < 0.30:
        return False

    # 4. Zone brillante compacte et grande
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright_mask > 0) / total
    if bright_ratio < 0.04:
        return False

    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(bright_mask, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    compact_bright = False
    for c in contours:
        area = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        ratio_forme = bw / (bh + 1e-5)
        if area > total * 0.03 and 0.3 < ratio_forme < 3.0:
            compact_bright = True
            break

    if not compact_bright:
        return False

    # 5. Texture uniforme = ensoleillement (pas de défaut localisé)
    std_gray = np.std(gray)

    return True

def detecter_brouillard(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    mean = np.mean(gray)
    return std < 20 and mean > 150

def detecter_ombre_structure(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_ratio = np.sum(gray < 40) / gray.size
    bright_ratio = np.sum(gray > 200) / gray.size
    return dark_ratio > 0.15 and bright_ratio > 0.15

def detecter_nuage_reflet(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright > 0) / gray.size
    std = np.std(gray)
    return bright_ratio > 0.3 and std < 40

# ══════════════════════════════
# FONCTION ANALYSER
# ══════════════════════════════
def get_bbox_ensoleillement(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img.shape[:2]
    
    _, bright = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20), np.uint8)
    bright_dilated = cv2.dilate(bright, kernel)
    contours, _ = cv2.findContours(bright_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        best = max(contours, key=lambda c: np.mean(
            gray[cv2.boundingRect(c)[1]:cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3],
                 cv2.boundingRect(c)[0]:cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2]]))
        x, y, w, h = cv2.boundingRect(best)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(w_img - x, w + 2 * margin)
        h = min(h_img - y, h + 2 * margin)
        return (x, y, w, h)
    else:
        return (w_img//4, h_img//4, w_img//2, h_img//2)
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
    # Vérifier ensoleillement avant tout
    physical_score = prediction[0][4] * 100
    electrical_score = prediction[0][3] * 100
    bird_score = prediction[0][2] * 100
    if detecter_ensoleillement(path):
       nom = "Ensoleillement"
       danger = "Aucun"
       couleur = "#f1c40f"
    # ══════════════════════════════
    # CORRECTIONS DES CONFUSIONS
    # ══════════════════════════════
    elif detecter_brouillard(path):
        nom = "Image floue / Brouillard"
        danger = "Non analysable"
        couleur = "#95a5a6"
    elif nom == "Snow Covered" and detecter_nuage_reflet(path):
        nom = "Reflet nuage"
        danger = "Aucun"
        couleur = "#7f8c8d"
    elif nom == "Snow Covered":
        img_check = cv2.imread(path)
        gray_check = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
        if np.mean(gray_check) < 120 and np.std(gray_check) > 30:
            nom = "Dusty"
            danger = "Faible"
            couleur = "#2980b9"
    elif nom == "Electrical Damage" and detecter_ombre_structure(path) and confidence < 80:
        nom = "Ombre de structure"
        danger = "Aucun"
        couleur = "#7f8c8d"
    elif nom == "Bird Drop":
        img_check = cv2.imread(path)
        gray_check = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_check, 200, 255, cv2.THRESH_BINARY)
        if np.sum(thresh > 0) / gray_check.size < 0.01 and confidence < 70:
            nom = "Tache (non défaut)"
            danger = "Aucun"
            couleur = "#7f8c8d"
    
    # Détection de la zone du défaut (uniquement si ce n'est pas normal)
    if nom != "Normal":
        # Détecter la zone dans l'image originale
        if nom == "Ensoleillement":
           bboxes_original = [get_bbox_ensoleillement(path)]
        else:
           bboxes_original = get_gradcam_bboxes(path, couleur)
        
        if bboxes_original:
            # Adapter les coordonnées à la taille d'affichage
            bboxes_display = []
            for (x, y, w, h) in bboxes_original:
              bboxes_display.append((
                 int(x * scale_x),
                 int(y * scale_y),
                 int(w * scale_x),
                 int(h * scale_y)
                ))
            # Dessiner le rectangle sur l'image
            img_with_rect = dessiner_rectangles(img_display, bboxes_display, couleur, nom)
            
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
