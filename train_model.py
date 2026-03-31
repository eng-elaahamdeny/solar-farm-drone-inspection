#═══════════════════════════════
# ÉTAPE 1 : Importer les outils
# ══════════════════════════════

import tensorflow as tf # TensorFlow : le moteur principal de l'IA
from tensorflow.keras.applications import MobileNetV2 # Importe un cerveau IA (MobileNet2) déjà pré-entraîné
from tensorflow.keras import layers, models #  les "briques" pour construire le modèle
from tensorflow.keras.preprocessing.image import ImageDataGenerator # outil qui lit et prépare les images automatiquement
import matplotlib.pyplot as plt # Pour dessiner les graphiques à la fin
import os #Pour créer des dossiers sur l'ordinateur

print(" Outils importés !")

# ══════════════════════════════════════
# ÉTAPE 2 : Les réglages (Configuration)
# ══════════════════════════════════════

DATASET_PATH = "dataset/"  
IMG_SIZE     = (224, 224)  # images redimensionnées en 224×224 pixel
BATCH_SIZE   = 32          #L'IA analyse 32 images à la fois
EPOCHS       = 10          #L'IA va relire tout le dataset 10 fois 

print(" Configuration prête !")

# ══════════════════════════════
# ÉTAPE 3 : Charger les images
# ══════════════════════════════

#train_gen : prépare les photos pour l'entraînement
train_gen = ImageDataGenerator(
    rescale=1./255, #  transforme les pixels (valeurs entre 0 et 255) en nombres entre 0 et 1 (plus facile pour l'IA)      
    validation_split=0.2,  # 20% des images sont mises de côté pour tester, 80% pour apprendre
    rotation_range=15,     # Tourne les images jusqu'à 15° , au hasard,(pour varier les exemples)
    horizontal_flip=True   # Crée des versions "miroir" des images,(pour varier les exemples)
)
# générateur de validation (rescale seulement)
val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# train_data et val_data : vont chercher les photos dans le dossier
# flow_from_directory : lit automatiquement l'organisation des dossiers (chaque dossier = une catégorie de défaut)

train_data = train_gen.flow_from_directory( 
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",#Dit à l'IA qu'il y a plusieurs classes (Bird Drop, Snow, Crack, etc.)
    subset="training" # prend les 80% pour apprendre
)

val_data = val_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical", 
    subset="validation" # prend les 20% pour apprendre
)

print(" Images chargées !")
print("Classes trouvées :", list(train_data.class_indices.keys()))#Dictionnaire qui associe les noms des dossiers à des numéros ex: {'Bird Drop': 0, 'Snow Covered': 1, ...}

# ══════════════════════════════
# ÉTAPE 4 : Créer le modèle IA
# ══════════════════════════════

# on prend base_model = MobileNetV2, un modèle déjà intelligent
base_model = MobileNetV2(
    input_shape=(224, 224, 3),# photos 224×224 pixels en couleurs (3 = rouge, vert, bleu)
    include_top=False, # on enlève sa partie "classification"
    weights='imagenet' #on lui donne ses connaissances déjà apprises sur 1 million d'images      
)
base_model.trainable = False # on bloque ses connaissances
# model : on ajoute nos propres étages 
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),# on réduit l'information
    layers.Dense(128, activation='relu'),#une couche avec 128 neurones qui réfléchit sur les informations
    layers.Dropout(0.3),# on éteint 30% des neurones au hasard (ça évite que l'IA apprenne par cœur)
    layers.Dense(train_data.num_classes, activation='softmax') # la dernière couche, elle donne la réponse finale
])

model.compile(
    optimizer='adam',# la méthode qu'elle utilise pour s'améliorer
    loss='categorical_crossentropy',# la façon de mesurer ses erreurs
    metrics=['accuracy']#  sa précision (combien de bonnes réponses)
)

print(" Modèle IA créé !")

# ════════════════════════════
# ÉTAPE 5 : Entraîner l'IA
# ════════════════════════════

print("\ Entraînement en cours... (10-20 minutes)\n")
#le moment où l'IA apprend vraiment
#history : on garde un journal de ses performances (précision, erreur)
history = model.fit(
    train_data,# les photos pour apprendre (80%)
    validation_data=val_data,# les photos pour vérifier (20%)
    epochs=EPOCHS # 10 passages complets sur toutes les photos
)

print("\n Entraînement terminé !")

# ══════════════════════════════
# ÉTAPE 6 : Sauvegarder l'IA
# ══════════════════════════════

os.makedirs("model", exist_ok=True) # crée un dossier "model" s'il n'existe pas
model.save("model/solar_defect_model.keras") # sauvegarde l'IA entraînée sur le disque dur
print(" IA sauvegardée dans model/solar_defect_model.keras")

# ════════════════════════════════════════
# ÉTAPE 7 : Graphiques pour le rapport
# ════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],
         label='Train', color='steelblue')
ax1.plot(history.history['val_accuracy'],
         label='Validation', color='orange')
ax1.set_title('Précision du modèle')
ax1.set_xlabel('Epoch')
# An epoch is one complete pass through the entire training dataset.
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],
         label='Train', color='steelblue')
ax2.plot(history.history['val_loss'],
         label='Validation', color='orange')
ax2.set_title('Perte du modèle')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("model/resultats.png")
plt.show()
print(" Graphiques sauvegardés ")


""""
════════════════════════
GRAPHIQUE DE PRECISION :
════════════════════════
Courbe bleue : comment l'IA progresse sur les photos d'entraînement
Courbe orange : comment elle progresse sur les photos de test
Si les deux montent ensemble → c'est bon !


═════════════════════
GRAPHIQUE DE PERTE :
═════════════════════
Courbe bleue : ses erreurs sur les photos d'entraînement
Courbe orange : ses erreurs sur les photos de test
Si les deux descendent ensemble → c'est bon !
"""
