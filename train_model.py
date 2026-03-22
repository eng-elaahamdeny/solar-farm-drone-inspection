# ═══════════════════════════════════════
# ÉTAPE 1 : Importer les outils
# ═══════════════════════════════════════
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print("✅ Outils importés !")

# ═══════════════════════════════════════
# ÉTAPE 2 : Configuration
# ═══════════════════════════════════════
DATASET_PATH = "dataset/"  # dossier de tes images
IMG_SIZE     = (224, 224)  # taille des images
BATCH_SIZE   = 32          # images analysées en même temps
EPOCHS       = 10          # nombre de fois que l'IA relit tout

print("✅ Configuration prête !")

# ═══════════════════════════════════════
# ÉTAPE 3 : Charger les images
# ═══════════════════════════════════════
train_gen = ImageDataGenerator(
    rescale=1./255,        # normalise les pixels
    validation_split=0.2,  # 80% train, 20% test
    rotation_range=15,     # tourne les images un peu
    horizontal_flip=True   # miroir des images
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = val_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("✅ Images chargées !")
print("Classes trouvées :", list(train_data.class_indices.keys()))

# ═══════════════════════════════════════
# ÉTAPE 4 : Créer le modèle IA
# ═══════════════════════════════════════
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'      # déjà intelligent !
)
base_model.trainable = False  # on garde son intelligence

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèle IA créé !")

# ═══════════════════════════════════════
# ÉTAPE 5 : Entraîner l'IA
# ═══════════════════════════════════════
print("\n🚀 Entraînement en cours... (10-20 minutes)\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

print("\n✅ Entraînement terminé !")

# ═══════════════════════════════════════
# ÉTAPE 6 : Sauvegarder l'IA
# ═══════════════════════════════════════
os.makedirs("model", exist_ok=True)
model.save("model/solar_defect_model.keras")
print("✅ IA sauvegardée dans model/solar_defect_model.keras")

# ═══════════════════════════════════════
# ÉTAPE 7 : Graphiques pour ton rapport
# ═══════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],
         label='Train', color='steelblue')
ax1.plot(history.history['val_accuracy'],
         label='Validation', color='orange')
ax1.set_title('Précision du modèle')
ax1.set_xlabel('Epoch')
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
print("📊 Graphiques sauvegardés !")