
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# 1. Initialiser RKNN Lite
rknn = RKNNLite()

# 2. Charger le modèle
print("Chargement du modèle RKNN...")
rknn.load_rknn('./models/human_recognition.rknn')

# 3. Initialiser le runtime (Assignation sur le NPU)
print("Initialisation du NPU...")
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
if ret != 0:
    print("Échec de l'initialisation du runtime !")
    exit(ret)

# 4. Préparer l'image
img_path = 'data/personne-redimensionne-taille-224x224.jpeg' # Remplacez par le chemin de votre image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # PyTorch / RKNN s'attendent souvent à du RGB
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0) # Ajouter la dimension du batch : shape devient (1, 224, 224, 3)

# 5. Lancer l'inférence
print("Lancement de l'inférence...")
outputs = rknn.inference(inputs=[img])

# 6. Analyser le résultat
# outputs[0] contient un tableau numpy avec les probabilités brutes (logits) des 2 classes
logits = outputs[0][0]
predicted_class = np.argmax(logits)

classes = ["Pas Humain", "Humain"]
print(f"Résultat brut : {logits}")
print(f"Prédiction : {classes[predicted_class]}")

# Libérer le NPU
rknn.release()


