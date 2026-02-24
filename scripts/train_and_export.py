import torch
import torchvision.models as models
import torch.nn as nn

# 1. Charger un modèle léger pré-entraîné
print("Chargement de MobileNetV2...")
# Note : on utilise weights=MobileNet_V2_Weights.DEFAULT dans les versions récentes de torchvision
model = models.mobilenet_v2(pretrained=True)

# 2. Modifier la dernière couche pour notre cas d'usage (2 classes : Humain / Pas humain)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# (Optionnel) Ici, vous devriez ajouter votre code d'entraînement avec vos images
# optimizer = torch.optim.Adam(...)
# loss_fn = nn.CrossEntropyLoss()
# ... boucle d'entraînement ...

# 3. Passer le modèle en mode évaluation
model.eval()

# 4. Créer un tenseur "factice" représentant la taille de l'image en entrée (Batch_size=1, Canaux=3, H=224, W=224)
dummy_input = torch.randn(1, 3, 224, 224)

# 5. Exporter le modèle au format ONNX
onnx_path = "./models/human_recognition.onnx"
print(f"Exportation du modèle vers {onnx_path}...")
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=12, 
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output']
)
print("Export terminé !")
