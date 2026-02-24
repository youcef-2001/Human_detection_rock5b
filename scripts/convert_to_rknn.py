
from rknn.api import RKNN

# Initialiser l'objet RKNN
rknn = RKNN(verbose=True)

# 1. Configurer le modèle pour le RK3588
# mean_values et std_values dépendent du prétraitement utilisé lors de l'entraînement PyTorch
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]], 
    std_values=[[58.395, 57.12, 57.375]], 
    target_platform='rk3588'
)

# 2. Charger le modèle ONNX
print("Chargement du modèle ONNX...")
ret = rknn.load_onnx(model='./models/human_recognition.onnx')
if ret != 0:
    print("Échec du chargement ONNX !")
    exit(ret)

# 3. Construire le modèle RKNN
# Note: do_quantization=False pour simplifier. En production, passez-le à True (INT8) 
# pour plus de performances, mais il faudra fournir un dataset de calibration.
print("Construction du modèle RKNN...")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print("Échec de la construction !")
    exit(ret)

# 4. Exporter le fichier .rknn
rknn_path = "./models/human_recognition.rknn"
print(f"Exportation vers {rknn_path}...")
ret = rknn.export_rknn(rknn_path)

# Libérer la mémoire
rknn.release()
print("Conversion RKNN réussie !")


