import sys
from pathlib import Path
from rknn.api import RKNN

# Chemins relatifs au projet
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = str(ROOT / 'dataset' / 'quantization_dataset.txt')  # dataset pour quantification
ONNX_DIR = ROOT / 'onnx'
RKNN_DIR = ROOT / 'rknn'
RKNN_DIR.mkdir(parents=True, exist_ok=True)

# Configuration par défaut pour Rock 5B
DEFAULT_PLATFORM = 'rk3588'
DEFAULT_QUANT = False  # i8 quantification (recommandé pour NPU)

def parse_arg():
    """
    Usage:
      python scripts/convert_to_rknn.py model.onnx [dtype] [output_name.rknn]
      
    Exemples:
      python scripts/convert_to_rknn.py newname.onnx
      python scripts/convert_to_rknn.py newname.onnx fp
      python scripts/convert_to_rknn.py newname.onnx i8 custom_output.rknn
    """
    if len(sys.argv) < 2:
        print("Usage: python3 {} onnx_model_name [dtype(optional)] [output_rknn_name(optional)]".format(sys.argv[0]))
        print("       dtype: i8 (quantifié, recommandé) ou fp (float32, plus lent)")
        print("       Le fichier ONNX doit être dans: {}".format(ONNX_DIR))
        print("\nExemple:")
        print("  python scripts/convert_to_rknn.py newname.onnx")
        print("  python scripts/convert_to_rknn.py newname.onnx i8 mon_modele.rknn")
        exit(1)

    # Résoudre le chemin ONNX
    onnx_name = sys.argv[1]
    model_path = ONNX_DIR / onnx_name
    if not model_path.exists():
        print(f"ERREUR: Fichier ONNX introuvable: {model_path}")
        exit(1)

    # Type de quantification
    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
        if model_type not in ['i8', 'fp']:
            print("ERREUR: dtype invalide. Choisir 'i8' (quantifié) ou 'fp' (float32)")
            exit(1)
        do_quant = (model_type == 'i8')

    # Nom de sortie RKNN
    if len(sys.argv) > 3:
        output_name = sys.argv[3]
    else:
        # Par défaut: remplacer .onnx par .rknn
        output_name = onnx_name.replace('.onnx', '.rknn')
    
    output_path = RKNN_DIR / output_name

    return str(model_path), DEFAULT_PLATFORM, do_quant, str(output_path)



def convert_to_rknn( model_path, platform, do_quant, output_path ):



    print(f"\n{'='*60}")
    print(f"Conversion ONNX → RKNN pour Rock 5B (RK3588)")
    print(f"{'='*60}")
    print(f"ONNX source    : {model_path}")
    print(f"RKNN sortie    : {output_path}")
    print(f"Quantification : {'i8 (NPU optimisé)' if do_quant else 'fp32 (lent)'}")
    print(f"Dataset calib  : {DATASET_PATH}")
    print(f"{'='*60}\n")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config (normalisation YOLOv8 standard)
    print('--> Configuration du modèle')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform=platform
    )
    print('✓ Configuration terminée\n')

    # Load ONNX model
    print('--> Chargement du modèle ONNX')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('✗ Échec du chargement!')
        exit(ret)
    print('✓ Modèle chargé\n')

    # Build model
    print('--> Construction du modèle RKNN')
    if do_quant:
        if not Path(DATASET_PATH).exists():
            print(f"AVERTISSEMENT: Dataset de quantification introuvable: {DATASET_PATH}")
            print("Créez un fichier texte avec ~20-100 chemins d'images de votre dataset.")
    
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH if do_quant else None)
    if ret != 0:
        print('✗ Échec de la construction!')
        exit(ret)
    print('✓ Modèle construit\n')

    # Export rknn model
    print('--> Export du modèle RKNN')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('✗ Échec de l\'export!')
        exit(ret)
    print(f'✓ Modèle exporté: {output_path}\n')

    # Release
    rknn.release()
    
    print(f"{'='*60}")
    print("✓ Conversion terminée avec succès!")
    print(f"Transférez {output_path} sur votre Rock 5B")
    print(f"{'='*60}\n")


if __name__ == '__main__':

    model_path, platform, do_quant, output_path = parse_arg()
    convert_to_rknn(model_path,platform,do_quant,output_path)