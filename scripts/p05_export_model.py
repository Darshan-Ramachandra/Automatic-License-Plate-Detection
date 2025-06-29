import os
import subprocess
import logging
import shutil
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root and key paths
# Assuming this script is in anpr_local_pipeline/scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
YOLOV5_DIR = os.path.join(PROJECT_ROOT, 'yolov5') # Assumed location of yolov5 clone
DEFAULT_TRAINED_MODEL_NAME = 'license_plate_model' # Name used during training

# Path to the weights file. The model is actually saved in yolov5/runs/train/
WEIGHTS_PATH = os.path.join(YOLOV5_DIR, 'runs', 'train', DEFAULT_TRAINED_MODEL_NAME, 'weights', 'best.pt')


def export_yolo_model(weights_path, img_size=640, device='cpu', include_formats=['torchscript', 'onnx']):
    """
    Exports a trained YOLOv5 model to various formats.

    Args:
        weights_path (str): Path to the trained model's .pt file.
        img_size (int): Image size.
        device (str): Device ('cpu', '0', etc.).
        include_formats (list): Formats to export to.

    Returns:
        dict: Paths to exported models.
              None if export fails.
    """
    logging.info("Starting YOLOv5 model export...")

    yolov5_export_script_path = os.path.join(YOLOV5_DIR, 'export.py')

    if not os.path.exists(YOLOV5_DIR) or not os.path.exists(yolov5_export_script_path):
        logging.error(f"YOLOv5 directory ({YOLOV5_DIR}) or export.py script not found.")
        return None

    if not os.path.exists(weights_path):
        logging.error(f"Weights file not found at {weights_path}.")
        return None

    logging.info(f"Exporting model from: {weights_path}")
    logging.info(f"  Image Size: {img_size}, Device: {device}, Formats: {', '.join(include_formats)}")

    cmd = [
        sys.executable, yolov5_export_script_path,
        '--weights', weights_path,
        '--imgsz', str(img_size),
        '--device', device,
        '--include', *include_formats
    ]
    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, 
                                cwd=PROJECT_ROOT, encoding='utf-8', errors='replace')
        logging.info("YOLOv5 model export completed successfully.")
        logging.debug(f"Export output:\n{process.stdout}")

        # Get the actual exported model paths
        exported_model_paths = {}
        weights_dir = os.path.dirname(weights_path)
        weights_basename = os.path.splitext(os.path.basename(weights_path))[0]

        for fmt in include_formats:
            if fmt == 'onnx':
                exported_path = os.path.join(weights_dir, f"{weights_basename}.onnx")
                if os.path.exists(exported_path):
                    exported_model_paths[fmt] = exported_path
                    logging.info(f"ONNX model exported to: {exported_path}")
                else:
                    logging.warning(f"ONNX model not found at expected location: {exported_path}")
            elif fmt == 'torchscript':
                exported_path = os.path.join(weights_dir, f"{weights_basename}.torchscript.pt")
                if os.path.exists(exported_path):
                    exported_model_paths[fmt] = exported_path
                    logging.info(f"TorchScript model exported to: {exported_path}")
                else:
                    logging.warning(f"TorchScript model not found at expected location: {exported_path}")

        return exported_model_paths

    except subprocess.CalledProcessError as e:
        logging.error(f"YOLOv5 model export failed. Return code: {e.returncode}")
        logging.error(f"Stdout:\n{e.stdout}")
        logging.error(f"Stderr:\n{e.stderr}")
        return None
    except FileNotFoundError:
        logging.error("`python` or `export.py` command not found.")
        return None

def main_export(run_training_first_if_needed=False):
    """
    Main function to run the model export process.
    """
    logging.info("Starting model export script (05_export_model.py)")

    # Check if the weights file exists
    if not os.path.exists(WEIGHTS_PATH):
        logging.error(f"Weights file not found at {WEIGHTS_PATH}.")
        logging.error("Please ensure the training step completed successfully.")
        return None

    logging.info(f"Found trained model at: {WEIGHTS_PATH}")

    exported_paths = export_yolo_model(weights_path=WEIGHTS_PATH,
                                       include_formats=['torchscript', 'onnx'])

    if exported_paths:
        logging.info("Export completed successfully. Exported models:")
        for fmt, path in exported_paths.items():
            logging.info(f"  {fmt}: {path}")
        return exported_paths
    else:
        logging.error("Export failed or no models were exported.")
        return None

if __name__ == '__main__':
    logging.info("Running 05_export_model.py directly for testing.")
    main_export()
    logging.info("Test run of 05_export_model.py finished.")
