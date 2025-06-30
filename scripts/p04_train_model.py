import os
import subprocess
import logging
import shutil
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root and key paths
# Assuming this script is in anpr_local_pipeline/scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
YOLOV5_DIR = os.path.join(PROJECT_ROOT, 'yolov5')
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, 'data.yaml') # data.yaml is in project root

# YOLOv5 training parameters (can be overridden by CLI args or config file)
DEFAULT_YOLO_MODEL_CFG = 'yolov5s.yaml' # Predefined model configuration
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 2 # Set to a smaller number for quick testing, e.g., 3-5
DEFAULT_RUN_NAME = 'license_plate_model'
DEFAULT_IMG_SIZE = 640 # Default image size for training

def clone_yolov5_repo():
    """Checks if the YOLOv5 repository is present."""
    if not os.path.exists(YOLOV5_DIR):
        logging.error(f"YOLOv5 directory not found at {YOLOV5_DIR}. Please clone YOLOv5 manually before running this script.")
        # Do not attempt to clone, just log error
    else:
        logging.info(f"YOLOv5 repository already exists at {YOLOV5_DIR}.")

def install_yolov5_requirements():
    """Installs requirements for YOLOv5."""
    requirements_path = os.path.join(YOLOV5_DIR, 'requirements.txt')
    if not os.path.exists(requirements_path):
        logging.error(f"YOLOv5 requirements.txt not found at {requirements_path}. Ensure YOLOv5 is cloned correctly.")
        return False

    logging.info(f"Installing YOLOv5 requirements from {requirements_path}...")
    try:
        # Use the same Python interpreter that's running this script
        process = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_path],
                                 check=True, capture_output=True, text=True, 
                                 cwd=YOLOV5_DIR, encoding='utf-8', errors='replace') # Run from YOLOV5_DIR with proper encoding
        logging.info("YOLOv5 requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install YOLOv5 requirements. Error: {e.stderr}")
        return False
    except FileNotFoundError: # If pip is not found
        logging.error("`pip` command not found. Ensure it's in your PATH.")
        return False


def train_yolo_model(data_yaml=DATA_YAML_PATH,
                     cfg=DEFAULT_YOLO_MODEL_CFG,
                     batch_size=DEFAULT_BATCH_SIZE,
                     epochs=DEFAULT_EPOCHS,
                     name=DEFAULT_RUN_NAME,
                     img_size=DEFAULT_IMG_SIZE,
                     device='cpu'): # Default to CPU for broader compatibility in local test
    """
    Trains the YOLOv5 model.

    Args:
        data_yaml (str): Path to the data.yaml file.
        cfg (str): Path to the model configuration file (e.g., yolov5s.yaml). This path should be relative to yolov5/models/ or absolute.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        name (str): Name of the training run (results saved in yolov5/runs/train/[name]).
        img_size (int): Input image size for training.
        device (str): Device to train on ('cpu', '0' for GPU 0, '0,1' for multiple GPUs).

    Returns:
        str: Path to the best trained model weights (best.pt) if training is successful, else None.
    """
    if not os.path.exists(YOLOV5_DIR):
        logging.error("YOLOv5 directory not found. Please clone or set up YOLOv5 first.")
        return None

    train_script_path = os.path.join(YOLOV5_DIR, 'train.py')
    if not os.path.exists(train_script_path):
        logging.error(f"YOLOv5 train.py not found at {train_script_path}.")
        return None

    # The cfg path for YOLOv5 can be just the filename if it's a standard model in yolov5/models
    # Or it can be a full path. If just filename, YOLOv5 prepends its own models path.
    model_cfg_path = cfg
    if not os.path.isabs(cfg) and not cfg.startswith('yolov5/models'):
        # Check if it's a standard model name like 'yolov5s.yaml'
        potential_cfg_path = os.path.join(YOLOV5_DIR, 'models', cfg)
        if os.path.exists(potential_cfg_path):
            model_cfg_path = potential_cfg_path # Use full path if it exists there
        # Else, assume YOLOv5 will find it or it's a custom path provided by user

    logging.info("Starting YOLOv5 model training...")
    logging.info(f"  Data YAML: {data_yaml}")
    logging.info(f"  Model CFG: {model_cfg_path}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Run Name: {name}")
    logging.info(f"  Image Size: {img_size}")
    logging.info(f"  Device: {device}")

    # Construct the training command
    # Note: YOLOv5's train.py should be run from within the yolov5 directory usually,
    # or paths in data.yaml need to be absolute or relative to where train.py is run.
    # Here, data_yaml path is absolute.
    cmd = [
        sys.executable, '-W', 'ignore', train_script_path,  # Use the same Python interpreter and ignore warnings
        '--data', data_yaml,
        '--cfg', model_cfg_path, # This should be path relative to yolov5 dir or absolute
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--name', name,
        '--img', str(img_size),
        '--device', device,
        '--weights', 'yolov5s.pt' # Start from pre-trained yolov5s weights for faster convergence
    ]

    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        # Running train.py from the PROJECT_ROOT, so paths in data.yaml (relative to project_root) are fine.
        # YOLOv5 train.py itself handles its internal relative paths from its own location.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, 
                                 cwd=PROJECT_ROOT, encoding='utf-8', errors='replace')
        
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
        
        process.stdout.close()
        return_code = process.wait()

        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

        logging.info("YOLOv5 training completed successfully.")

        # Find the path to the best model weights
        # Default save location: yolov5/runs/train/[name]/weights/best.pt
        best_weights_path = os.path.join(YOLOV5_DIR, 'runs', 'train', name, 'weights', 'best.pt')
        if os.path.exists(best_weights_path):
            logging.info(f"Best model weights saved at: {best_weights_path}")
            return best_weights_path
        else:
            logging.warning(f"Could not find best.pt at expected location: {best_weights_path}. Training might have had issues or saved elsewhere.")
            # Check for last.pt as a fallback
            last_weights_path = os.path.join(YOLOV5_DIR, 'runs', 'train', name, 'weights', 'last.pt')
            if os.path.exists(last_weights_path):
                 logging.warning(f"Found last.pt instead: {last_weights_path}")
                 return last_weights_path
            return None

    except subprocess.CalledProcessError as e:
        logging.error(f"YOLOv5 training failed. Return code: {e.returncode}")
        logging.error("See console output above for details.")
        return None
    except FileNotFoundError: # If python is not found
        logging.error("`python` command not found. Ensure it's in your PATH and points to the correct interpreter for YOLOv5.")
        return None

def main_train(epochs_override=None):
    """Main function to run the training preparation and process."""
    logging.info("Starting model training script (04_train_model.py)")

    # Remove the call to clone_yolov5_repo() and just check if YOLOV5_DIR exists
    if not os.path.exists(YOLOV5_DIR):
        logging.error(f"YOLOv5 directory not found at {YOLOV5_DIR}. Please clone YOLOv5 manually before running this script.")
        return

    # Before installing requirements, check if a yolov5 specific venv might be better
    # For now, install into current environment.
    if not install_yolov5_requirements():
        logging.error("Halting due to failure in installing YOLOv5 requirements.")
        return

    # Use override for epochs if provided (e.g., for testing)
    current_epochs = epochs_override if epochs_override is not None else DEFAULT_EPOCHS

    # Check if data.yaml exists
    if not os.path.exists(DATA_YAML_PATH):
        logging.error(f"data.yaml not found at {DATA_YAML_PATH}. This file is crucial for training.")
        logging.error("Please ensure '03_generate_labels.py' was run successfully and 'data.yaml' is in the project root.")
        return

    # Check if data_images/train and data_images/test directories exist and are not empty
    # These paths are relative to PROJECT_ROOT as per data.yaml structure
    train_images_path = os.path.join(PROJECT_ROOT, 'data_images', 'train')
    val_images_path = os.path.join(PROJECT_ROOT, 'data_images', 'test')

    if not (os.path.exists(train_images_path) and os.listdir(train_images_path)):
        logging.error(f"Training data directory {train_images_path} is missing or empty.")
        logging.error("Please run '03_generate_labels.py' to populate it.")
        return
    if not (os.path.exists(val_images_path) and os.listdir(val_images_path)):
        logging.error(f"Validation data directory {val_images_path} is missing or empty.")
        logging.error("Please run '03_generate_labels.py' to populate it.")
        return

    logging.info(f"Using {current_epochs} epochs for training.")

    # Check for existing training run to prevent conflicts or resume if supported (YOLOv5 handles resume with --resume)
    # For now, let's clear previous run of the same name to ensure a fresh start for this script's purpose.
    # More advanced handling could allow resuming.
    run_dir_to_clear = os.path.join(YOLOV5_DIR, 'runs', 'train', DEFAULT_RUN_NAME)
    if os.path.exists(run_dir_to_clear):
        logging.warning(f"Found existing training run at {run_dir_to_clear}. Removing it for a fresh start.")
        try:
            shutil.rmtree(run_dir_to_clear)
            logging.info(f"Successfully removed {run_dir_to_clear}.")
        except OSError as e:
            logging.error(f"Error removing {run_dir_to_clear}: {e.strerror}. Please remove manually if issues persist.")
            # Decide if to halt or proceed. For now, proceed, YOLO might overwrite or error.

    best_model_path = train_yolo_model(epochs=current_epochs, device='cpu') # Default to CPU for this assignment

    if best_model_path:
        logging.info(f"Training complete. Best model saved to: {best_model_path}")
    else:
        logging.error("Training failed or model path not found.")

if __name__ == '__main__':
    # For testing, run with a small number of epochs
    # In a real pipeline, these would be configured parameters.
    logging.info("Running 04_train_model.py directly for testing with 1 epoch.")

    # Create dummy data_images/train and data_images/test for the script to pass checks
    # This would normally be done by 03_generate_labels.py
    dummy_train_img_dir = os.path.join(PROJECT_ROOT, 'data_images', 'train')
    dummy_test_img_dir = os.path.join(PROJECT_ROOT, 'data_images', 'test')
    os.makedirs(dummy_train_img_dir, exist_ok=True)
    os.makedirs(dummy_test_img_dir, exist_ok=True)

    # Create at least one dummy file in each, so os.listdir is not empty
    # These are not actual images or labels, just to pass the pre-run check.
    # YOLOv5 will error if these are not valid images/labels, but the script structure will be tested.
    # For a full test, 01, 02, 03 must run first.
    if not os.listdir(dummy_train_img_dir):
        with open(os.path.join(dummy_train_img_dir, 'dummy.jpg'), 'w') as f: f.write('')
        with open(os.path.join(dummy_train_img_dir, 'dummy.txt'), 'w') as f: f.write('0 0.5 0.5 0.1 0.1') # dummy label
    if not os.listdir(dummy_test_img_dir):
        with open(os.path.join(dummy_test_img_dir, 'dummy.jpg'), 'w') as f: f.write('')
        with open(os.path.join(dummy_test_img_dir, 'dummy.txt'), 'w') as f: f.write('0 0.5 0.5 0.1 0.1') # dummy label

    if not os.path.exists(DATA_YAML_PATH):
        logging.error(f"{DATA_YAML_PATH} not found. Please create it or run previous scripts.")
        logging.info("Creating a minimal data.yaml for this test run...")
        with open(DATA_YAML_PATH, 'w') as f:
            f.write(f"train: {os.path.join(PROJECT_ROOT, 'data_images', 'train')}\n")
            f.write(f"val: {os.path.join(PROJECT_ROOT, 'data_images', 'test')}\n")
            f.write("nc: 1\n")
            f.write("names: ['license_plate']\n")
        logging.info(f"Minimal {DATA_YAML_PATH} created.")

    main_train(epochs_override=1) # Run with 1 epoch for a quick test of the script flow

    # Clean up dummy files created ONLY for this direct test scenario
    # In a pipeline, 03_generate_labels.py would create real data.
    # shutil.rmtree(dummy_train_img_dir) # Be careful if real data exists
    # shutil.rmtree(dummy_test_img_dir)
    # os.remove(DATA_YAML_PATH) # if created by this test block
    logging.info("Test run of 04_train_model.py finished. Check logs for details.")
    logging.warning("Note: This test run uses dummy image/label files. For actual training, ensure previous pipeline steps are completed.")
