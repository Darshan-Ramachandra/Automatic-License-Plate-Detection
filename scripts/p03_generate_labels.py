import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_yolo_labels(df_yolo, output_image_dir_base, output_label_dir_base, source_image_dir, test_size=0.2, random_state=42):
    """
    Splits data into train/test sets, copies images, and saves YOLO label files.

    Args:
        df_yolo (pd.DataFrame): DataFrame with YOLO formatted annotations
                                (must include 'filename', 'class_id', 'center_x_norm',
                                'center_y_norm', 'width_norm', 'height_norm').
        output_image_dir_base (str): Base directory to save train/test image folders (e.g., 'data_images').
        output_label_dir_base (str): Base directory to save train/test label folders (e.g., 'data_labels').
                                     Often this is the same as output_image_dir_base for YOLO structure.
        source_image_dir (str): Directory where original images are stored.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    if df_yolo.empty:
        logging.error("Input DataFrame is empty. Cannot generate labels.")
        return

    # Ensure 'filename' is present
    if 'filename' not in df_yolo.columns:
        logging.error("'filename' column is missing from the DataFrame.")
        return

    # Get unique filenames for train/test split to keep all annotations for an image in the same set
    unique_files = df_yolo['filename'].unique()
    if len(unique_files) == 0:
        logging.warning("No unique filenames found in the DataFrame.")
        return

    if len(unique_files) < 2 : # Cannot split if only one unique image
        logging.warning(f"Only {len(unique_files)} unique image(s). Splitting may not be effective or possible if less than 2 images for train and test each.")
        # Decide how to handle: either put all in train or raise error
        # For now, let's put all in train if only 1 image, or proceed if more but still small
        if len(unique_files) == 1:
            train_files = unique_files
            test_files = []
        else: # Need at least one for test if test_size > 0
             # sklearn train_test_split needs at least 1 sample for each set if stratify is not None
             # and at least 2 samples in total.
             if test_size > 0 and test_size < 1:
                try:
                    train_files, test_files = train_test_split(unique_files, test_size=test_size, random_state=random_state)
                except ValueError as e:
                    logging.warning(f"Could not perform train/test split with {len(unique_files)} unique files and test_size={test_size}. Error: {e}. Assigning all to train.")
                    train_files = unique_files
                    test_files = []
             elif test_size == 0:
                 train_files = unique_files
                 test_files = []
             elif test_size == 1:
                 train_files = []
                 test_files = unique_files
             else: # Should not happen with typical test_size values
                 logging.error(f"Invalid test_size: {test_size}")
                 return

    else: # Standard case with enough unique files
        train_files, test_files = train_test_split(unique_files, test_size=test_size, random_state=random_state)


    logging.info(f"Number of unique images: {len(unique_files)}")
    logging.info(f"Training images: {len(train_files)}, Testing images: {len(test_files)}")

    sets = {'train': train_files, 'test': test_files}

    for set_name, files_in_set in sets.items():
        if not isinstance(files_in_set, list): # Convert numpy array to list if necessary, or use .size
             files_in_set_list = list(files_in_set) # Convert to list for easier checking / iteration
        else:
             files_in_set_list = files_in_set

        if not files_in_set_list: # Check if the list is empty
            logging.info(f"No files allocated to {set_name} set. Skipping.")
            continue

        # Path for images and labels (YOLO expects labels in same dir as images or parallel dir)
        # For simplicity here, we'll create data_images/train, data_images/test
        # and data_labels/train, data_labels/test (can be same as image path)
        current_image_output_dir = os.path.join(output_image_dir_base, set_name)
        current_label_output_dir = os.path.join(output_label_dir_base, set_name) # YOLO expects labels with images

        os.makedirs(current_image_output_dir, exist_ok=True)
        os.makedirs(current_label_output_dir, exist_ok=True) # Ensure label dir exists

        logging.info(f"Processing {set_name} set...")
        for image_filename in files_in_set_list: # Iterate over the list
            # Copy image
            source_image_path = os.path.join(source_image_dir, image_filename)
            destination_image_path = os.path.join(current_image_output_dir, image_filename)

            if os.path.exists(source_image_path):
                try:
                    shutil.copy(source_image_path, destination_image_path)
                except Exception as e:
                    logging.error(f"Error copying {source_image_path} to {destination_image_path}: {e}")
                    continue # Skip this image if copying fails
            else:
                logging.warning(f"Source image not found: {source_image_path}. Skipping this image.")
                continue

            # Get annotations for this image
            annotations = df_yolo[df_yolo['filename'] == image_filename]

            # Create label file path (e.g., image1.jpg -> image1.txt)
            label_filename_base = os.path.splitext(image_filename)[0]
            label_file_path = os.path.join(current_label_output_dir, f"{label_filename_base}.txt")

            with open(label_file_path, 'w') as f_label:
                for _, row in annotations.iterrows():
                    # YOLO format: class_id center_x_norm center_y_norm width_norm height_norm
                    yolo_line = f"{int(row['class_id'])} {row['center_x_norm']:.6f} {row['center_y_norm']:.6f} {row['width_norm']:.6f} {row['height_norm']:.6f}\n"
                    f_label.write(yolo_line)
            logging.debug(f"Saved label file: {label_file_path}")
        logging.info(f"Finished processing {set_name} set. Images in {current_image_output_dir}, Labels in {current_label_output_dir}")


if __name__ == '__main__':
    logging.info("Testing 03_generate_labels.py")

    # Create dummy data and directories for testing
    # This assumes 02_preprocess_dataset.py output format
    data_yolo = {
        'filename': ['img1.jpg', 'img1.jpg', 'img2.png', 'img3.jpeg', 'img4.jpg', 'img5.bmp'],
        'class_id': [0, 0, 0, 0, 0, 0],
        'center_x_norm': [0.25, 0.5, 0.46875, 0.3, 0.6, 0.7],
        'center_y_norm': [0.375, 0.6, 0.46875, 0.4, 0.5, 0.8],
        'width_norm': [0.25, 0.1, 0.3125, 0.2, 0.15, 0.1],
        'height_norm': [0.083333, 0.15, 0.3125, 0.1, 0.2, 0.05],
        'width': [800, 800, 640, 1000, 1200, 500], # Original image width/height
        'height': [600, 600, 480, 800, 900, 400]
    }
    test_yolo_df = pd.DataFrame(data_yolo)

    # Project root relative to this script anpr_local_pipeline/scripts/03_generate_labels.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Dummy source images directory
    test_source_image_dir = os.path.join(project_root, 'data', 'test_source_images')
    os.makedirs(test_source_image_dir, exist_ok=True)

    # Create dummy image files
    for img_name in test_yolo_df['filename'].unique():
        with open(os.path.join(test_source_image_dir, img_name), 'w') as f_img:
            f_img.write("dummy image content") # Create empty files for testing copy

    # Output directories for the test
    test_output_img_base = os.path.join(project_root, 'test_data_images') # For YOLO: data_images/train, data_images/test
    # YOLO expects labels next to images, so often label base is same as image base
    test_output_label_base = test_output_img_base

    # Clean up previous test run directories
    if os.path.exists(test_output_img_base):
        shutil.rmtree(test_output_img_base)
    # if os.path.exists(test_output_label_base) and test_output_label_base != test_output_img_base:
    #     shutil.rmtree(test_output_label_base) # Only if they are different

    logging.info(f"Dummy source images in: {test_source_image_dir}")
    logging.info(f"Output images will be in: {test_output_img_base}")
    logging.info(f"Output labels will be in: {test_output_label_base}")

    generate_yolo_labels(test_yolo_df,
                         output_image_dir_base=test_output_img_base,
                         output_label_dir_base=test_output_label_base, # For YOLO, labels are with images
                         source_image_dir=test_source_image_dir,
                         test_size=0.4, # 5 images -> 3 train, 2 test
                         random_state=42)

    # Verification
    logging.info("Verification of generated files:")
    expected_train_files = 0
    expected_test_files = 0

    # Based on random_state=42 and 5 unique files ['img1.jpg', 'img2.png', 'img3.jpeg', 'img4.jpg', 'img5.bmp']
    # train_test_split(..., test_size=0.4, random_state=42) for 5 items gives 3 train, 2 test.
    # We need to know which files went where if we want to be precise.
    # unique_files = ['img1.jpg', 'img2.png', 'img3.jpeg', 'img4.jpg', 'img5.bmp']
    # With random_state=42, indices for test might be [idx1, idx2]
    # For now, let's just count.

    train_img_dir = os.path.join(test_output_img_base, 'train')
    test_img_dir = os.path.join(test_output_img_base, 'test')
    train_label_dir = os.path.join(test_output_label_base, 'train') # Should be same as train_img_dir for YOLO
    test_label_dir = os.path.join(test_output_label_base, 'test')   # Should be same as test_img_dir for YOLO

    num_train_imgs = len([f for f in os.listdir(train_img_dir) if not f.endswith('.txt')]) if os.path.exists(train_img_dir) else 0
    num_test_imgs = len([f for f in os.listdir(test_img_dir) if not f.endswith('.txt')]) if os.path.exists(test_img_dir) else 0
    num_train_labels = len([f for f in os.listdir(train_label_dir) if f.endswith('.txt')]) if os.path.exists(train_label_dir) else 0
    num_test_labels = len([f for f in os.listdir(test_label_dir) if f.endswith('.txt')]) if os.path.exists(test_label_dir) else 0

    logging.info(f"Found {num_train_imgs} images in train image directory.")
    logging.info(f"Found {num_train_labels} labels in train label directory.")
    logging.info(f"Found {num_test_imgs} images in test image directory.")
    logging.info(f"Found {num_test_labels} labels in test label directory.")

    if num_train_imgs == num_train_labels and num_test_imgs == num_test_labels:
        logging.info("Number of images matches number of labels in both train and test sets.")
    else:
        logging.error("Mismatch between number of images and labels.")

    # Check content of one label file (e.g., first training image's label)
    # Need to know which file ended up in train set.
    # For this test, let's assume 'img1.jpg' (2 annotations) or 'img2.png' (1 annotation) is in train.
    # If 'img1.jpg' is in train:
    # train_files after split: ['img5.bmp', 'img2.png', 'img1.jpg']
    # test_files after split:  ['img4.jpg', 'img3.jpeg']

    example_train_img_name = 'img1.jpg' # Has 2 annotations
    example_train_label_file = os.path.join(train_label_dir, os.path.splitext(example_train_img_name)[0] + '.txt')
    if os.path.exists(example_train_label_file):
        with open(example_train_label_file, 'r') as f:
            lines = f.readlines()
            logging.info(f"Content of {example_train_label_file}:")
            for line in lines:
                logging.info(line.strip())
            if len(lines) == 2: # img1.jpg has two annotations in the dummy data
                logging.info(f"Correct number of annotations found for {example_train_img_name}.")
            else:
                logging.error(f"Incorrect number of annotations for {example_train_img_name}. Expected 2, Got {len(lines)}")
    else:
        logging.warning(f"Could not find example label file {example_train_label_file} to verify content. This might be due to split variation or file not being in train set.")


    # Clean up test directories and files
    logging.info("Cleaning up test files and directories...")
    if os.path.exists(test_source_image_dir):
        shutil.rmtree(test_source_image_dir)
    if os.path.exists(test_output_img_base):
        shutil.rmtree(test_output_img_base)
    # if os.path.exists(test_output_label_base) and test_output_label_base != test_output_img_base:
    #     shutil.rmtree(test_output_label_base) # Only if they are different

    logging.info("\nTest completed for 03_generate_labels.py.")
