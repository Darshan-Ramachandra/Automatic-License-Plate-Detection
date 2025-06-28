import pandas as pd
from sklearn.model_selection import train_test_split
import os

def convert_to_yolo_format(df, class_mapping={'number_plate': 0}):
    """
    Converts bounding box coordinates to YOLO format (normalized).
    Assumes class 'license_plate' is 0. Other classes will be ignored.

    Args:
        df (pd.DataFrame): DataFrame with columns [filename, width, height, name, xmin, ymin, xmax, ymax].
        class_mapping (dict): A dictionary mapping class names to integer IDs.

    Returns:
        pd.DataFrame: DataFrame with YOLO formatted annotations and additional columns:
                      [class_id, center_x_norm, center_y_norm, width_norm, height_norm].
                      Rows with classes not in class_mapping are dropped.
    """
    if df.empty:
        return pd.DataFrame(columns=['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax',
                                     'class_id', 'center_x_norm', 'center_y_norm', 'width_norm', 'height_norm'])

    # Filter for relevant classes
    df_filtered = df[df['name'].isin(class_mapping.keys())].copy()
    if df_filtered.empty:
        print("Warning: No relevant classes found in the dataframe after filtering.")
        return pd.DataFrame(columns=['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax',
                                     'class_id', 'center_x_norm', 'center_y_norm', 'width_norm', 'height_norm'])

    df_filtered.loc[:, 'class_id'] = df_filtered['name'].map(class_mapping)

    # Calculate center_x, center_y, width, height
    df_filtered.loc[:, 'bb_center_x'] = (df_filtered['xmin'] + df_filtered['xmax']) / 2
    df_filtered.loc[:, 'bb_center_y'] = (df_filtered['ymin'] + df_filtered['ymax']) / 2
    df_filtered.loc[:, 'bb_width'] = df_filtered['xmax'] - df_filtered['xmin']
    df_filtered.loc[:, 'bb_height'] = df_filtered['ymax'] - df_filtered['ymin']

    # Normalize
    df_filtered.loc[:, 'center_x_norm'] = df_filtered['bb_center_x'] / df_filtered['width']
    df_filtered.loc[:, 'center_y_norm'] = df_filtered['bb_center_y'] / df_filtered['height']
    df_filtered.loc[:, 'width_norm'] = df_filtered['bb_width'] / df_filtered['width']
    df_filtered.loc[:, 'height_norm'] = df_filtered['bb_height'] / df_filtered['height']

    # Select and rename columns for YOLO format
    yolo_df = df_filtered[['filename', 'class_id', 'center_x_norm', 'center_y_norm', 'width_norm', 'height_norm',
                           'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']].copy() # Keep original for reference if needed

    return yolo_df

if __name__ == '__main__':
    print("Testing 02_preprocess_dataset.py")

    # Create a dummy DataFrame similar to what 01_parse_annotations.py would produce
    data = {
        'filename': ['img1.jpg', 'img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.xml'],
        'width': [800, 800, 640, 1024, 800],
        'height': [600, 600, 480, 768, 600],
        'name': ['license_plate', 'car', 'license_plate', 'truck', 'license_plate'], # 'car' and 'truck' should be filtered out
        'xmin': [100, 50, 200, 300, 0],
        'ymin': [200, 100, 150, 250, 0],
        'xmax': [300, 250, 400, 500, 0], # For img4, xmax=xmin, ymax=ymin to test zero width/height
        'ymax': [250, 200, 300, 400, 0]
    }
    test_df = pd.DataFrame(data)
    print("\nOriginal DataFrame:")
    print(test_df)

    yolo_annotations_df = convert_to_yolo_format(test_df)

    print("\nYOLO Formatted DataFrame (only license_plate):")
    if not yolo_annotations_df.empty:
        print(yolo_annotations_df[['filename', 'class_id', 'center_x_norm', 'center_y_norm', 'width_norm', 'height_norm']])

        # Basic checks
        if 'class_id' in yolo_annotations_df.columns and yolo_annotations_df['class_id'].unique() == [0]:
            print("\nClass ID is correctly mapped to 0 for 'license_plate'.")
        else:
            print("\nError in class ID mapping.")

        if len(yolo_annotations_df) == 3: # img1.jpg, img2.jpg, img4.xml (car and truck filtered)
            print("Correct number of rows after filtering for 'license_plate'.")
        else:
            print(f"Incorrect number of rows. Expected 3, Got {len(yolo_annotations_df)}")

        # Check normalization for the first valid entry
        # For img1.jpg, license_plate:
        # xmin=100, ymin=200, xmax=300, ymax=250, width=800, height=600
        # center_x = (100+300)/2 = 200 -> norm = 200/800 = 0.25
        # center_y = (200+250)/2 = 225 -> norm = 225/600 = 0.375
        # bb_width = 300-100 = 200 -> norm = 200/800 = 0.25
        # bb_height = 250-200 = 50 -> norm = 50/600 = 0.08333...
        first_entry = yolo_annotations_df.iloc[0]
        expected_cx_norm = (100 + 300) / 2 / 800
        expected_cy_norm = (200 + 250) / 2 / 600
        expected_w_norm = (300 - 100) / 800
        expected_h_norm = (250 - 200) / 600

        if (abs(first_entry['center_x_norm'] - expected_cx_norm) < 1e-6 and
            abs(first_entry['center_y_norm'] - expected_cy_norm) < 1e-6 and
            abs(first_entry['width_norm'] - expected_w_norm) < 1e-6 and
            abs(first_entry['height_norm'] - expected_h_norm) < 1e-6):
            print("Normalization calculations are correct for the first entry.")
        else:
            print("Error in normalization calculations for the first entry.")
            print(f"Expected: cx={expected_cx_norm}, cy={expected_cy_norm}, w={expected_w_norm}, h={expected_h_norm}")
            print(f"Got:      cx={first_entry['center_x_norm']}, cy={first_entry['center_y_norm']}, w={first_entry['width_norm']}, h={first_entry['height_norm']}")

        # Check zero width/height case (img4)
        # xmin=0, ymin=0, xmax=0, ymax=0, width=800, height=600
        # center_x = 0 -> norm = 0
        # center_y = 0 -> norm = 0
        # bb_width = 0 -> norm = 0
        # bb_height = 0 -> norm = 0
        zero_entry = yolo_annotations_df[yolo_annotations_df['filename'] == 'img4.xml'].iloc[0]
        if (abs(zero_entry['center_x_norm'] - 0) < 1e-6 and
            abs(zero_entry['center_y_norm'] - 0) < 1e-6 and
            abs(zero_entry['width_norm'] - 0) < 1e-6 and
            abs(zero_entry['height_norm'] - 0) < 1e-6):
            print("Normalization for zero width/height bounding box is correct.")
        else:
            print("Error in normalization for zero width/height bounding box.")


    else:
        print("YOLO conversion returned an empty DataFrame. Check for errors.")

    print("\nTest completed.")
