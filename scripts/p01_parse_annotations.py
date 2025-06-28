import xml.etree.ElementTree as ET
import os
import pandas as pd

def parsing(annotation_path):
    """
    Parses XML annotation files to extract image metadata and bounding box information.

    Args:
        annotation_path (str): Path to the directory containing XML annotation files.

    Returns:
        pandas.DataFrame: A DataFrame containing parsed data with columns:
                          [filename, width, height, name, xmin, ymin, xmax, ymax].
                          Returns an empty DataFrame if no XML files are found or if errors occur.
    """
    all_data = []

    if not os.path.exists(annotation_path):
        print(f"Error: Annotation path '{annotation_path}' does not exist.")
        return pd.DataFrame()

    xml_files = [f for f in os.listdir(annotation_path) if f.endswith('.xml')]
    if not xml_files:
        print(f"No XML files found in '{annotation_path}'.")
        return pd.DataFrame()

    for file in xml_files:
        file_path = os.path.join(annotation_path, file)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            for obj in root.findall('object'):
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                all_data.append([filename, width, height, name, xmin, ymin, xmax, ymax])
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            continue

    if not all_data:
        print("No data extracted from XML files.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax'])
    return df

if __name__ == '__main__':
    # Create dummy annotation files for testing
    # Corrected path: relative to the script's directory, then up to project root, then to data
    project_root_relative_to_script = os.path.join(os.path.dirname(__file__), '..')
    test_annotation_dir_abs = os.path.abspath(os.path.join(project_root_relative_to_script, 'data', 'annotations_test'))
    os.makedirs(test_annotation_dir_abs, exist_ok=True)

    # Dummy XML 1
    xml_content1 = """
    <annotation>
        <folder>images</folder>
        <filename>test_image1.jpg</filename>
        <path>/path/to/test_image1.jpg</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>800</width>
            <height>600</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>license_plate</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>100</xmin>
                <ymin>200</ymin>
                <xmax>300</xmax>
                <ymax>250</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    with open(os.path.join(test_annotation_dir_abs, 'test_annotation1.xml'), 'w') as f:
        f.write(xml_content1)

    # Dummy XML 2
    xml_content2 = """
    <annotation>
        <folder>images</folder>
        <filename>test_image2.png</filename>
        <path>/path/to/test_image2.png</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>1024</width>
            <height>768</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>license_plate</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>50</xmin>
                <ymin>150</ymin>
                <xmax>250</xmax>
                <ymax>200</ymax>
            </bndbox>
        </object>
         <object>
            <name>car</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>10</xmin>
                <ymin>20</ymin>
                <xmax>800</xmax>
                <ymax>600</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    with open(os.path.join(test_annotation_dir_abs, 'test_annotation2.xml'), 'w') as f:
        f.write(xml_content2)

    # Test the parsing function
    print("Testing 01_parse_annotations.py")
    # The test_annotation_dir_abs is already the absolute path to the test annotations

    df_parsed = parsing(test_annotation_dir_abs)

    if not df_parsed.empty:
        print("Parsed DataFrame:")
        print(df_parsed.head())
        print(f"\nTotal records parsed: {len(df_parsed)}")
        # Basic checks
        expected_columns = ['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']
        if all(col in df_parsed.columns for col in expected_columns):
            print("\nAll expected columns are present.")
        else:
            print(f"\nMissing columns. Expected: {expected_columns}, Got: {list(df_parsed.columns)}")

        if len(df_parsed) == 3: # 1 object in first file, 2 in second
             print("Correct number of objects parsed.")
        else:
            print(f"Incorrect number of objects parsed. Expected 3, Got {len(df_parsed)}")

        if df_parsed['filename'].nunique() == 2:
            print("Correct number of unique filenames found.")
        else:
            print(f"Incorrect number of unique filenames. Expected 2, Got {df_parsed['filename'].nunique()}")

    else:
        print("Parsing returned an empty DataFrame. Check for errors.")

    # Clean up dummy files and directory
    # Note: In a real test suite, you might use a temporary directory fixture.
    file1_path = os.path.join(test_annotation_dir_abs, 'test_annotation1.xml')
    file2_path = os.path.join(test_annotation_dir_abs, 'test_annotation2.xml')

    if os.path.exists(file1_path):
      os.remove(file1_path)
    if os.path.exists(file2_path):
      os.remove(file2_path)
    if os.path.exists(test_annotation_dir_abs) and not os.listdir(test_annotation_dir_abs):
      os.rmdir(test_annotation_dir_abs)
    print("\nTest completed and dummy files cleaned up.")
