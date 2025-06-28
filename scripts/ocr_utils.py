import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import logging
import numpy as np # For potential image manipulation with OpenCV if needed later
import cv2 # OpenCV might be used for preprocessing before OCR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_for_ocr(image_np):
    """
    Preprocesses an image crop for better OCR results.
    Args:
        image_np (numpy.ndarray): Image region (license plate) as a NumPy array (BGR format from OpenCV).
    Returns:
        PIL.Image: Preprocessed image in PIL format, suitable for Pytesseract.
    """
    if image_np is None or image_np.size == 0:
        logging.warning("preprocess_for_ocr: Received empty image.")
        return None

    try:
        # Convert from BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Convert to grayscale
        gray_image = pil_image.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0) # Factor 2 can be tuned

        # Optional: Binarization (thresholding) - can be tricky, depends on image
        # enhanced_image = enhanced_image.point(lambda x: 0 if x < 128 else 255, '1') # Example threshold

        # Optional: Denoising (median filter)
        # final_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
        final_image = enhanced_image # Keep it simpler for now

        logging.debug("Image preprocessed for OCR.")
        return final_image
    except Exception as e:
        logging.error(f"Error during OCR preprocessing: {e}")
        # Fallback to converting original numpy array to PIL Image if preprocessing fails
        try:
            if image_np.shape[2] == 3: # Color
                 return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            elif len(image_np.shape) == 2: # Grayscale
                 return Image.fromarray(image_np)
        except Exception as fallback_e:
            logging.error(f"Error in fallback image conversion for OCR: {fallback_e}")
        return None


def extract_text_from_image_crop(image_crop_np, lang='eng', psm=7):
    """
    Extracts text from a given image crop (NumPy array from OpenCV BGR format) using Pytesseract.

    Args:
        image_crop_np (numpy.ndarray): The image crop (region of interest) as a NumPy array.
        lang (str): Language for OCR (default 'eng').
        psm (int): Pytesseract Page Segmentation Mode.
                   Mode 7: Treat the image as a single text line.
                   Mode 6: Assume a single uniform block of text.
                   Mode 8: Treat the image as a single word.
                   Mode 13: Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
                   (See `pytesseract --help-psm` for more options)

    Returns:
        str: The extracted text, cleaned of common OCR noise for license plates.
             Returns an empty string if no text is found or an error occurs.
    """
    if image_crop_np is None or image_crop_np.size == 0:
        logging.warning("Received empty image crop for OCR.")
        return ""

    try:
        # Preprocess the image for better OCR
        preprocessed_pil_image = preprocess_for_ocr(image_crop_np)

        if preprocessed_pil_image is None:
            logging.error("Image preprocessing failed, cannot perform OCR.")
            return ""

        # Tesseract configuration options for license plates
        # --oem 3 (Default OCR Engine Mode)
        # -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 (Customize as needed)
        custom_config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        logging.debug(f"Running Pytesseract with lang='{lang}', config='{custom_config}'")
        text = pytesseract.image_to_string(preprocessed_pil_image, lang=lang, config=custom_config)

        # Clean the extracted text
        # Remove non-alphanumeric characters (except those common in some plates if needed)
        # Convert to uppercase
        cleaned_text = ''.join(char for char in text if char.isalnum()).upper()

        logging.info(f"OCR raw text: '{text.strip()}', Cleaned text: '{cleaned_text}'")
        return cleaned_text

    except pytesseract.TesseractNotFoundError:
        logging.error("Pytesseract Error: Tesseract is not installed or not in your PATH.")
        logging.error("Please install Tesseract OCR: https://github.com/tesseract-ocr/tesseract#installing-tesseract")
        # This is a system-level dependency.
        return "" # Cannot proceed without Tesseract
    except Exception as e:
        logging.error(f"An error occurred during OCR text extraction: {e}")
        return ""

if __name__ == '__main__':
    logging.info("Testing 07_ocr_utils.py")

    # To test this script, we need a sample image crop.
    # Since we don't have image loading utilities here yet (that's for inference pipeline),
    # we'll create a dummy NumPy array representing an image crop.
    # A real test would use an actual image file.

    # Create a simple dummy image (e.g., black background with white text-like shapes)
    # This is a very basic test and may not yield good OCR results without a real image.
    # It's more to test the flow of the functions.
    dummy_height, dummy_width = 50, 150

    # Create a black image (BGR format)
    dummy_plate_np = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)

    # Add some white "text" (very crudely)
    # For a real test, load an actual license plate image snippet
    try:
        # Simulating text "TEST123"
        # This is highly dependent on Tesseract's ability to see this as text.
        # It's better to test with a real image if Tesseract is installed.
        cv2.putText(dummy_plate_np, "TEST123", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        logging.info("Created a dummy image with text 'TEST123' for OCR testing.")

        # Test with PSM mode 7 (single line of text)
        extracted_text_psm7 = extract_text_from_image_crop(dummy_plate_np, psm=7)
        logging.info(f"Extracted text (PSM 7) from dummy image: '{extracted_text_psm7}'")

        # Test with PSM mode 13 (raw line)
        extracted_text_psm13 = extract_text_from_image_crop(dummy_plate_np, psm=13)
        logging.info(f"Extracted text (PSM 13) from dummy image: '{extracted_text_psm13}'")

        # Test with an empty image
        logging.info("\nTesting with an empty image:")
        empty_image = np.array([])
        text_from_empty = extract_text_from_image_crop(empty_image)
        logging.info(f"Extracted text from empty image: '{text_from_empty}' (Expected: '')")
        if text_from_empty == "":
            logging.info("Correctly handled empty image.")
        else:
            logging.error("Empty image test failed.")

    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract is not installed or not in PATH. OCR functionality cannot be tested.")
        logging.error("Please install Tesseract: https://github.com/tesseract-ocr/tesseract and ensure 'tesseract' command is available.")
    except ImportError as e:
        if 'pytesseract' in str(e):
            logging.error("Pytesseract library is not installed. Please install it: pip install pytesseract")
        elif 'PIL' in str(e) or 'Pillow' in str(e):
             logging.error("Pillow library is not installed. Please install it: pip install Pillow")
        elif 'cv2' in str(e):
            logging.error("OpenCV (cv2) library is not installed. Please install it: pip install opencv-python")
        else:
            logging.error(f"Import error: {e}. Some dependencies might be missing.")
    except Exception as e:
        logging.error(f"An error occurred during the test: {e}")

    logging.info("\nOCR utils script test finished.")
    logging.warning("Note: OCR accuracy with the dummy image is not indicative of real performance. "
                    "Test with actual license plate images for meaningful results.")
    logging.warning("Ensure Tesseract OCR engine is installed on your system for this script to work.")
