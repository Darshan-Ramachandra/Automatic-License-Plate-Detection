import cv2
import numpy as np
import logging
import os
import sys

try:
    from . import ocr_utils
except ImportError:
    SCRIPT_DIR_P06 = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR_P06 not in sys.path:
        sys.path.append(SCRIPT_DIR_P06)
    import ocr_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, 'yolov5', 'runs', 'train', 'license_plate_model', 'weights', 'best.onnx')
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.5
class_list = ['number_plate']

def load_yolo_model(model_path=ONNX_MODEL_PATH):
    logging.info(f"Loading ONNX model from: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}.")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        logging.info("Model loaded successfully.")
        return net
    except Exception as e:
        logging.error(f"Error loading ONNX model: {e}")
        return None

def get_detections(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds, image

def non_maximum_suppression(predictions, original_image_shape):
    # YOLOv5 ONNX output: (batch, num_boxes, 85) for COCO, (batch, num_boxes, 6) for 1 class
    # [center_x, center_y, width, height, obj_conf, class_score...]
    h, w = original_image_shape[:2]
    boxes = []
    confidences = []
    class_ids = []
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    for pred in predictions:
        obj_conf = pred[4]
        class_scores = pred[5:]
        if len(class_scores) == 0:
            continue
        class_id = np.argmax(class_scores)
        conf = obj_conf * class_scores[class_id]
        if conf > CONFIDENCE_THRESHOLD:
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x = int((cx - bw / 2) * w / INPUT_WIDTH)
            y = int((cy - bh / 2) * h / INPUT_HEIGHT)
            bw = int(bw * w / INPUT_WIDTH)
            bh = int(bh * h / INPUT_HEIGHT)
            boxes.append([x, y, bw, bh])
            confidences.append(float(conf))
            class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    results = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, bw, bh = boxes[i]
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        results.append([x1, y1, x2, y2, class_list[class_ids[i]], confidences[i]])
    return results

def yolo_predictions(image_np, net):
    if net is None:
        logging.error("YOLO model (net) is None. Cannot make predictions.")
        return [], image_np
    if image_np is None or image_np.size == 0:
        logging.warning("Input image is empty.")
        return [], image_np
    raw_detections, _ = get_detections(image_np, net)
    final_boxes = non_maximum_suppression(raw_detections, image_np.shape)
    detection_results = []
    image_with_boxes = image_np.copy()
    for box in final_boxes:
        x1, y1, x2, y2, class_name, conf = box
        h, w = image_np.shape[:2]
        crop_x1, crop_y1 = max(0, x1), max(0, y1)
        crop_x2, crop_y2 = min(w, x2), min(h, y2)
        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            plate_crop_np = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
            extracted_text = ocr_utils.extract_text_from_image_crop(plate_crop_np)
            detection_results.append(((x1, y1, x2, y2), class_name, conf, extracted_text))
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {extracted_text} ({conf:.2f})"
            cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            logging.warning(f"Invalid crop dimensions for box [{x1},{y1},{x2},{y2}]. Skipping crop and OCR.")
    return detection_results, image_with_boxes

def main_inference():
    logging.info("Starting inference pipeline script (06_inference_pipeline.py)")
    net = load_yolo_model()
    if net is None:
        logging.error("Failed to load YOLO model. Cannot proceed with inference.")
        return
    sample_image_path = os.path.join(PROJECT_ROOT, 'data', 'sample_test_image.jpg')
    if not os.path.exists(sample_image_path):
        logging.error(f"Sample image for inference not found: {sample_image_path}")
        return
    image_np = cv2.imread(sample_image_path)
    if image_np is None:
        logging.error("Failed to load the sample image.")
        return
    detections, image_with_boxes = yolo_predictions(image_np, net)
    logging.info("\nInference Results:")
    if detections:
        for i, (box, class_name, conf, text) in enumerate(detections):
            logging.info(f"  Detection {i+1}:")
            logging.info(f"    Box: {box}, Class: {class_name}, Confidence: {conf:.2f}, OCR Text: '{text}'")
    else:
        logging.info("  No detections made.")
    output_image_path = os.path.join(PROJECT_ROOT, 'data', 'sample_detections_output.jpg')
    cv2.imwrite(output_image_path, image_with_boxes)
    logging.info(f"Image with detections saved at: {output_image_path}")

if __name__ == '__main__':
    logging.info("Running 06_inference_pipeline.py directly for testing.")
    main_inference()
