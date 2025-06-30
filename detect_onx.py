import onnxruntime as ort
import numpy as np
import cv2
import time

# Load ONNX model
session = ort.InferenceSession("yolov5/best.onnx", providers=['CPUExecutionProvider'])

# Preprocess image
img_path = 'data_images/test/N227.jpeg'
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_shape = (640, 640)

# Resize + normalize
resized = cv2.resize(image_rgb, input_shape)
input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})[0]

# Post-process
conf_threshold = 0.4
nms_threshold = 0.5

boxes, scores, class_ids = [], [], []
for output in outputs[0]:
    conf = output[4]
    if conf > conf_threshold:
        x, y, w, h = output[:4]
        boxes.append([x - w / 2, y - h / 2, w, h])
        scores.append(conf)
        class_ids.append(np.argmax(output[5:]))

# Draw boxes
for box, score, cls in zip(boxes, scores, class_ids):
    x, y, w, h = map(int, box)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{cls} {score:.2f}"
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# Show result
cv2.imshow("ONNX Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
