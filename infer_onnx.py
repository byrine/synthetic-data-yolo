import cv2
import numpy as np
import onnxruntime as ort
import os

# --- Configuration --- #
MODEL_PATH = "models/best.onnx"
TEST_IMAGE_DIR = "dataset/val/images"
OUTPUT_DIR = "output/onnx_results"
CONFIDENCE_THRESHOLD = 0.5


def preprocess(image):
    """
    Manually prepare the image for the AI (Resize -> Normalize -> Transpose)
    :param image:
    :return:
    """

    # 1. Resize to 640*640 (Square)
    img_resized = cv2.resize(image, (640, 640))

    # 2.Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3. Transpose Dimensions: (Height, Width, Channels) -> (Channels, Height, Width)
    img_transposed = img_rgb.transpose(2, 0, 1)

    # 4. Add Batch Dimension: (1,3, 640, 640) and Normalize (0-1)
    input_tensor = np.expand_dims(img_transposed, axis=0)
    input_tensor = input_tensor.astype(np.float32) / 255.0

    return input_tensor, image.shape[:2]


def postprocess(outputs, original_shape):
    """
    Manually decode the raw numbers from the AI back into a Bounding Box
    :param outputs:
    :param original_shape:
    :return:
    """
    # YOLO output shape is [1, 5, 8400]. We squeeze it to [5, 8400] then transpose to [8400, 5]
    predictions = np.transpose(outputs[0], (0, 2, 1))
    predictions = predictions[0]

    h_orig, w_orig = original_shape
    x_scale = w_orig / 640
    y_scale = h_orig / 640

    detected_boxes = []

    for row in predictions:
        # row format: [x,y,w,h, probability]
        score = row[4]

        if score >= CONFIDENCE_THRESHOLD:
            xc, yc, w, h = row[0:4]

            # Scale back to original image size
            x_center = xc * x_scale
            y_center = yc * y_scale
            width = w * x_scale
            height = h * y_scale

            # Calculate Top_Left corner
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            detected_boxes.append([x1, y1, x2, y2, score])

    return detected_boxes


def run_inference():
    # 1. Load the ONNX Model (The "Brain")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading ONNX Model")
    session = ort.InferenceSession(MODEL_PATH)

    # Setup Output Folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Pick a random image to test
    files = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith((".jpg", ".jpeg"))]
    if not files:
        print("No images found to test.")
        return
    test_file = files[0]  # just pick the first one
    img_path = os.path.join(TEST_IMAGE_DIR, test_file)

    original_img = cv2.imread(img_path)
    print(f"Testing on {img_path}")

    # 3. Preprocess
    input_tensor, original_shape = preprocess(original_img)

    # 4. Run Inference (The Magic)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # 5.Postprocess
    boxes = postprocess(outputs, original_shape)
    print(f"Found {len(boxes)} detections")

    # 6. Draw and Save
    for (x1, y1, x2, y2, score) in boxes:
        # Draw Green Box
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw Label
        label = f"Stop: {score: .2f}"
        cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    save_path = os.path.join(OUTPUT_DIR, "onnx_result.jpg")
    cv2.imwrite(save_path, original_img)
    print(f"Result saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    run_inference()
