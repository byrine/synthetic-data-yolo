# we need to draw the labels back onto the image
# to check that the coordinates of the bboxes are correct

import cv2
import os
import glob

# Configuration
IMG_DIR = "output/images"
LBL_DIR = "output/labels"
VERIFY_DIR = "output/verification"


def visualize():
    # Get all images
    img_files = (glob.glob(os.path.join(IMG_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_DIR, "*.jpeg")) + glob.glob(
        os.path.join(IMG_DIR, "*.JPG")))

    if not img_files:
        print("No images found")
        return

    if not os.path.exists(VERIFY_DIR):
        os.makedirs(VERIFY_DIR)
        print(f"Folder {VERIFY_DIR} created")
    print("Press any key to see the next image. Press 'q' to quit.")

    for i, img_path in enumerate(img_files):
        # 1.Load Image
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # 2.Load matching Label
        # derived from "output/images/syn_0000_jpg" -> "output/labels/syn_0000.txt"
        # basename = os.path.basename(img_path).replace(".jpg", ".txt")
        # label_path = os.path.join(LBL_DIR, basename)

        # better to get the filename without extension
        filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        label_filename = filename_no_ext + ".txt"
        label_path = LBL_DIR + "/" + label_filename

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                # Parse YOLO format: class x_center y_center width height
                parts = line.strip().split()
                class_id = int(parts[0])
                x_c, y_c, w_norm, h_norm = map(float, parts[1:])

                # 3. UN-Normalize (The Reverse Math)
                # Convert % back to pixels to draw the box
                box_w = int(w_norm * w)
                box_h = int(h_norm * h)
                box_x_center = int(x_c * w)
                box_y_center = int(y_c * h)

                # calculate Top-Left Corner and Bottom-Right Corner for cv2.rectangle
                x1 = int(box_x_center - box_w / 2)
                y1 = int(box_y_center - box_h / 2)
                x2 = x1 + box_w
                y2 = y1 + box_h

                # Draw the box (Green, 2px thickness)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(image, "Stop Sign", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. Save Images
        save_path = VERIFY_DIR + "/" + f"verify_{i}.jpg"
        cv2.imwrite(save_path, image)
        print(f"Saved: {save_path}")
        """
        # Show the image
        cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)  # allow resizing
        cv2.resizeWindow("Verification", 500, 500)
        cv2.imshow("Verification (Press Key for Next", image)

        # Wait for key press
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    """


print(f"Done, go open the {VERIFY_DIR} folder to check your boxes")

if __name__ == "__main__":
    visualize()
