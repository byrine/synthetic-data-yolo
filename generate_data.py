import cv2
import numpy as np
import os
import random
import glob
import albumentations as A

# configuration
BACKGROUND_DIR = "assets"  # empty road images
SIGN_PATH = "assets/sign.png"
OUTPUT_IMG_DIR = "output/images"
OUTPUT_LBL_DIR = "output/labels"
NUM_IMAGES_TO_GENERATE = 50

# Make sure output directories exist
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# Define the Augmentation Pipeline
# p is probability
transform = A.Compose([A.RandomBrightnessContrast(p=0.5),  # Simulate day/night/shadows
                       A.MotionBlur(blur_limit=7, p=0.3),  # Simulate car  moving fast
                       A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=0.3),  # fog
                       A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.3),  # Rain
                       A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Grainy Camera Sensor
                       ])


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """
    Pastes a transparent image (overlay) onto a background (img) at (x,y)
    :param img:
    :param img_overlay:
    :param x:
    :param y:
    :param alpha_mask:
    :return:
    """
    # Image ranges
    h, w = img_overlay.shape[0:2]  #: .shape returns 3 numbers (Height, Width, Channels)

    # Take the region of the background where the sign will be placed
    background_slice = img[y:y + h, x:x + w]

    # 1.normalize alpha mask to keep values between 0 and 1
    alpha = alpha_mask / 255
    alpha = np.dstack([alpha] * 3)  # make it 3 channels (R,G,B) to match image

    # 2.perform the overlay math: (Foreground*Alpha) + (Background * (1-Alpha))
    composite = (img_overlay[:, :, :3] * alpha) + (background_slice * (1.0 - alpha))

    # 3. Put the combined piece back into the main image
    img[y:y + h, x:x + w] = composite
    return img


def create_synthetic_data():
    # 1.Load Assets
    sign_img_full = cv2.imread(SIGN_PATH, cv2.IMREAD_UNCHANGED)  # Load with Alpha Channel
    if sign_img_full is None:
        print(f"Error: Could not load {sign_img_full}")

    background_files = (
            glob.glob(os.path.join(BACKGROUND_DIR, "*.jpg")) +
            glob.glob(os.path.join(BACKGROUND_DIR, "*.jpeg")) +
            glob.glob(os.path.join(BACKGROUND_DIR, "*.JPG"))
    )
    if not background_files:
        print("Error: No background .jpg images found in assets/")
        return

    print(f"Starting generation of {NUM_IMAGES_TO_GENERATE} images...")

    for i in range(NUM_IMAGES_TO_GENERATE):
        # Pick random background
        bg_path = random.choice(background_files)
        background = cv2.imread(bg_path)
        bg_h, bg_w = background.shape[:2]

        # Resize sign randomly (simulates distance)
        scale = random.uniform(0.1, 0.3)  # Sign will be 10% to 30% of image size
        new_w = int(sign_img_full.shape[1] * scale)
        new_h = int(sign_img_full.shape[0] * scale)
        sign_resized = cv2.resize(sign_img_full, (new_w, new_h))

        # Extract Alpha Mask (Transparency) and RGB channels
        sign_rgb = sign_resized[:, :, :3]
        sign_alpha = sign_resized[:, :, 3]

        # Pick random position (ensure it says inside image)
        max_x = bg_w - new_w
        max_y = bg_h - new_h
        x_pos = random.randint(0, max_x)
        y_pos = random.randint(0, max_y)

        # Overlay the sign
        final_image = overlay_image_alpha(background, sign_rgb, x_pos, y_pos, sign_alpha)

        #--- Apply Albumentations---#
        # Albumentations expects RGB but OpenCV uses BGR
        # We need to convert to RGB then Augment then back to BGR
        scene_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=scene_rgb)["image"]
        final_image = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)



        # --- CALCULATE YOLO COORDINATES ---#
        # YOLO format: class_id x_center y_center width height (normalized 0.0-1.0)
        x_center = (x_pos + new_w / 2) / bg_w
        y_center = (y_pos + new_h / 2) / bg_h
        width_norm = new_w / bg_w
        height_norm = new_h / bg_h

        # Save Image
        filename = f"sync_{i:04d}"
        cv2.imwrite((f"{OUTPUT_IMG_DIR}/{filename}.jpg"), final_image)

        # Save Label (Class ID 0 for Stop Sign)
        with open(f"{OUTPUT_LBL_DIR}/{filename}.txt", "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        print("Done! Check the output folder")


if __name__ == "__main__":
    create_synthetic_data()
