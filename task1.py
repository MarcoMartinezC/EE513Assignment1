import sys, platform, textwrap

import rawpy
import numpy as np

# Load the RAW image
# raw = rawpy.imread('DSC_6997.NEF')

# # Extract required parameters
# print("Raw Type:", raw.raw_type)
# print("Shape of raw_image_visible:", raw.raw_image_visible.shape)
# print("Dtype of raw_image_visible:", raw.raw_image_visible.dtype)
# print("Raw Pattern:\n", raw.raw_pattern)
# print("Color Desc:", raw.color_desc)
# print("Black Level per Channel:", raw.black_level_per_channel)
# print("White Level:", raw.white_level)


# import matplotlib.pyplot as plt

# # Extract a 30x30 region from the 2D raw data
# raw_crop = raw.raw_image_visible[1000:1030, 1000:1030]

# plt.figure(figsize=(5, 5))
# plt.imshow(raw_crop, cmap='gray')
# plt.title("30x30 RAW Pixel Data")
# plt.axis('on')
# plt.show()



# import matplotlib.image as mpimg

# # 1. Load RAW and extract the 30x30 region
# with rawpy.imread('DSC_6997.NEF') as raw:
#     raw_crop = raw.raw_image_visible[1000:1030, 1000:1030]

# # 2. Load JPEG and extract the same 30x30 region
# # (Note: Use your actual JPEG filename. The exact scene alignment might shift 
# # slightly if the camera applied lens distortion correction to the JPEG).
# jpg_img = mpimg.imread('DSC_6997.JPG')
# jpg_crop = jpg_img[1000:1030, 1000:1030]

# # 3. Set up the side-by-side plot
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # Plot RAW: Displays the raw sensor measurements (the checkerboard)
# axes[0].imshow(raw_crop, cmap='gray', interpolation='nearest')
# axes[0].set_title("30x30 RAW (1-Channel Mosaic)")
# axes[0].axis('off')

# # Plot JPEG: Displays the demosaiced, color-interpolated result
# axes[1].imshow(jpg_crop, interpolation='nearest')
# axes[1].set_title("30x30 JPEG (3-Channel RGB)")
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()


#GEMINI PART
# import sys, platform, textwrap

# import rawpy
# import numpy as np

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # --- Configuration ---
# CAPTURE_DIR = 'converted_images' # UPDATE THIS to your image folder
# SQUARE_MM = 20.0                    # UPDATE THIS to your measured physical size
# BOARD = (9, 6)

# # Load image paths
# folder = Path(CAPTURE_DIR)
# files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".dng"}])

# # ---------------------------------------------------------
# # Deliverable 2: Contact Sheet
# # ---------------------------------------------------------
# fig, axes = plt.subplots(3, 8, figsize=(24, 9))
# for ax, f in zip(axes.ravel(), files[:24]): # Plots the first 24 images
#     # Read and resize for a quick contact sheet
#     img = cv2.imread(str(f))
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     ax.imshow(img_rgb)
#     ax.axis('off')
# fig.suptitle("Contact Sheet of Calibration Captures")
# plt.tight_layout()
# plt.show()

# # ---------------------------------------------------------
# # Deliverable 3: Find Corners
# # ---------------------------------------------------------
# objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE_MM

# obj_points = []
# img_points = []
# drawn_img = None
# img_shape = None

# for f in files:
#     img = cv2.imread(str(f))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if img_shape is None:
#         img_shape = gray.shape[::-1] # (width, height)
    
#     # Detect corners
#     ok, corners = cv2.findChessboardCorners(
#         gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
#     )
    
#     if ok:
#         # Refine corner locations to sub-pixel accuracy
#         crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        
#         obj_points.append(objp)
#         img_points.append(corners_refined)
        
#         # Save one visualization for the deliverable
#         if drawn_img is None:
#             drawn_img = img.copy()
#             cv2.drawChessboardCorners(drawn_img, BOARD, corners_refined, ok)

# print(f"Deliverable 3 Count: {len(img_points)} out of {len(files)} captures succeeded in finding corners.")

# if drawn_img is not None:
#     plt.figure(figsize=(10, 7))
#     plt.imshow(cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Calibration Corners")
#     plt.axis('off')
#     plt.show()

# # ---------------------------------------------------------
# # Deliverable 4: Calibrate Camera
# # ---------------------------------------------------------
# if len(img_points) > 0:
#     rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
#         obj_points, img_points, img_shape, None, None
#     )
#     print(f"Deliverable 4: RMS Reprojection Error = {rms:.4f} px")
# else:
#     print("Not enough successful corner detections to calibrate.")

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CAPTURE_DIR = "converted_images"   # folder of JPEG/PNG/TIFF captures
SQUARE_MM = 20.0                   # measured physical square size
BOARD = (9, 6)                     # INNER corners (squares - 1)
DETECT_SCALE = 0.5                 # detect on a smaller copy for speed; refine at full res
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}   # no .dng: cv2.imread can't decode raw

folder = Path(CAPTURE_DIR)
if not folder.is_dir():
    sys.exit(f"Folder not found: {folder.resolve()}  (run from the right directory?)")
files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
if not files:
    sys.exit(f"No usable images in {folder.resolve()}")
print(f"{len(files)} image files found")


def load_bgr(path):
    """Read an image, or return None (and say why) instead of crashing later."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"  could not decode, skipping: {path.name}")
    return img


# ---------------- Deliverable 2: contact sheet ----------------
n = min(24, len(files))
fig, axes = plt.subplots(3, 8, figsize=(24, 9))
for ax in axes.ravel():
    ax.axis("off")
for ax, f in zip(axes.ravel(), files[:n]):
    img = load_bgr(f)
    if img is None:
        continue
    thumb = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)  # small, saves RAM
    ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
    ax.set_title(f.name, fontsize=7)
fig.suptitle("Contact Sheet of Calibration Captures")
plt.tight_layout()
plt.show()

# ---------------- Deliverable 3: find corners ----------------
objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE_MM

obj_points, img_points, used = [], [], []
drawn_img, img_shape = None, None
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

for f in files:
    img = load_bgr(f)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None:
        img_shape = gray.shape[::-1]                       # (width, height)
    elif gray.shape[::-1] != img_shape:                    # e.g. portrait vs landscape
        print(f"  size {gray.shape[::-1]} != {img_shape}, skipping: {f.name}")
        continue

    small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
    ok, corners = cv2.findChessboardCorners(small, BOARD, flags)
    if not ok:
        print(f"  no corners: {f.name}")
        continue
    corners = (corners / DETECT_SCALE).astype(np.float32)  # back to full-res pixel coords
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_points.append(objp)
    img_points.append(corners)
    used.append(f)
    if drawn_img is None:
        drawn_img = img.copy()
        cv2.drawChessboardCorners(drawn_img, BOARD, corners, ok)

print(f"Deliverable 3 Count: {len(img_points)} out of {len(files)} captures succeeded in finding corners.")

if drawn_img is not None:
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Calibration Corners")
    plt.axis("off")
    plt.show()

# ---------------- Deliverable 4: calibrate ----------------
if len(img_points) >= 3:      # need several, differently tilted views
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
    print(f"Deliverable 4: RMS Reprojection Error = {rms:.4f} px")
    print("K =\n", K, "\ndist =", dist.ravel())
else:
    print("Not enough successful corner detections to calibrate.")