# auto_crop.py
import cv2
import os
import sys

MAX_DISPLAY_WIDTH = 1000
MAX_DISPLAY_HEIGHT = 800

# === Input ===
image_path = sys.argv[1]  # e.g., uploads/20250630_182033/20250630_182033_front.jpg

if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    exit()

# === Parse original path and info ===
uploads_folder = os.path.normpath(image_path).split(os.sep)
timestamp_folder = uploads_folder[-2]  # Get the folder name like '20250630_182033'
basename = os.path.basename(image_path)  # e.g., 20250630_182033_front.jpg
name, ext = os.path.splitext(basename)

# === Setup cropped folder ===
cropped_dir = os.path.join("cropped", timestamp_folder)
os.makedirs(cropped_dir, exist_ok=True)

# === Read Image ===
original = cv2.imread(image_path)
orig_h, orig_w = original.shape[:2]
scale = min(MAX_DISPLAY_WIDTH / orig_w, MAX_DISPLAY_HEIGHT / orig_h, 1.0)
disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
display = cv2.resize(original, (disp_w, disp_h))

# === Initial Crop Box ===
box_w = 400
box_h = 600
ix = (disp_w - box_w) // 2
iy = (disp_h - box_h) // 2
ex = ix + box_w
ey = iy + box_h
dragging = False
resizing = False

def mouse_callback(event, x, y, flags, param):
    global ix, iy, ex, ey, dragging, resizing
    if event == cv2.EVENT_LBUTTONDOWN:
        if abs(x - ex) < 10 and abs(y - ey) < 10:
            resizing = True
        else:
            dragging = True
            param['offset'] = (x - ix, y - iy)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx, dy = param['offset']
            w, h = ex - ix, ey - iy
            ix = max(0, min(x - dx, disp_w - w))
            iy = max(0, min(y - dy, disp_h - h))
            ex = ix + w
            ey = iy + h
        elif resizing:
            ex = min(disp_w, max(ix + 50, x))
            ey = min(disp_h, max(iy + 50, y))
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        resizing = False

cv2.namedWindow("Crop Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Crop Image", mouse_callback, {'offset': (0, 0)})

while True:
    view = display.copy()
    cv2.rectangle(view, (ix, iy), (ex, ey), (0, 255, 0), 2)
    cv2.putText(view, "Drag/resize | Press 'c' to crop | 'q' to quit",
                (10, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Crop Image", view)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        scale_x = orig_w / disp_w
        scale_y = orig_h / disp_h
        x1, y1 = int(ix * scale_x), int(iy * scale_y)
        x2, y2 = int(ex * scale_x), int(ey * scale_y)
        cropped = original[y1:y2, x1:x2]

        # === Save to cropped/timestamp/filename_cropped.jpg ===
        cropped_filename = f"{name}_cropped{ext}"
        cropped_path = os.path.join(cropped_dir, cropped_filename)
        cv2.imwrite(cropped_path, cropped)
        print(cropped_path)
        break

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
