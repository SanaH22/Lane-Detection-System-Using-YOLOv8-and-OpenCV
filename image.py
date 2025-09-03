import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64

# --- Lane detection ---
def region_of_interest(img):
    height, width = img.shape[:2]
    # trapezoid ROI
    polygons = np.array([[
        (int(0.1*width), height),
        (int(0.35*width), int(0.4*height)),
        (int(0.65*width), int(0.4*height)),
        (int(0.9*width), height)
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def lane_detection(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]

    # White mask (bright lanes)
    white_mask = cv2.inRange(l_channel, 200, 255)

    # Yellow mask
    lower_yellow = np.array([18, 120, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    # Combine masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply ROI
    mask = region_of_interest(mask)

    # Edge detection
    edges = cv2.Canny(mask, 50, 150)

    # Get lane pixels
    ys, xs = np.where(edges == 255)
    if len(xs) == 0:
        return img  # fallback

######
    center = img.shape[1] // 2
    left_xs, left_ys = xs[xs < center], ys[xs < center]
    right_xs, right_ys = xs[xs >= center], ys[xs >= center]

    left_fitx, right_fitx = None, None
    height = img.shape[0]

            # Force yellow lane lines to follow the full green fill range
    y_bottom = height - 1
    if len(left_ys) > 0 and len(right_ys) > 0:
        y_top = max(min(left_ys), min(right_ys))  # common top
    elif len(left_ys) > 0:
        y_top = min(left_ys)
    elif len(right_ys) > 0:
        y_top = min(right_ys)
    else:
        y_top = 0

    ploty = np.linspace(y_top, y_bottom, y_bottom - y_top + 1)

        # --- LEFT lane ---
    if len(left_xs) > 0:
        left_fit = np.polyfit(left_ys, left_xs, 2)
        y_min, y_max = min(left_ys), max(left_ys)
        ploty = np.linspace(y_min, y_max, y_max-y_min+1)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        cv2.polylines(img, pts, isClosed=False, color=(0, 255, 255), thickness=5)

    # --- RIGHT lane ---
    if len(right_xs) > 0:
        right_fit = np.polyfit(right_ys, right_xs, 2)
        y_min, y_max = min(right_ys), max(right_ys)
        ploty = np.linspace(y_min, y_max, y_max-y_min+1)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
        cv2.polylines(img, pts, isClosed=False, color=(0, 255, 255), thickness=5)


        ########

        # --- Fill lane only if both exist ---
    if left_fitx is not None and right_fitx is not None:
        # extend lane fill from bottom of the image to top of detected lanes
        common_y_min = min(min(left_ys), min(right_ys))   # top-most detected point
        common_y_max = height - 1                        # force bottom of image
        ploty = np.linspace(common_y_min, common_y_max, common_y_max-common_y_min+1)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_points = np.transpose(np.vstack([left_fitx, ploty]))
        right_points = np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
        lane_points = np.vstack((left_points, right_points)).astype(np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [lane_points], (0, 255, 0))
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    return img



# --- Car detection ---
def car_detection(img, model):
    results = model(img)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if model.names[cls] == 'car' and conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'car {conf:.2f}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

# --- Main ---
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img = cv2.resize(img, (1280, 720))

    # Lane detection
    img = lane_detection(img)
# Car detection
    model = YOLO('models/yolov8n.pt')
    img = car_detection(img, model)

    # Convert BGR → RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Save to memory buffer instead of disk
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Encode as base64 string
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64

