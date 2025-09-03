from flask import Flask, request, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

app = Flask(__name__)

# Load YOLO segmentation model once
model = YOLO('weights/yolov8n.pt')

# Lane detection and helper functions
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0]):
    line_img = np.zeros_like(img)
    pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], np.int32)
    cv2.fillPoly(line_img, pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0)

def pipeline(image):
    h, w = image.shape[:2]
    roi = [(0, h), (w / 2, h / 2), (w, h)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cropped = region_of_interest(edges, np.array([roi], np.int32))
    lines = cv2.HoughLinesP(cropped, 6, np.pi / 60, 160, minLineLength=40, maxLineGap=25)
    left_x, left_y, right_x, right_y = [], [], [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
    min_y = int(h * 3 / 5)
    max_y = h
    poly_left = np.poly1d(np.polyfit(left_y, left_x, 1)) if left_x else lambda y: 0
    poly_right = np.poly1d(np.polyfit(right_y, right_x, 1)) if right_x else lambda y: w
    lane_img = draw_lane_lines(
        image,
        [int(poly_left(max_y)), max_y, int(poly_left(min_y)), min_y],
        [int(poly_right(max_y)), max_y, int(poly_right(min_y)), min_y]
    )
    return lane_img

def estimate_distance(bbox_width):
    focal_length = 1000
    known_width = 2.0
    return (known_width * focal_length) / bbox_width

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', output_type=None)

@app.route('/upload-video', methods=['POST'])
def upload_video():
    file = request.files['video_file']
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    file.save(tmp_file.name)
    cap = cv2.VideoCapture(tmp_file.name)

    def generate():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            lane_frame = pipeline(frame)

            results = model(frame)

            for result in results:
                if result.masks is not None:
                    masks_array = result.masks.masks.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for mask, conf, cls in zip(masks_array, confs, classes):
                        if model.names[cls] == "car" and conf >= 0.5:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if len(contours) == 0:
                                continue
                            largest_contour = max(contours, key=cv2.contourArea)

                            # Bounding rect
                            x, y, w, h = cv2.boundingRect(largest_contour)

                            frame_height = frame.shape[0]

                            # Bounding rect
                            x, y, w, h = cv2.boundingRect(largest_contour)

                            frame_height, frame_width = frame.shape[:2]

                            #  Ignore detections overlapping bottom 20% of frame
                            if y + h > int(frame_height * 0.8):
                                continue

                            # Ignore very small contours
                            if cv2.contourArea(largest_contour) < 1500:
                                continue

                            # Draw car mask contour
                            cv2.polylines(lane_frame, [largest_contour], isClosed=True, color=(0, 255, 255), thickness=2)
                            distance = estimate_distance(w)

                            cv2.putText(lane_frame, f"car {conf:.2f}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(lane_frame, f"Distance: {distance:.2f}m", (x, y + h + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', lane_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
        os.unlink(tmp_file.name)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
