import os
from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import cv2
from image import process_image
from video import pipeline  # your lane detection pipeline
from ultralytics import YOLO  # YOLOv8

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

# Handle image upload
@app.route("/upload-image", methods=["POST"])
def upload_image():
    file = request.files["image_file"]
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        processed_base64 = process_image(filepath)
        return render_template(
            "index.html",
            output_image=processed_base64,
            output_type="image"
        )
    return redirect(url_for("home"))

# Handle video upload
@app.route("/upload-video", methods=["POST"])
def upload_video():
    file = request.files["video_file"]
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        return render_template(
            "index.html",
            output_file=file.filename,  # just the filename
            output_type="video"
        )
    return redirect(url_for("home"))

# Serve uploaded video file directly
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Stream video with lane detection
@app.route("/video_feed/<filename>")
def video_feed(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return Response(process_video_file(filepath),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# Generator function for streaming
def process_video_file(input_path):
    model = YOLO('weights/yolov8n.pt')
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lane_frame = pipeline(frame)  # lane detection

        results = model(frame)  # YOLO detection
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                if model.names[cls] == 'car' and conf >= 0.3:
                    # draw bounding box
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(lane_frame, f'car {conf:.2f}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # distance estimation
                    distance = (2.0 * 1000) / (x2 - x1)  # same as estimate_distance
                    cv2.putText(lane_frame, f'Distance: {distance:.2f}m', (x1, y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # encode frame
        ret, buffer = cv2.imencode('.jpg', lane_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    app.run(debug=True, port=5006)
