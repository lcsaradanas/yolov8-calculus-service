from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = FastAPI()

# Load model once at startup
MODEL_PATH = "mbest.pt"
model = YOLO(MODEL_PATH)


def draw_boxes_with_labels(image, result):
    """
    Draw bounding boxes and confidence labels.
    """
    if result is None or len(result.boxes) == 0:
        return image

    colors = Colors()

    xyxy = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, cls_id, conf in zip(xyxy, cls_ids, confs):

        x1, y1, x2, y2 = [int(v) for v in box]

        color = tuple(
            int(c) for c in colors(int(cls_id), bgr=True)
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            image,
            f"{conf:.2f}",
            (x1 + 5, y1 + 18),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return image


@app.get("/")
def health_check():
    return {
        "status": "running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        image_bytes = await file.read()

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image_np = np.array(pil_image)

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        # Run YOLO prediction
        results = model.predict(
            source=image_bgr,
            imgsz=640,
            conf=0.25,
            save=False,
            show=False,
            verbose=False
        )

        result = results[0]

        detections = []

        confidences = []

        if len(result.boxes) > 0:

            for box in result.boxes:

                confidence = float(box.conf[0])
                confidences.append(confidence)

                detections.append({
                    "confidence": round(
                        confidence * 100,
                        2
                    ),
                    "bbox": [
                        round(v, 2)
                        for v in box.xyxy[0].tolist()
                    ]
                })

        if confidences:
            average_confidence = sum(confidences) / len(confidences)
        else:
            average_confidence = 0.0
        # Number of detected calculus regions
        calculus_amount = len(result.boxes)

        # Calculus detected
        calculus_detected = (
            "Yes"
            if calculus_amount > 0
            else "No"
        )

        # Oral health classification
        #
        # Adjust these thresholds later
        # based on adviser recommendation
        #
        if calculus_amount == 0:
            oral_health_status = "Healthy"

        elif calculus_amount <= 5:
            oral_health_status = "Mild"

        else:
            oral_health_status = "Unhealthy"

        # Draw annotations
        annotated = draw_boxes_with_labels(
            image_bgr.copy(),
            result
        )

        # Convert image to JPEG bytes
        success, buffer = cv2.imencode(
            ".jpg",
            annotated
        )

        if not success:
            raise Exception(
                "Failed to encode image"
            )

        encoded_image = base64.b64encode(
            buffer
        ).decode("utf-8")

        return JSONResponse({
            "success": True,

            "calculus_detected":
                calculus_detected,

            "calculus_amount":
                calculus_amount,

            "oral_health_status":
                oral_health_status,

            "average_confidence":
                round(
                    average_confidence * 100,
                    2
                ),

            "detections":
                detections,

            "annotated_image":
                encoded_image
        })

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
