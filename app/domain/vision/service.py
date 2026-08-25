from ultralytics import YOLO

from app.domain.vision.constants import (
    CALIBRATED_CONF,
    CALIBRATED_IMGSZ,
    CALIBRATED_IOU,
    COUNT_MODERATELY_STOCKED,
    COUNT_WELL_STOCKED,
)


def load_shelf_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def analyze_shelf(shelf_model: YOLO, image_path: str) -> dict:
    """Run YOLO detection and compute a count-based shelf status.

    Known limitation: distant/small objects in wide-angle shots are
    undercounted due to limited training epochs and base resolution.
    Thresholds calibrated empirically against sample images, not
    derived analytically.
    """
    results = shelf_model.predict(
        source=image_path,
        save=False,
        conf=CALIBRATED_CONF,
        imgsz=CALIBRATED_IMGSZ,
        iou=CALIBRATED_IOU,
    )
    num_detections = len(results[0].boxes)

    if num_detections >= COUNT_WELL_STOCKED:
        status = "well-stocked"
    elif num_detections >= COUNT_MODERATELY_STOCKED:
        status = "moderately stocked"
    else:
        status = "low stock / significant gaps"

    return {"num_items_detected": num_detections, "status": status}
