import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

import utils

tracked_objects = []


class MyTracker(sv.ByteTrack):
    def __init__(self):
        super().__init__()
        self.boxes = []

    def update(self, detections):
        self.boxes = detections.boxes.xywh.cpu().numpy()

    def get_boxes(self):
        return self.boxes


class MouseClickHandler:
    def __init__(self, detections, tracker, max_tracked_objects=3):
        self.tracker = tracker
        self.max_tracked_objects = max_tracked_objects
        self.detections = detections
        self.selected_tracker_id = None

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # check whether one of the existing objects is clicked
            for xyxy, tracker_id in zip(self.detections.xyxy, self.detections.tracker_id):
                x1, y1, x2, y2 = xyxy
                # check whether the mouse cursor is inside the object
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Save tracker-ID
                    self.selected_tracker_id = tracker_id
                    # add tracker id
                    if self.selected_tracker_id not in tracked_objects:
                        if len(tracked_objects) < self.max_tracked_objects:
                            tracked_objects.append(self.selected_tracker_id)
                        print(f"tracked_objects: {tracked_objects}")
                    else:
                        # Removes an object from the list if it is already there
                        tracked_objects.remove(self.selected_tracker_id)
                    break

    def get_tracked_objects(self):
        return tracked_objects

    def get_selected_tracker_id(self):
        return self.selected_tracker_id


def main():
    model = YOLO("yolov8n.pt")
    source = "people-walking.mp4"
    tracker = MyTracker()

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    corner_annotator = sv.BoxCornerAnnotator(
        color=sv.Color(255, 0, 0),
        thickness=3
    )

    trace_annotator = sv.TraceAnnotator(
        trace_length=20,
        color=sv.Color(255, 0, 0))

    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color(0, 0, 255))

    # Getting results from YOLO
    for result in model.track(source=source,
                              show=False,
                              stream=True,
                              agnostic_nms=True):

        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        detections = tracker.update_with_detections(detections)
        mouse_handler = MouseClickHandler(detections, tracker)

        cv2.namedWindow('yolov8', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('yolov8', mouse_handler.handle_click)
        # Using the ID selected by the cursor
        selected_tracker_id = mouse_handler.get_selected_tracker_id()
        if tracked_objects is not None:
            detections = detections[np.isin(detections.tracker_id, tracked_objects)]

        # # Visualization of the text of the selected object in the upper left corner
        for tracker_id in zip(detections.tracker_id):
            if tracker_id == selected_tracker_id:
                utils.draw_tracked_objects(frame, tracker_id)
            utils.draw_tracked_objects(frame, mouse_handler.get_tracked_objects())

        # Visualization of frames
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence,
                   detections.class_id,
                   detections.tracker_id)
        ]

        annotated_frame = corner_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections=detections,
            labels=labels)
        frame = trace_annotator.annotate(annotated_frame, detections=detections)

        # Frame display
        cv2.imshow('yolov8', frame)

        # Processing of keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
