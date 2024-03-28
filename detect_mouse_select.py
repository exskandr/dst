import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import logging
import sqlite3

import utils


logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler()


max_objects = 3
tracked_objects = {}
free_positions = [i for i in range(1, max_objects + 1)]

# Database connection and table creation (assuming 'tracker_data.db')
conn = sqlite3.connect('tracker_data.db')
c = conn.cursor()

# Drop the tracked_objects table if it exists
c.execute('''DROP TABLE IF EXISTS tracked_objects''')
logging.info(f'DROP TABLE')
# Create the tracked_objects table
c.execute('''CREATE TABLE IF NOT EXISTS tracked_objects
             (position INTEGER PRIMARY KEY, tracker_id INTEGER)''')
logging.info(f'CREATE TABLE')


class MyTracker(sv.ByteTrack):
    def __init__(self):
        super().__init__()
        self.boxes = []

    def update(self, detections):
        self.boxes = detections.boxes.xywh.cpu().numpy()

    def get_boxes(self):
        return self.boxes


class MouseClickHandler:
    def __init__(self, detections, tracker, max_tracked_objects=max_objects):
        self.tracker = tracker
        self.max_tracked_objects = max_tracked_objects
        self.detections = detections
        self.selected_tracker_id = None

    def add_remove_object(self, tracker_id):
        tracker_id = str(tracker_id)
        # Check if tracker_id already exists
        c.execute("SELECT * FROM tracked_objects WHERE tracker_id = ?", (tracker_id,))
        existing_entry = c.fetchone()

        if existing_entry:
            logging.info(f'ID {tracker_id} is already being tracked (position: {existing_entry[0]})')
            print(f'ID {tracker_id} is already being tracked (position: {existing_entry[0]}')
            c.execute("DELETE FROM tracked_objects WHERE tracker_id = ?", (tracker_id,))
            conn.commit()
            self.update_tracked_objects()
            logging.info(f"Object removed from position {tracker_id}")
            return

        # Check for free positions and handle edge cases
        free_position_id = self.get_next_position_id()
        if free_position_id is None:
            logging.warning("No free positions available")
            return

        # Add new object to database
        c.execute("INSERT INTO tracked_objects (position, tracker_id) VALUES (?, ?)",
                  (free_position_id, tracker_id))
        conn.commit()

        logging.info(f'ID {tracker_id} added to tracking at position {free_position_id}')
        self.update_tracked_objects()

    def update_tracked_objects(self):
        # Fetch tracked objects from database
        c.execute("SELECT * FROM tracked_objects")
        tracked_objects.clear()
        for row in c.fetchall():
            tracked_objects[row[0]] = row[1]

    def get_free_position_id(self):
        # Check for empty positions in the database
        c.execute("SELECT * FROM tracked_objects")
        occupied_positions = [row[0] for row in c.fetchall()]
        free_positions = [i for i in range(1, max_objects + 1) if i not in occupied_positions]
        return free_positions[0] if free_positions else None

    def get_next_position_id(self):
        # Check if there are less than max_objects entries
        c.execute("SELECT COUNT(*) FROM tracked_objects")
        num_tracked_objects = c.fetchone()[0]
        if num_tracked_objects < max_objects:
            return self.get_free_position_id()

    def get_selected_tracker_id(self):
        return self.selected_tracker_id

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # check whether one of the existing objects is clicked
            for xyxy, tracker_id in zip(self.detections.xyxy, self.detections.tracker_id):
                x1, y1, x2, y2 = xyxy
                # check whether the mouse cursor is inside the object
                if x1 <= x <= x2 and y1 <= y <= y2:
                    # Save tracker-ID
                    self.selected_tracker_id = tracker_id
                    print(tracker_id)
                    # add tracker id
                    self.add_remove_object(self.selected_tracker_id)

    def get_list_tracked_objects(self) -> list:
        c.execute("SELECT * FROM tracked_objects")
        data = [row[1] for row in c.fetchall()]
        conn.commit()
        logging.info(f'get list tracked objects - {data}')
        return data

    def get_tracked_objects(self) -> dict:
        c.execute("SELECT * FROM tracked_objects")
        # data = dict(c.fetchall())
        data = {}
        for row in c.fetchall():
            data[row[0]] = row[1]
        conn.commit()
        logging.info(f'get dictionary tracked objects - {data}')
        return data


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

        cv2.namedWindow('Operator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Operator', mouse_handler.handle_click)
        # Using the ID selected by the cursor
        selected_tracker_id = mouse_handler.get_selected_tracker_id()
        if tracked_objects is not None:
            detections = detections[
                np.isin(detections.tracker_id,
                        mouse_handler.get_list_tracked_objects())
            ]

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
        cv2.imshow('Operator', frame)

        # Processing of keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()