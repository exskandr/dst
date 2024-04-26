import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import logging
import sqlite3

import utils


logging.basicConfig(filename='ip_cam.log', filemode='w', level=logging.INFO)
handler = logging.StreamHandler()
current_tracker_id = None

# Database connection and table creation (assuming 'tracker_data.db')
conn = sqlite3.connect('tracker_data.db')
c = conn.cursor()
logging.info(f'Connected to database')


def get_tracked_objects() -> dict:
    """
    Fetches tracked objects from the database as a dictionary.


    Returns:
        A dictionary where keys are tracker IDs and values are additional data.
    """
    c.execute("SELECT * FROM tracked_objects")
    data = {}
    for row in c.fetchall():
        data[row[0]] = row[1]
    return data


def get_list_tracked_object(dictionary: dict, n) -> list or None:
    """
     A function that creates a list from the value dictionary or returns None.

     Args:
         dictionary: The dictionary from which to retrieve the value.

     Returns:
         A list of dictionary value or None if value does not exist or is not a list.
    """
    if not dictionary:
        return None

    data = []
    for key, value in dictionary.items():
        if key == n:
            data.append(value)
            return data


def select_id(n):
    """
    Selects a tracker ID from the database based on position (n).

    Args:
        n: The position value used to identify the desired tracker ID.

    Returns:
        The tracker ID as an integer, or None if not found.
    """
    global current_tracker_id

    position = str(n)
    c.execute("SELECT tracker_id FROM tracked_objects WHERE position = ?", (position,))
    data = c.fetchone()
    if data:
        current_tracker_id = data[0]
        return current_tracker_id
    else:
        logging.info(f'No tracking ID found for position {n}')
        return None


# url = f'http://{utils.ip_cam}/videostream.cgi?loginuse={utils.login}&loginpas={utils.password}'
url = 0
logging.info(f"{url}")


def device(tracker_id, n):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(url)
    frame_count = 0  # Initialize frame counter
    cadr = 5
    tracker = sv.ByteTrack()  # lost_track_buffer=40, frame_rate=30

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    corner_annotator = sv.BoxCornerAnnotator(
        color=sv.Color(255, 0, 0),
        thickness=3
    )

    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color(0, 0, 255))

    # Getting results from YOLO
    while True:
        ret, frame = cap.read()
        frame_count += 1
        for result in model.track(source=frame,
                                  show=False,
                                  stream=True,
                                  agnostic_nms=True):

            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            detections = tracker.update_with_detections(detections)

            cv2.namedWindow(f'Drone-{n}', cv2.WINDOW_NORMAL)

            # Filter detections based on selected tracker ID
            if tracker_id is not None:
                detections = detections[
                    np.isin(detections.tracker_id,
                            get_list_tracked_object(get_tracked_objects(), n))
                ]
                # detections = detections[detections.tracker_id == get_tracked_objects()[1]]

            # # Visualization of the text of the selected object in the upper left corner
            for tracker_id in zip(detections.tracker_id):
                utils.draw_object_for_tracking(frame, get_tracked_objects(), n)
                utils.draw_center(frame)
            ######
            # Coordinate analysis and camera control every cadr=20 frames
            if frame_count % cadr == 0:
                utils.move_cam(frame, detections)

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
            frame = label_annotator.annotate(
                annotated_frame,
                detections=detections,
                labels=labels)

            cv2.imshow(f'Drone-{n}', frame)

            # Processing of keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Close database connection after processing
    conn.close()
    logging.info(f'Database connection closed')


def main():
    # n = int(sys.argv[1])
    n = 1
    device(select_id(n), n)


if __name__ == "__main__":
    main()
# cmd python ip_cam.py 5