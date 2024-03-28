import cv2


def draw_tracked_objects(frame, tracked_objects):
    """
    Displays the tracked_objects list in the upper left corner of the window.

    Args:
        frame: Numpy image array.
        tracked_objects: Dict of monitored objects.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)
    thickness = 2
    x, y = 20, 40

    if not tracked_objects:
        # Displaying the text "No objects found"
        text = "No objects selected!!!"
        font_color = (0, 0, 255)
        font_scale = 0.7
        cv2.putText(frame, text, (10, 30), font, font_scale, font_color, thickness)
        return


    # Display the ID of each tracked object(for dict)
    for key, value in tracked_objects.items():
        text = f"DRONE: {key} - ID: {value}"
        font_scale = 0.9
        cv2.putText(frame, text, (x, y + 35 * (key-1)), font, font_scale, font_color, thickness)

        # Frame display
        cv2.rectangle(frame, (x - 10, y - 30), (x + 280, y + 10 + 35 * (key-1)), (255, 255, 0), thickness)


# draw in uaw
def draw_object_for_tracking(frame, tracked_objects, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)
    thickness = 2
    x, y = 20, 40

    if not tracked_objects:
        # Displaying the text "No objects found"
        text = "No objects selected!!!"
        font_color = (0, 0, 255)
        font_scale = 0.7
        cv2.putText(frame, text, (10, 30), font, font_scale, font_color, thickness)
        return

    text = f"DRONE: {position} - ID: {tracked_objects[position]}"
    font_scale = 0.9
    cv2.putText(frame, text, (x, y), font, font_scale, font_color, thickness)

    # Frame display
    cv2.rectangle(frame, (x - 10, y - 30), (x + 280, y + 10), (255, 255, 0), thickness)