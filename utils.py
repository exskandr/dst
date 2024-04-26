import os
import cv2
import logging
import requests

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
load_dotenv()


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


def draw_center(frame):
    height, width, _ = frame.shape
    center_x = int(width / 2)
    center_y = int(height / 2)
    radius = 100
    center = (center_x, center_y)
    color = (0, 255, 0)
    cv2.circle(frame, center, radius, color, 2)
    x1, y1 = center_x-150, center_y
    x2, y2 = center_x+150, center_y
    line_thickness = 2
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
    x1, y1 = center_x, center_y-150
    x2, y2 = center_x, center_y+150
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)


# commands
login = os.getenv("LOGIN")
password = os.getenv("PASSWORD")
ip_cam = os.getenv("IP_CAM")
url_command = f'http://{ip_cam}/decoder_control.cgi?loginuse={login}&loginpas={password}'
cam_up = f'{url_command}&command=0&onestep=1&17024724560030.8794677227005614&_=170247245600'
cam_down = f'{url_command}&command=2&onestep=1&17024723729880.2857110046917418&_=1702472372988'
cam_left = f'{url_command}&command=6&onestep=1&17024707176350.2731615502645298&_=1702470717636'
cam_right = f'{url_command}&command=4&onestep=1&17024706046250.6925916266171585&_=1702470604625'


def cam_command_left():
    logging.info(f'turn left')
    # return requests.request("GET", url=cam_left)


def cam_command_right():
    logging.info(f'turn right')
    # return requests.request("GET", url=cam_right)


def cam_command_up():
    logging.info(f'turn up')
    # return requests.request("GET", url=cam_up)


def cam_command_down():
    logging.info(f'turn down')
    # return requests.request("GET", url=cam_down)


def move_cam(frame, detect):
    height, width, _ = frame.shape
    center_w = int(width / 2)  # X = 320
    center_h = int(height / 2)  # Y = 240
    for xyxy in zip(detect.xyxy):
        x1, y1, x2, y2 = xyxy[0]
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        # center_x = x1 + ((x2 - x1) / 2)
        # center_y = y1 + (y2 - y1) / 2
        # procent = 0.2
        if x1 < center_w < x2 and y1 < center_h < y2:
            #   0|0|0
            #   0|X|0
            #   0|0|0
            logging.info(f'object is in the center. \n')
            pass
        else:
            logging.info(f'object is not in the center\n'
                         f'{center_h}, {center_w} - ({x1}, {y1}),({x2}, {y2})')
            if y1 > center_h:
                #   X|X|X
                #   0|O|0
                #   0|0|0
                cam_command_down()
            if x2 < center_w and y1 < center_h < y2:
                #   0|0|0
                #   X|O|0
                #   0|0|0
                cam_command_right()
            if x1 > center_w and y1 < center_h < y2:
                #   0|0|0
                #   0|O|X
                #   0|0|0
                cam_command_left()
            if y2 < center_h:
                #   0|0|0
                #   0|O|0
                #   X|X|X
                cam_command_up()

