#!/usr/bin/python3

# In the name of God
# Project: Face Tracker
# Author: Pooya Shams kolahi

"""
This project is called face tracker, the program
basically tracks your face and according to the
difference of the position of nose tip and the
center of the frame captured by the camera, it
tries to understand if you are looking down or up
and then moves the text that you are reading or
editing, correspondingly.
"""


from imutils.video import VideoStream
from imutils import face_utils
from pynput.keyboard import Key
from pynput.mouse import Button
import numpy as np
import time
import threading
import json
import os
import imutils
import pynput
import dlib
import cv2

# loading the facial landmark data file

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
print("[INFO] camera sensor warming up...")

# loading video stream

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2)

# initializing primary variables
width = 400
frame = vs.read()
h, w, _ = frame.shape
h = h * width // w
w = width
xc = w // 2  # x center
yc = h // 2  # y center
default_settings_file = "default_settings.json"
settings_file = "settings.json"

# loading json settings
data = {
    "scroll_coefficient": 10,
    "y_inv": -1,
    "vscode_mode": True,
    "visual_mode": False,
    "accurate": True,
    "running": False,
    "no_effect_area_up": 0.5,
    "no_effect_area_down": -0.5,
    "y_scroll_thresh_up": 3,
    "y_scroll_thresh_down": -3,
    "shortcuts": {
        "toggle_vscode_mode": ["v", ["cmd", "ctrl"]],
        "toggle_visual_mode": ["e", ["cmd", "ctrl"]],
        "toggle_accurate_mode": ["a", ["cmd", "ctrl"]],
        "toggle_running_mode": ["s", ["cmd", "ctrl"]],
        "quit_program": ["q", ["cmd", "ctrl"]],
    }
}

# settings
# scroll_coefficient: defines the speed of scroll
# y_inv: defines if y ordinate will be inverted or not
# vscode_mode: defines if the scrolling will be in vs code way or not.
#   if set to true, it uses ctrl+[up/down] to move up/down.
#   if set to false, it uses mouse scrolling
#   (click 4 for up and click 5 for down)
# visual_mode: defines if your camera input will be shown on the screen ot not.
# accurate: defines if the program will be using face landmark data
#   to find nose tip or just using the face rect.
#   the face landmark data is more accurate.
# no_effect_area_[up/down]:
#   define the area in which the program will ignore your nose tip if seen.
# y_scroll_thresh_[up/down]: defines the amount of time (relatively)
#   needed for your face to stay up or down to move one line up or down.
#   (actually it has opposite relation to speed of scrolling)

# shortcuts:
# shortcuts are in the following format
# "function name": ["main key", ["dependency", "keys"]]
# if you want to set more than one shortcut that uses
# a main key, you should order them in a way that if the
# first one is called, the ones after it wont be called
# but if the last one is called means that the ones before
# it couldn't be called.
# overall the above statement means that place the shortcuts
# with more dependency keys at first and the shortcuts with
# less dependency keys at last.
# I recommend reading the shortcuts_dict creation segment
# (it's a for loop placed after first declaration of the
#  shortcuts_dict variable)
# and the on_release function to understand how the program
# handles the shortcuts

# checking for the files to open
if os.path.isfile(settings_file):
    try:
        data = json.load(open(settings_file))
    except Exception:
        pass

elif os.path.isfile(default_settings_file):
    try:
        data = json.load(open(default_settings_file))
    except Exception:
        pass

# setting the rest of the variables
scroll_coefficient = data["scroll_coefficient"]
no_effect_area_up = data["no_effect_area_up"]
no_effect_area_down = data["no_effect_area_down"]
y_inv = data["y_inv"]  # 1 does nothing, -1 inverts the y scroll direction
y_scroll_counter = 0
y_scroll_thresh_up = data["y_scroll_thresh_up"]
y_scroll_thresh_down = data["y_scroll_thresh_down"]
clicked_after_scroll = True
vscode_mode = data["vscode_mode"]
visual_mode = data["visual_mode"]
shortcuts = data["shortcuts"]
accurate = data["accurate"]
running = data["running"]

# key map (dictionary)
# used to monitor pressed and released keys
# the 'keys' will be strings of keyboard's keys and their
# value will be boolean (True: pressed, False released)
key_map = {}

# key dicts
# special keys
# used to distinguish pynput special keys and normal 'characters'
special_keys = {
    Key.alt: 'alt',
    Key.alt_gr: 'alt_gr',
    Key.alt_r: 'alt_r',
    Key.backspace: 'backspace',
    Key.caps_lock: 'caps_lock',
    Key.cmd: 'cmd',
    Key.cmd_r: 'cmd_r',
    Key.ctrl: 'ctrl',
    Key.ctrl_r: 'ctrl_r',
    Key.delete: 'delete',
    Key.down: 'down',
    Key.end: 'end',
    Key.enter: 'enter',
    Key.esc: 'esc',
    Key.f1: 'f1',
    Key.f10: 'f10',
    Key.f11: 'f11',
    Key.f12: 'f12',
    Key.f13: 'f13',
    Key.f14: 'f14',
    Key.f15: 'f15',
    Key.f16: 'f16',
    Key.f17: 'f17',
    Key.f18: 'f18',
    Key.f19: 'f19',
    Key.f2: 'f2',
    Key.f20: 'f20',
    Key.f3: 'f3',
    Key.f4: 'f4',
    Key.f5: 'f5',
    Key.f6: 'f6',
    Key.f7: 'f7',
    Key.f8: 'f8',
    Key.f9: 'f9',
    Key.home: 'home',
    Key.insert: 'insert',
    Key.left: 'left',
    Key.media_play_pause: 'media_play_pause',
    Key.menu: 'menu',
    Key.num_lock: 'num_lock',
    Key.page_down: 'page_down',
    Key.page_up: 'page_up',
    Key.pause: 'pause',
    Key.print_screen: 'print_screen',
    Key.right: 'right',
    Key.scroll_lock: 'scroll_lock',
    Key.shift: 'shift',
    Key.shift_r: 'shift_r',
    Key.space: 'space',
    Key.tab: 'tab',
    Key.up: 'up',
}

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# setting keyboard and mouse controller
# these will be used later to scroll
keyboard = pynput.keyboard.Controller()
mouse = pynput.mouse.Controller()


# primary functions

def check_pressed(key):
    if key in key_map and key_map[key] is True:
        return True
    return False


def ctrl_up():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.up)
    keyboard.release(Key.ctrl)
    keyboard.release(Key.up)


def ctrl_down():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.down)
    keyboard.release(Key.ctrl)
    keyboard.release(Key.down)


def scroll_up():
    mouse.scroll(0, 1)


def scroll_down():
    mouse.scroll(0, -1)


def go_up():
    if vscode_mode:
        ctrl_up()
    else:
        scroll_up()


def go_down():
    if vscode_mode:
        ctrl_down()
    else:
        scroll_down()


# shortcut functions


def toggle_vscode_mode():
    global vscode_mode
    vscode_mode = not vscode_mode


def toggle_visual_mode():
    global visual_mode
    visual_mode = not visual_mode


def toggle_accurate_mode():
    global accurate
    accurate = not accurate


def toggle_running_mode():
    global running
    running = not running


def quit_program():
    exit()


# function names
# a dictionary which will be used to transform
# string of function's name to a real function object


function_names = {
    "toggle_vscode_mode": toggle_vscode_mode,
    "toggle_visual_mode": toggle_visual_mode,
    "toggle_accurate_mode": toggle_accurate_mode,
    "toggle_running_mode": toggle_running_mode,
    "quit_program": quit_program,
}

# shortcuts class


class shortcut:
    def __init__(self, main_key: str, dependency_keys: list, function: str):
        """
        arguments:
        main key:
            it's the main key that will trigger the function
            it's mostly a one character string
        dependency keys:
            these are the keys that should be pressed so that
            the function will be called, otherwise the main key
            being pressed does nothing
        it's a list of strings from the list of 'special_keys' values
        function:
            name of the function that will be called in string format
        """
        self.main_key = main_key
        # same as document
        self.dependency_keys = dependency_keys
        # same as document
        self.function = function_names[function]
        # the function that will be called(itself not the string name)

    def check(self, pressed_key):
        # checks if the dependency keys are clicked
        # and the pressed key is the main key
        if (all(map(check_pressed, self.dependency_keys))
            and pressed_key not in special_keys
                and pressed_key == self.main_key):
            return True
        return False

    def call_function(self, pressed_key):
        # will call the function if the keys are properly pressed
        if self.check(pressed_key):
            self.function()


# creating shortcuts dictionary
# will be used to call each shortcut when pressed
shortcuts_dict = {}
for sh in shortcuts:
    main_key = shortcuts[sh][0]
    dep_keys = shortcuts[sh][1]
    tmp_shortcut = shortcut(main_key, dep_keys, sh)
    if main_key in shortcuts_dict:
        if type(shortcuts_dict[main_key]) == list:
            shortcuts_dict[main_key].append(tmp_shortcut)
        else:  # if type(shortcuts_dict[main_key]) == shortcut
            shortcuts_dict[main_key] = [shortcuts_dict[main_key], tmp_shortcut]
    else:
        shortcuts_dict[main_key] = tmp_shortcut


# event functions
# will be used to check and log pressed keys

def convert_key_name(key):
    dict_key = None
    if key in special_keys:
        dict_key = special_keys[key]
    else:
        if hasattr(key, 'vk') and hasattr(key, 'char'):
            char = key.char
            vk = key.vk
            if char in nums and vk is None:
                char = '[' + char + ']'
            dict_key = char
    return dict_key


def set_key_in_key_map(key, amount):
    dict_key = convert_key_name(key)  # will be used to set a value in key_map
    key_map[dict_key] = amount


def on_press(key):
    set_key_in_key_map(key, True)


def on_release(key):
    nkey = convert_key_name(key)
    if nkey in shortcuts_dict:
        tmp = shortcuts_dict[nkey]
        if type(tmp) == list:
            for _sh in tmp:
                if _sh.check(nkey):
                    _sh.call_function(nkey)
                    break
        else:
            tmp.call_function(nkey)
    set_key_in_key_map(key, False)


def listen_funtion():
    with pynput.keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as keyboard_listener:
        keyboard_listener.join()


listen_obj = threading.Thread(target=listen_funtion)
listen_obj.start()


while True:
    # reading and preprocessing frame
    frame = vs.read()
    frame = np.flip(frame, axis=1)
    frame = imutils.resize(frame, width=width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the faces rects
    rects = detector(gray, 0)

    # checking the number of found faces and
    # if there was just one face, using that
    # face to determine inputs
    len_rects = len(rects)
    if len_rects == 1 and running:
        rect = rects[0]
        if accurate:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            x, y = shape[4]
        else:
            p = rect.center()
            x, y = p.x, p.y
            # using the center of the rect instead of finding the nose
            # less accurate
        y_dir = y - yc
        y_scroll = y_dir / h * scroll_coefficient * y_inv
        if y_scroll > no_effect_area_up:
            y_scroll_counter += y_scroll
            if y_scroll_counter >= y_scroll_thresh_up:
                go_up()
                y_scroll_counter = 0
            clicked_after_scroll = False
        elif y_scroll < no_effect_area_down:
            y_scroll_counter += y_scroll
            if y_scroll_counter <= y_scroll_thresh_down:
                go_down()
                y_scroll_counter = 0
            clicked_after_scroll = False
        elif not clicked_after_scroll:
            mouse.click(Button.left)
            clicked_after_scroll = True

        if visual_mode:
            cv2.line(frame, (xc, yc), (xc, yc + y_dir), (0, 255, 0))
            bX, bY, bW, bH = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                          (0, 255, 0), 1)

    # checking if there was any faces found to write
    # the number of faces found on the screen
    # this will happen if visual mode is enabled
    if visual_mode:
        if len_rects > 0:
            text = f"{len_rects} face(s) found"
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)

    # resting cpu for 10ms and getting pressed key
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
listen_obj.join()
