# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import mmap
import os
import sys
import select
import termios
import tty

from utils.constants import (
    GRAY,
    GREEN,
    WHITE,
    FONT,
    FONT_THICKNESS,
    LOADING_TEXT,
    FONT_SCALE,
    INTERPOLATION_METHOD,
    NAV_ZONE_INFO_OFFSET_Y,
)


def display_to_framebuffer(image_np, fb_device_path="/dev/fb0"):
    """Display OpenCV image directly to Linux framebuffer"""
    fbN = os.path.basename(fb_device_path)

    try:
        with open(f"/sys/class/graphics/{fbN}/virtual_size", "r") as f:
            w_str, h_str = f.read().strip().split(",")
            w = int(w_str)
            h = int(h_str)
        with open(f"/sys/class/graphics/{fbN}/bits_per_pixel", "r") as f:
            bpp = int(f.read().strip())
        bytes_per_pixel = bpp // 8
        screen_size_bytes = w * h * bytes_per_pixel
    except Exception as e:
        print(f"Error reading framebuffer info: {e}")
        return False

    if image_np.shape[0] != h or image_np.shape[1] != w:
        image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_AREA)

    if bpp == 16:
        B = image_np[:, :, 0].astype(np.uint32)
        G = image_np[:, :, 1].astype(np.uint32)
        R = image_np[:, :, 2].astype(np.uint32)
        
        packed_image = ((R >> 3) << 11) | ((G >> 2) << 5) | (B >> 3)
        framebuffer_data = packed_image.astype(np.uint16)
    else:
        print(f"Error: Unsupported bpp: {bpp}")
        return False

    try:
        with open(fb_device_path, mode="r+b") as fbdev:
            with mmap.mmap(fbdev.fileno(), screen_size_bytes, mmap.MAP_SHARED, mmap.PROT_WRITE) as fb_mmap:
                fb_mmap.write(framebuffer_data.tobytes())
        return True
    except Exception as e:
        print(f"Error accessing framebuffer: {e}")
        return False


class UIManager:
    def __init__(self, window_width, window_height, key_mapping):
        os.system("clear")
        os.system("setterm -cursor off")
        self.window_width = window_width
        self.window_height = window_height
        self.key_mapping = key_mapping
        self.action = None
        self.action_lock = threading.Lock()
        self.fb_device = "/dev/fb0"
        self.visualization_window = "framebuffer"

        self.show_main_instructions()

    def mouse_callback(self, event, x, y, flags, param):
        """Placeholder for mouse callback (not used in framebuffer mode)"""
        pass

    def show_main_instructions(self):
        """Display main control instructions"""
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        button_width = 500
        button_height = 100
        button_x = (self.window_width - button_width) // 2
        button_y = (self.window_height - button_height) // 2

        # Button background
        cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), GRAY, -1)
        # Button border
        cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height), GREEN, 2)

        # Button text
        text = "Start New Inspection"
        text_size = cv2.getTextSize(text, FONT, 1, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), FONT, 1, GREEN, 2, cv2.LINE_AA)

        display_to_framebuffer(img, self.fb_device)

    def show_loading_screen(self):
        """Display loading screen"""
        loading_screen = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(LOADING_TEXT, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (self.window_width - text_size[0]) // 2
        text_y = (self.window_height + text_size[1]) // 2
        cv2.putText(loading_screen, LOADING_TEXT, (text_x, text_y), FONT, FONT_SCALE, WHITE, FONT_THICKNESS)
        display_to_framebuffer(loading_screen, self.fb_device)

    def hide_loading_screen(self):
        """Hide loading screen - just clear to black"""
        black_screen = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        display_to_framebuffer(black_screen, self.fb_device)

    def show_zone_visualization(self, zone_num, zone_img, total_zones):
        """Display zone visualization"""
        img_height, img_width = zone_img.shape[:2]
        scale = min(self.window_width / img_width, self.window_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_img = cv2.resize(zone_img, (new_width, new_height), interpolation=INTERPOLATION_METHOD)
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        x_offset = (self.window_width - new_width) // 2
        y_offset = (self.window_height - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        # Zone info
        zone_info = f"Zone {zone_num}/{total_zones}"
        zone_info_size = cv2.getTextSize(zone_info, FONT, 1, 2)[0]
        zone_info_x = (self.window_width - zone_info_size[0]) // 2
        cv2.putText(canvas, zone_info, (zone_info_x, NAV_ZONE_INFO_OFFSET_Y + zone_info_size[1]), FONT, 1, WHITE, 2)

        display_to_framebuffer(canvas, self.fb_device)

    def wait_for_action(self, timeout=50):
        """Wait for keyboard input (console-based)"""
        with self.action_lock:
            if self.action:
                action_to_return = self.action
                self.action = None
                return action_to_return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], timeout / 1000.0)[0]:
                key = sys.stdin.read(1)
                if key == '\r' or key == '\n': return "start"
                if key == ' ': return "finish"
                if key == '\x1b':
                    # Increased timeout and added up/down arrow support
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        next1 = sys.stdin.read(1)
                        if next1 == '[':
                            if select.select([sys.stdin], [], [], 0.05)[0]:
                                next2 = sys.stdin.read(1)
                                if next2 in ('B', 'C'):  # Down or Right
                                    return "next"
                                if next2 in ('A', 'D'):  # Up or Left
                                    return "previous"
                    return "exit"
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return None

    def hide_instruction_window(self):
        """Clear the framebuffer"""
        black_screen = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        display_to_framebuffer(black_screen, self.fb_device)

    def cleanup(self):
        """Clean up UI resources"""
        os.system("setterm -cursor on")
        black_screen = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        display_to_framebuffer(black_screen, self.fb_device)
