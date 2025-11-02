# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import mmap
import os

from utils.constants import (
    CONTENT_WIDTH,
    CONTENT_HEIGHT,
    TITLE_OFFSET_Y,
    LINE_HEIGHT,
    BUTTON_HEIGHT,
    BUTTON_WIDTH,
    BUTTON_OFFSET_Y,
    TITLE_LINE_OFFSET,
    BUTTON_START_OFFSET,
    CONTENT_PADDING,
    DIVIDER_WIDTH,
    DARK_GRAY,
    YELLOW,
    LIGHTER_GRAY,
    GRAY,
    GREEN,
    ORANGE,
    FONT,
    TITLE_FONT_SCALE,
    FONT_THICKNESS,
    LOADING_TEXT,
    NAV_BUTTON_WIDTH,
    WHITE,
    FONT_SCALE,
    NAV_ZONE_INFO_OFFSET_Y,
    INTERPOLATION_METHOD,
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

    # Resize image to fit screen
    if image_np.shape[0] != h or image_np.shape[1] != w:
        image_np = cv2.resize(image_np, (w, h), interpolation=cv2.INTER_AREA)

    if bpp == 16:
        # RGB565 format
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        R = image_rgb[:, :, 0]
        G = image_rgb[:, :, 1]
        B = image_rgb[:, :, 2]
        packed_image = ((R >> 3) << 11) | ((G >> 2) << 5) | (B >> 3)
        framebuffer_data = packed_image.astype(np.uint16)
    elif bpp == 24 or bpp == 32:
        if bpp == 32 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2BGRA)
        elif bpp == 24 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        framebuffer_data = image_np
    else:
        print(f"Error: Unsupported bpp: {bpp}")
        return False

    try:
        fbdev = open(fb_device_path, mode="r+b")
        fb_mmap = mmap.mmap(
            fbdev.fileno(), screen_size_bytes, mmap.MAP_SHARED, mmap.PROT_WRITE
        )

        if bpp == 16:
            fb_mmap.write(framebuffer_data.tobytes())
        else:
            # Write directly to avoid holding references
            fb_mmap.seek(0)
            fb_mmap.write(framebuffer_data.tobytes())

        fb_mmap.close()
        fbdev.close()
        return True
    except Exception as e:
        print(f"Error accessing framebuffer: {e}")
        return False


class UIManager:
    def __init__(self, window_width, window_height, key_mapping):
        self.window_width = window_width
        self.window_height = window_height
        self.key_mapping = key_mapping
        self.action = None
        self.buttons = []
        self.action_lock = threading.Lock()
        self.fb_device = "/dev/fb0"
        self.visualization_window = "framebuffer"  # Compatibility attribute

        print("UIManager initialized for framebuffer display")
        self.show_main_instructions()

    def mouse_callback(self, event, x, y, flags, param):
        """Placeholder for mouse callback (not used in framebuffer mode)"""
        pass

    def show_main_instructions(self):
        """Display main control instructions"""
        self.buttons = []
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        content_x = (self.window_width - CONTENT_WIDTH) // 2
        content_y = (self.window_height - CONTENT_HEIGHT) // 2

        # Background
        cv2.rectangle(
            img,
            (content_x, content_y),
            (content_x + CONTENT_WIDTH, content_y + CONTENT_HEIGHT),
            DARK_GRAY,
            -1,
        )
        # Border
        cv2.rectangle(
            img,
            (content_x, content_y),
            (content_x + CONTENT_WIDTH, content_y + CONTENT_HEIGHT),
            YELLOW,
            FONT_THICKNESS // 2,
        )

        center_x = self.window_width // 2
        start_y = content_y + TITLE_OFFSET_Y

        # Title
        title = "INSPECTION CONTROLS"
        title_size = cv2.getTextSize(title, FONT, TITLE_FONT_SCALE, FONT_THICKNESS)[0]
        title_x = center_x - title_size[0] // 2
        cv2.putText(
            img,
            title,
            (title_x, start_y),
            FONT,
            TITLE_FONT_SCALE,
            YELLOW,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

        # Line under title
        line_y = start_y + TITLE_LINE_OFFSET
        cv2.line(
            img,
            (content_x + CONTENT_PADDING, line_y),
            (content_x + CONTENT_WIDTH - CONTENT_PADDING, line_y),
            LIGHTER_GRAY,
            FONT_THICKNESS // 2,
        )

        # Buttons
        instructions = [
            ("ENTER", "Start new inspection", "start", GREEN),
            ("ESC", "Exit program", "exit", ORANGE),
        ]

        y_pos = start_y + BUTTON_START_OFFSET

        for idx, (key_text, desc_text, action, color) in enumerate(instructions):
            button_x = center_x - BUTTON_WIDTH // 2
            button_y_top = y_pos - BUTTON_HEIGHT + BUTTON_OFFSET_Y
            button_y_bottom = y_pos + BUTTON_OFFSET_Y

            # Button background
            cv2.rectangle(
                img,
                (button_x, button_y_top),
                (button_x + BUTTON_WIDTH, button_y_bottom),
                GRAY,
                -1,
            )
            # Button border
            cv2.rectangle(
                img,
                (button_x, button_y_top),
                (button_x + BUTTON_WIDTH, button_y_bottom),
                color,
                FONT_THICKNESS // 2,
            )

            self.buttons.append(
                (button_x, button_y_top, BUTTON_WIDTH, BUTTON_HEIGHT, action)
            )

            # Button text
            full_text = f"{desc_text}"
            text_size = cv2.getTextSize(
                full_text, FONT, FONT_SCALE / 2, FONT_THICKNESS // 2
            )[0]
            text_x = center_x - text_size[0] // 2
            text_y = button_y_top + (BUTTON_HEIGHT + text_size[1]) // 2
            cv2.putText(
                img,
                full_text,
                (text_x, text_y),
                FONT,
                FONT_SCALE / 2,
                color,
                FONT_THICKNESS // 2,
                cv2.LINE_AA,
            )

            y_pos += LINE_HEIGHT

            # Divider
            if idx < len(instructions) - 1:
                divider_y = y_pos - LINE_HEIGHT + BUTTON_HEIGHT + 20
                cv2.line(
                    img,
                    (center_x - DIVIDER_WIDTH, divider_y),
                    (center_x + DIVIDER_WIDTH, divider_y),
                    LIGHTER_GRAY,
                    FONT_THICKNESS // 2,
                )

        # Display to framebuffer
        display_to_framebuffer(img, self.fb_device)
        print("Main instructions displayed on framebuffer")

    def show_loading_screen(self):
        """Display loading screen"""
        loading_screen = np.zeros(
            (self.window_height, self.window_width, 3), dtype=np.uint8
        )

        text_size = cv2.getTextSize(LOADING_TEXT, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (self.window_width - text_size[0]) // 2
        text_y = (self.window_height + text_size[1]) // 2

        cv2.putText(
            loading_screen,
            LOADING_TEXT,
            (text_x, text_y),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )

        display_to_framebuffer(loading_screen, self.fb_device)
        print("Loading screen displayed")

    def hide_loading_screen(self):
        """Hide loading screen - just clear to black"""
        black_screen = np.zeros(
            (self.window_height, self.window_width, 3), dtype=np.uint8
        )
        display_to_framebuffer(black_screen, self.fb_device)

    def show_zone_visualization(self, zone_num, zone_img, total_zones):
        """Display zone visualization"""
        available_width = self.window_width - 2 * NAV_BUTTON_WIDTH
        img_height, img_width = zone_img.shape[:2]

        scale = min(available_width / img_width, self.window_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_img = cv2.resize(
            zone_img, (new_width, new_height), interpolation=INTERPOLATION_METHOD
        )

        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        x_offset = NAV_BUTTON_WIDTH + (available_width - new_width) // 2
        y_offset = (self.window_height - new_height) // 2

        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized_img
        )

        # Previous button
        cv2.rectangle(canvas, (0, 0), (NAV_BUTTON_WIDTH, self.window_height), GRAY, -1)
        prev_text = "<"
        prev_text_size = cv2.getTextSize(prev_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        prev_text_x = (NAV_BUTTON_WIDTH - prev_text_size[0]) // 2
        prev_text_y = self.window_height // 2 + prev_text_size[1] // 2
        cv2.putText(
            canvas,
            prev_text,
            (prev_text_x, prev_text_y),
            FONT,
            FONT_SCALE,
            WHITE,
            FONT_THICKNESS,
        )

        # Next button
        cv2.rectangle(
            canvas,
            (self.window_width - NAV_BUTTON_WIDTH, 0),
            (self.window_width, self.window_height),
            GRAY,
            -1,
        )
        next_text = ">"
        next_text_size = cv2.getTextSize(next_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        next_text_x = (
            self.window_width
            - NAV_BUTTON_WIDTH
            + (NAV_BUTTON_WIDTH - next_text_size[0]) // 2
        )
        next_text_y = self.window_height // 2 + next_text_size[1] // 2
        cv2.putText(
            canvas,
            next_text,
            (next_text_x, next_text_y),
            FONT,
            FONT_SCALE,
            WHITE,
            FONT_THICKNESS,
        )

        # Zone info
        zone_info = f"Zone {zone_num}/{total_zones}"
        zone_info_size = cv2.getTextSize(zone_info, FONT, 0.8, 2)[0]
        zone_info_x = (self.window_width - zone_info_size[0]) // 2
        cv2.putText(
            canvas,
            zone_info,
            (zone_info_x, NAV_ZONE_INFO_OFFSET_Y + zone_info_size[1]),
            FONT,
            0.8,
            WHITE,
            2,
        )

        # Update buttons for navigation
        self.buttons = [
            (0, 0, NAV_BUTTON_WIDTH, self.window_height, "previous"),
            (
                self.window_width - NAV_BUTTON_WIDTH,
                0,
                NAV_BUTTON_WIDTH,
                self.window_height,
                "next",
            ),
        ]

        display_to_framebuffer(canvas, self.fb_device)
        print(f"Zone {zone_num}/{total_zones} displayed")

    def wait_for_action(self, timeout=50):
        """Wait for keyboard input (console-based)"""
        import sys
        import select
        import termios
        import tty

        # Check if action was set programmatically
        with self.action_lock:
            if self.action:
                action = self.action
                self.action = None
                return action

        # Non-blocking keyboard input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            # Check if input is available
            if select.select([sys.stdin], [], [], timeout / 1000.0)[0]:
                key = sys.stdin.read(1)

                # Map keys
                if key == "\r" or key == "\n":  # Enter
                    return "start"
                elif key == " ":  # Space
                    return "finish"
                elif key == "\x1b":  # ESC
                    # Check for arrow keys (ESC [ A/B/C/D)
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        next1 = sys.stdin.read(1)
                        if next1 == "[":
                            if select.select([sys.stdin], [], [], 0.01)[0]:
                                next2 = sys.stdin.read(1)
                                if next2 == "C":  # Right arrow
                                    return "next"
                                elif next2 == "D":  # Left arrow
                                    return "previous"
                    return "exit"

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        return None

    def hide_instruction_window(self):
        """Clear the framebuffer"""
        black_screen = np.zeros(
            (self.window_height, self.window_width, 3), dtype=np.uint8
        )
        display_to_framebuffer(black_screen, self.fb_device)
        print("Instruction window cleared")

    def cleanup(self):
        """Clean up UI resources"""
        # Clear framebuffer to black
        black_screen = np.zeros(
            (self.window_height, self.window_width, 3), dtype=np.uint8
        )
        display_to_framebuffer(black_screen, self.fb_device)
        print("UI cleanup completed")
