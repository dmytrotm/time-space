import cv2
import numpy as np
import threading
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
    GRAY,
    WHITE,
    FONT,
    FONT_SCALE,
    FONT_THICKNESS,
    NAV_ZONE_INFO_OFFSET_Y,
    INTERPOLATION_METHOD,
)


class UIManager:
    def __init__(self, window_width, window_height, key_mapping):
        self.window_width = window_width
        self.window_height = window_height
        self.key_mapping = key_mapping
        self.action = None
        self.buttons = []
        self.action_lock = threading.Lock()
        self.instruction_window = "Controls"
        self.visualization_window = "Zone Inspection Results"
        self.window_created = False

        self._create_window()
        self.show_main_instructions()

    def _create_window(self):
        """Create or recreate the instruction window"""
        if not self.window_created:
            cv2.namedWindow(self.instruction_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.instruction_window, self.window_width, self.window_height
            )
            self.window_created = True
        cv2.setMouseCallback(self.instruction_window, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.action_lock:
                for x_b, y_b, w_b, h_b, action in self.buttons:
                    if x_b <= x <= x_b + w_b and y_b <= y <= y_b + h_b:
                        self.action = action
                        break

    def show_main_instructions(self):
        """Display main control instructions with centered buttons"""
        self._create_window()

        self.buttons = []
        img = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        content_x = (self.window_width - CONTENT_WIDTH) // 2
        content_y = (self.window_height - CONTENT_HEIGHT) // 2

        cv2.rectangle(
            img,
            (content_x, content_y),
            (content_x + CONTENT_WIDTH, content_y + CONTENT_HEIGHT),
            DARK_GRAY,
            -1,
        )
        cv2.rectangle(
            img,
            (content_x, content_y),
            (content_x + CONTENT_WIDTH, content_y + CONTENT_HEIGHT),
            YELLOW,
            FONT_THICKNESS // 2,
        )

        center_x = self.window_width // 2
        start_y = content_y + TITLE_OFFSET_Y

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

        line_y = start_y + TITLE_LINE_OFFSET
        cv2.line(
            img,
            (content_x + CONTENT_PADDING, line_y),
            (content_x + CONTENT_WIDTH - CONTENT_PADDING, line_y),
            LIGHTER_GRAY,
            FONT_THICKNESS // 2,
        )

        instructions = [
            ("ENTER", "Start new inspection", "start", GREEN),
            ("ESC", "Exit program", "exit", ORANGE),
        ]

        y_pos = start_y + BUTTON_START_OFFSET

        for idx, (key_text, desc_text, action, color) in enumerate(instructions):
            button_x = center_x - BUTTON_WIDTH // 2
            button_y_top = y_pos - BUTTON_HEIGHT + BUTTON_OFFSET_Y
            button_y_bottom = y_pos + BUTTON_OFFSET_Y

            cv2.rectangle(
                img,
                (button_x, button_y_top),
                (button_x + BUTTON_WIDTH, button_y_bottom),
                GRAY,
                -1,
            )
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

            if idx < len(instructions) - 1:
                divider_y = y_pos - LINE_HEIGHT + BUTTON_HEIGHT + 20
                cv2.line(
                    img,
                    (center_x - DIVIDER_WIDTH, divider_y),
                    (center_x + DIVIDER_WIDTH, divider_y),
                    LIGHTER_GRAY,
                    FONT_THICKNESS // 2,
                )

        cv2.imshow(self.instruction_window, img)

    def show_loading_screen(self):
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
        cv2.imshow("Loading", loading_screen)
        cv2.waitKey(1)

    def hide_loading_screen(self):
        cv2.destroyWindow("Loading")

    def show_zone_visualization(self, zone_num, zone_img, total_zones):
        cv2.namedWindow(self.visualization_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.visualization_window, self.window_width, self.window_height
        )
        cv2.setMouseCallback(self.visualization_window, self.mouse_callback)

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

        cv2.imshow(self.visualization_window, canvas)

    def wait_for_action(self, timeout=50):
        key = cv2.waitKey(timeout)

        with self.action_lock:
            if self.action:
                action = self.action
                self.action = None
                return action

        if key != -1:
            key &= 0xFF
            if key == 255:
                return None
            return self.key_mapping.get(key)

        return None

    def hide_instruction_window(self):
        cv2.destroyWindow(self.instruction_window)
        self.window_created = False
