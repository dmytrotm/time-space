import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf
from utils.find_keyboard import find_keyboard_by_name

try:
    from evdev import ecodes
except ImportError:

    class ecodes:
        EV_KEY = 1
        KEY_1 = 2
        KEY_2 = 3
        KEY_7 = 4
        KEY_R = 5

import time
import threading
from queue import Queue
import cv2


import os
from datetime import datetime
# State constants
STATE_START = 0
STATE_LOADING = 1
STATE_ERROR = 2
STATE_SUCCESS = 3


class UIManager:
    def __init__(
        self, verification_manager, image_server, window_width=800, window_height=480, resource_monitor=None
    ):
        """
        Args:
            verification_manager: Instance of VerificationManager
            image_server: Instance of ImageServer
            resource_monitor: Optional instance of ResourceMonitor
        """
        self.verification_manager = verification_manager
        self.image_server = image_server
        self.resource_monitor = resource_monitor
        self.WIDTH = window_width
        self.HEIGHT = window_height
        self.is_running_verification = False

    def render_text(self, renderer, font, message, color=(255, 255, 255)):
        """Render text and return texture with dimensions."""
        text_surface = sdlttf.TTF_RenderUTF8_Blended(
            font, message.encode("utf-8"), sdl2.SDL_Color(color[0], color[1], color[2])
        )
        texture = sdl2.SDL_CreateTextureFromSurface(renderer, text_surface)
        w = text_surface.contents.w
        h = text_surface.contents.h
        sdl2.SDL_FreeSurface(text_surface)
        return texture, w, h

    def render_centered_text(
        self, renderer, font, message, y_pos, color=(255, 255, 255)
    ):
        """Render text centered horizontally at given y position."""
        tex, w, h = self.render_text(renderer, font, message, color)
        x_pos = (self.WIDTH - w) // 2
        dst = sdl2.SDL_Rect(x_pos, y_pos, w, h)
        sdl2.SDL_RenderCopy(renderer, tex, None, dst)
        sdl2.SDL_DestroyTexture(tex)
        return h

    def render_start_screen(self, renderer, font_large, font_small):
        """Render the start screen."""
        self.render_centered_text(
            renderer, font_large, "TIME&SPACE", self.HEIGHT // 3, (255, 255, 255)
        )

        self.render_centered_text(
            renderer,
            font_small,
            "Press KEY_1 to start",
            self.HEIGHT // 2,
            (200, 200, 200),
        )
        
        # Show resource monitor status
        if self.resource_monitor:
            status = "ON" if self.resource_monitor.is_monitoring else "OFF"
            color = (50, 255, 50) if self.resource_monitor.is_monitoring else (255, 50, 50)
            self.render_centered_text(
                renderer,
                font_small,
                f"Resource Monitor: {status} (Press KEY_R to toggle)",
                self.HEIGHT // 2 + 40,
                color,
            )

    def render_loading_screen(self, renderer, font_large, font_small):
        """Render the loading screen."""
        self.render_centered_text(
            renderer, font_large, "Loading...", self.HEIGHT // 3, (255, 255, 0)
        )

        self.render_centered_text(
            renderer, font_small, "Please wait...", self.HEIGHT // 2, (200, 200, 200)
        )

    def render_error_screen(self, renderer, font_large, font_small, error_msg):
        """Render the error screen."""
        self.render_centered_text(
            renderer, font_large, error_msg, self.HEIGHT // 2 - 20, (255, 100, 100)
        )

        self.render_centered_text(
            renderer,
            font_small,
            "Press KEY_1 to retry",
            self.HEIGHT // 2 + 40,
            (200, 200, 200),
        )
        self.render_centered_text(
            renderer,
            font_small,
            "Press KEY_2 to return to main",
            self.HEIGHT // 2 + 80,
            (150, 150, 150),
        )

    def render_success_screen(self, renderer, font_large, font_small):
        """Render the success screen."""
        self.render_centered_text(
            renderer, font_large, "SUCCESS!", self.HEIGHT // 3, (50, 255, 50)
        )

        self.render_centered_text(
            renderer,
            font_small,
            "Inspection completed successfully",
            self.HEIGHT // 2,
            (100, 255, 100),
        )

        self.render_centered_text(
            renderer,
            font_small,
            "Press KEY_2 to return to main",
            self.HEIGHT // 2 + 100,
            (150, 150, 150),
        )

    def toggle_resource_monitor(self):
        """Toggle resource monitoring on/off."""
        if self.resource_monitor:
            if self.resource_monitor.is_monitoring:
                self.resource_monitor.stop_monitoring()
                print("Resource monitoring paused")
            else:
                self.resource_monitor.start_monitoring()
                print("Resource monitoring resumed")
        else:
            print("Resource monitor not available")

    def start_verification(self):
        """Start the verification process."""
        if not self.is_running_verification:
            self.is_running_verification = True

            try:
                #TODO move this operation to the worker
                images = self.image_server.take_photos()

                if not images or len(images) != 2:
                    raise ValueError("There is not images captured")
                
                self.verification_manager.trigger_verification(images)
                return True
            except Exception as e:
                print(f"Error starting verification: {e}")
                self.is_running_verification = False
                return False
        return False


    def main_loop(self):
        """Main UI loop."""
        sdl2.ext.init()
        sdlttf.TTF_Init()

        font_large = sdlttf.TTF_OpenFont(
            b"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
        )
        
        font_small = sdlttf.TTF_OpenFont(
            b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
        if not font_large or not font_small:
            raise RuntimeError("Fonts are not loaded, please check the paths")
        window = sdl2.ext.Window("TIME&SPACE", size=(self.WIDTH, self.HEIGHT))
        window.show()
        sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
        renderer = sdl2.SDL_CreateRenderer(window.window, -1, 0)

        keys = Queue()

        def keyboard_thread():
            while True:
                kbd = find_keyboard_by_name()
                
                if kbd is None:
                    time_to_sleep = 2
                    print(f"Keyboard is not found, another try in {time_to_sleep} second")
                    time.sleep(time_to_sleep)
                    continue  
                    
                print(f"Keyboard is found")
                
                try:
                    for e in kbd.read_loop():
                        if e.type == ecodes.EV_KEY:
                            if e.value == 0:  
                                keys.put(e.code)
                                
                except (IOError, OSError) as e:
                    print(f"Keyboard is out. Trying to reconnect...")
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Uknown error: {e}")
                    time.sleep(1)

        threading.Thread(target=keyboard_thread, daemon=True).start()

        current_state = STATE_START
        error_message = ""

        running = True
        TARGET_FPS = 30
        FRAME_DELAY = int(1000 / TARGET_FPS)

        while running:
            frame_start = sdl2.SDL_GetTicks()
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    running = False
            result = self.verification_manager.check_results()
            if result:
                self.is_running_verification = False
                if result["success"]:
                    current_state = STATE_SUCCESS
                else:
                    current_state = STATE_ERROR
                    error_message = result["error"]
                    if not os.path.exists("log"):
                        os.makedirs("log")
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    log_dir = os.path.join("log", timestamp)
                    os.makedirs(log_dir, exist_ok=True)
                    if "error_images" in result:
                        images = result["error_images"]
                    
                        for key, value in images.items():
                            print(f"{key}")
                            file_path = os.path.join(log_dir, f"{key}.png")
                            cv2.imwrite(file_path, value)

            if not keys.empty():
                key = keys.get()

                if key == ecodes.KEY_7:
                    running = False

                elif key == ecodes.KEY_1:
                    if not self.is_running_verification:
                        if current_state == STATE_START or current_state == STATE_ERROR:
                            current_state = STATE_LOADING
                            if not self.start_verification():
                                current_state = STATE_ERROR
                                error_message = "Camera Error"
                        elif current_state == STATE_SUCCESS:
                            current_state = STATE_START
                elif key == ecodes.KEY_2:
                    if not self.is_running_verification:
                        if (
                            current_state == STATE_ERROR
                            or current_state == STATE_SUCCESS
                        ):
                            current_state = STATE_START
                elif key == ecodes.KEY_R:
                    self.toggle_resource_monitor()

            sdl2.SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255)
            sdl2.SDL_RenderClear(renderer)

            if current_state == STATE_START:
                self.render_start_screen(renderer, font_large, font_small)

            elif current_state == STATE_LOADING:
                self.render_loading_screen(renderer, font_large, font_small)

            elif current_state == STATE_ERROR:
                self.render_error_screen(
                    renderer, font_large, font_small, error_message
                )

            elif current_state == STATE_SUCCESS:
                self.render_success_screen(renderer, font_large, font_small)

            sdl2.SDL_RenderPresent(renderer)

            frame_time = sdl2.SDL_GetTicks() - frame_start
            if frame_time < FRAME_DELAY:
                sdl2.SDL_Delay(FRAME_DELAY - frame_time)

        sdlttf.TTF_CloseFont(font_large)
        sdlttf.TTF_CloseFont(font_small)
        sdlttf.TTF_Quit()
        sdl2.SDL_DestroyRenderer(renderer)
        sdl2.ext.quit()
