# -*- coding: utf-8 -*-
import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf
from find_keyboard import find_keyboard_by_name
from evdev import ecodes
import threading
from queue import Queue


# State constants
STATE_START = 0
STATE_LOADING = 1
STATE_ERROR = 2
STATE_SUCCESS = 3


class UIManager:
    def __init__(self, func, window_width=800, window_height=480):
        """
        func - function that returns (bool, str)
               bool: True if success, False if error
               str: error message (used only when bool is False)
        """
        self.func = func
        self.result_queue = Queue()
        self.WIDTH = window_width
        self.HEIGHT = window_height
        self.is_running_verification = False

    def render_text(self, renderer, font, message, color=(255, 255, 255)):
        """Render text and return texture with dimensions."""
        text_surface = sdlttf.TTF_RenderUTF8_Blended(
            font, message.encode('utf-8'),
            sdl2.SDL_Color(color[0], color[1], color[2])
        )
        texture = sdl2.SDL_CreateTextureFromSurface(renderer, text_surface)
        w = text_surface.contents.w
        h = text_surface.contents.h
        sdl2.SDL_FreeSurface(text_surface)
        return texture, w, h

    def render_centered_text(self, renderer, font, message, y_pos, color=(255, 255, 255)):
        """Render text centered horizontally at given y position."""
        tex, w, h = self.render_text(renderer, font, message, color)
        x_pos = (self.WIDTH - w) // 2
        dst = sdl2.SDL_Rect(x_pos, y_pos, w, h)
        sdl2.SDL_RenderCopy(renderer, tex, None, dst)
        sdl2.SDL_DestroyTexture(tex)
        return h

    def render_start_screen(self, renderer, font_large, font_small):
        """Render the start screen."""
        self.render_centered_text(renderer, font_large, "TIME&SPACE", 
                                  self.HEIGHT // 3, (255, 255, 255))
        
        self.render_centered_text(renderer, font_small, "Press KEY_1 to start", 
                                  self.HEIGHT // 2, (200, 200, 200))

    def render_loading_screen(self, renderer, font_large, font_small):
        """Render the loading screen."""
        self.render_centered_text(renderer, font_large, "Loading...", 
                                  self.HEIGHT // 3, (255, 255, 0))
        
        self.render_centered_text(renderer, font_small, "Please wait...", 
                                  self.HEIGHT // 2, (200, 200, 200))

    def render_error_screen(self, renderer, font_large, font_small, error_msg):
        """Render the error screen."""
        self.render_centered_text(renderer, font_large, error_msg, 
                                  self.HEIGHT // 2 - 20, (255, 100, 100))
        
        self.render_centered_text(renderer, font_small, "Press KEY_1 to retry", 
                                  self.HEIGHT // 2 + 40, (200, 200, 200))
        self.render_centered_text(renderer, font_small, "Press KEY_2 to return to main", 
                                  self.HEIGHT // 2 + 80, (150, 150, 150))

    def render_success_screen(self, renderer, font_large, font_small):
        """Render the success screen."""
        self.render_centered_text(renderer, font_large, "SUCCESS!", 
                                  self.HEIGHT // 3, (50, 255, 50))
        
        self.render_centered_text(renderer, font_small, "Inspection completed successfully", 
                                  self.HEIGHT // 2, (100, 255, 100))
        
        self.render_centered_text(renderer, font_small, "Press KEY_2 to return to main", 
                                  self.HEIGHT // 2 + 100, (150, 150, 150))

    def run_function_thread(self):
        """Execute the verification function in a separate thread."""
        try:
            success, error_msg = self.func()
            self.result_queue.put((success, error_msg))
        except Exception as e:
            self.result_queue.put((False, f"Exception: {str(e)}"))
        finally:
            self.is_running_verification = False

    def start_verification(self):
        """Start the verification function in a separate thread."""
        if not self.is_running_verification:
            self.is_running_verification = True
            thread = threading.Thread(target=self.run_function_thread, daemon=True)
            thread.start()

    def main_loop(self):
        """Main UI loop."""
        # Initialize SDL and TTF
        sdl2.ext.init()
        sdlttf.TTF_Init()
        
        # Load fonts
        font_large = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        
        # Window setup
        window = sdl2.ext.Window("TIME&SPACE", size=(self.WIDTH, self.HEIGHT))
        window.show()
        sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
        renderer = sdl2.SDL_CreateRenderer(window.window, -1, 0)
        
        # Keyboard input queue
        keys = Queue()
        
        def keyboard_thread():
            kbd = find_keyboard_by_name()
            if kbd is None:
                print("ERROR: Keyboard device not found. UI will run without keyboard input.")
                return
            for e in kbd.read_loop():
                if e.type == ecodes.EV_KEY:
                    if e.value == 0:  # Key release
                        keys.put(e.code)
        
        threading.Thread(target=keyboard_thread, daemon=True).start()
        
        # State management
        current_state = STATE_START
        error_message = ""
        
        running = True
        TARGET_FPS = 30
        FRAME_DELAY = int(1000 / TARGET_FPS)
        
        while running:
            frame_start = sdl2.SDL_GetTicks()
            
            # Check if function returned result
            if not self.result_queue.empty():
                success, err_msg = self.result_queue.get()
                if success:
                    current_state = STATE_SUCCESS
                else:
                    current_state = STATE_ERROR
                    error_message = err_msg
            
            # Handle keyboard input
            if not keys.empty():
                key = keys.get()
                
                if key == ecodes.KEY_7:
                    running = False
                
                elif key == ecodes.KEY_1:
                    # Only start verification if not already running
                    if not self.is_running_verification:
                        if current_state == STATE_START:
                            current_state = STATE_LOADING
                            self.start_verification()
                        elif current_state == STATE_ERROR:
                            current_state = STATE_LOADING
                            self.start_verification()
                        elif current_state == STATE_SUCCESS:
                            current_state = STATE_START
                elif key == ecodes.KEY_2:
                    if not self.is_running_verification:
                        if current_state == STATE_ERROR or current_state == STATE_SUCCESS:
                            current_state = STATE_START
            
            # Clear screen
            sdl2.SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255)
            sdl2.SDL_RenderClear(renderer)
            
            # Render current state
            if current_state == STATE_START:
                self.render_start_screen(renderer, font_large, font_small)
            
            elif current_state == STATE_LOADING:
                self.render_loading_screen(renderer, font_large, font_small)
            
            elif current_state == STATE_ERROR:
                self.render_error_screen(renderer, font_large, font_small, error_message)
            
            elif current_state == STATE_SUCCESS:
                self.render_success_screen(renderer, font_large, font_small)
            
            # Present
            sdl2.SDL_RenderPresent(renderer)
            
            # Frame timing
            frame_time = sdl2.SDL_GetTicks() - frame_start
            if frame_time < FRAME_DELAY:
                sdl2.SDL_Delay(FRAME_DELAY - frame_time)
        
        # Cleanup
        sdlttf.TTF_CloseFont(font_large)
        sdlttf.TTF_CloseFont(font_small)
        sdlttf.TTF_Quit()
        sdl2.SDL_DestroyRenderer(renderer)
        sdl2.ext.quit()
