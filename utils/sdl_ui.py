import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf
from find_keyboard import find_keyboard_by_name
from evdev import ecodes
import threading
from queue import Queue
import math

# Menu states
STATE_START = 0
STATE_LOADING = 1
STATE_ERROR = 2
STATE_SUCCESS = 3

def render_text(renderer, font, message, color=(255, 255, 255)):
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

def render_centered_text(renderer, font, message, y_pos, color=(255, 255, 255), width=800):
    """Render text centered horizontally at given y position."""
    tex, w, h = render_text(renderer, font, message, color)
    x_pos = (width - w) // 2
    dst = sdl2.SDL_Rect(x_pos, y_pos, w, h)
    sdl2.SDL_RenderCopy(renderer, tex, None, dst)
    sdl2.SDL_DestroyTexture(tex)
    return h

def render_start_screen(renderer, font_large, font_small, width, height):
    """Render the start screen."""
    # Title
    render_centered_text(renderer, font_large, "TIME&SPACE", height // 3, (255, 255, 255), width)
    
    # Instructions
    render_centered_text(renderer, font_small, "Press {place_holder} to start", 
                        height // 2, (200, 200, 200), width)

def render_loading_screen(renderer, font_large, font_small, width, height, frame_count):
    """Render the loading screen with animation."""
    # Title
    render_centered_text(renderer, font_large, "Loading...", height // 3, (255, 255, 0), width)
    
    # Animated dots
    dots = "." * ((frame_count // 15) % 4)
    render_centered_text(renderer, font_small, f"Please wait{dots}", 
                        height // 2, (200, 200, 200), width)
    
    # Spinner animation
    angle = (frame_count * 6) % 360
    center_x = width // 2
    center_y = height // 2 + 60
    radius = 20
    
    for i in range(8):
        current_angle = math.radians(angle + i * 45)
        x = int(center_x + radius * math.cos(current_angle))
        y = int(center_y + radius * math.sin(current_angle))
        
        # Fade effect
        alpha = 255 - (i * 30)
        color = sdl2.SDL_Color(alpha, alpha, 0)
        
        rect = sdl2.SDL_Rect(x - 3, y - 3, 6, 6)
        sdl2.SDL_SetRenderDrawColor(renderer, alpha, alpha, 0, 255)
        sdl2.SDL_RenderFillRect(renderer, rect)

def render_error_screen(renderer, font_large, font_small, width, height, error_msg):
    """Render the error screen."""
    # Title
    render_centered_text(renderer, font_large, "ERROR", height // 4, (255, 50, 50), width)
    
    # Error message
    render_centered_text(renderer, font_small, error_msg, 
                        height // 2 - 20, (255, 100, 100), width)
    
    # Instructions
    render_centered_text(renderer, font_small, "Press {place_holder} to retry", 
                        height // 2 + 40, (200, 200, 200), width)
    render_centered_text(renderer, font_small, "Press {place_holder_2} to return", 
                        height // 2 + 80, (150, 150, 150), width)

def render_success_screen(renderer, font_large, font_small, width, height):
    """Render the success screen."""
    # Title
    render_centered_text(renderer, font_large, "SUCCESS!", height // 3, (50, 255, 50), width)
    
    # Message
    render_centered_text(renderer, font_small, "Verification completed successfully", 
                        height // 2, (100, 255, 100), width)
    
    # Instructions
    render_centered_text(renderer, font_small, "Press {place_holder_2}", 
                        height // 2 + 60, (200, 200, 200), width)


def main():
    # Initialize SDL and TTF
    sdl2.ext.init()
    sdlttf.TTF_Init()
    
    # Load fonts
    font_large = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    if not font_large:
        print("Error loading large font!")
        print("Trying alternative font...")
        font_large = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
    
    font_small = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    if not font_small:
        print("Error loading small font!")
        sdlttf.TTF_Quit()
        sdl2.ext.quit()
        exit(1)
    
    # Window setup
    WIDTH = 800
    HEIGHT = 480
    window = sdl2.ext.Window("Menu System", size=(WIDTH, HEIGHT))
    window.show()
    
    renderer = sdl2.SDL_CreateRenderer(window.window, -1, 0)
    
    # Keyboard input queue
    keys = Queue()
    
    def keyboard_thread():
        kbd = find_keyboard_by_name()
        for e in kbd.read_loop():
            if e.type == ecodes.EV_KEY:
                if e.value == 0:  # Key release
                    keys.put(e.code)
    
    threading.Thread(target=keyboard_thread, daemon=True).start()
    
    # State management
    current_state = STATE_START
    frame_count = 0
    loading_timer = 0
    error_message = "Connection failed"
    
    # Error messages for different scenarios
    error_messages = [
        "Error_place_holder_1",
        "Error_place_holder_2",
        "Error_place_holder_3",
        "Error_place_holder_4"
    ]
    error_index = 0
    
    running = True
    TARGET_FPS = 30
    FRAME_DELAY = int(1000 / TARGET_FPS)
    
    while running:
        frame_start = sdl2.SDL_GetTicks()
        
        # Handle keyboard input
        if not keys.empty():
            key = keys.get()
            
            if key == ecodes.KEY_7:
                running = False
            
            elif key == ecodes.KEY_1:
                if current_state == STATE_START:
                    current_state = STATE_LOADING
                    loading_timer = 0
                    frame_count = 0
                elif current_state == STATE_ERROR:
                    current_state = STATE_LOADING
                    loading_timer = 0
                    frame_count = 0
                elif current_state == STATE_SUCCESS:
                    current_state = STATE_START
            
            
            elif key == ecodes.KEY_2:  # Force error
                if current_state == STATE_LOADING:
                    current_state = STATE_ERROR
                    error_message = error_messages[error_index]
                    error_index = (error_index + 1) % len(error_messages)
            
            elif key == ecodes.KEY_3:  # Force success
                if current_state == STATE_LOADING:
                    current_state = STATE_SUCCESS
        
        # State updates
        if current_state == STATE_LOADING:
            loading_timer += 1
            # Simulate loading completion after 3 seconds (90 frames at 30 FPS)
            if loading_timer > 90:
                # Randomly choose success or error (70% success)
                import random
                if random.random() < 0.7:
                    current_state = STATE_SUCCESS
                else:
                    current_state = STATE_ERROR
                    error_message = error_messages[error_index]
                    error_index = (error_index + 1) % len(error_messages)
        
        # Clear screen
        sdl2.SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255)
        sdl2.SDL_RenderClear(renderer)
        
        # Render current state
        if current_state == STATE_START:
            render_start_screen(renderer, font_large, font_small, WIDTH, HEIGHT)
        
        elif current_state == STATE_LOADING:
            render_loading_screen(renderer, font_large, font_small, WIDTH, HEIGHT, frame_count)
        
        elif current_state == STATE_ERROR:
            render_error_screen(renderer, font_large, font_small, WIDTH, HEIGHT, error_message)
        
        elif current_state == STATE_SUCCESS:
            render_success_screen(renderer, font_large, font_small, WIDTH, HEIGHT)
        
        # Present
        sdl2.SDL_RenderPresent(renderer)
        
        # Frame timing
        frame_count += 1
        frame_time = sdl2.SDL_GetTicks() - frame_start
        if frame_time < FRAME_DELAY:
            sdl2.SDL_Delay(FRAME_DELAY - frame_time)
    
    # Cleanup
    sdlttf.TTF_CloseFont(font_large)
    sdlttf.TTF_CloseFont(font_small)
    sdlttf.TTF_Quit()
    sdl2.SDL_DestroyRenderer(renderer)
    sdl2.ext.quit()

if __name__ == "__main__":
    main()
