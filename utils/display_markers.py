import cv2
import numpy as np
import time
import sdl2
import sdl2.ext
import argparse
import sys
import os

from configs.config import CAMERA_RESOLUTION

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from processors.workspace_extractor import WorkspaceExtractor

from processors.aruco_detector import aruco_factory 
def setup_camera(camera_id):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 5)
    for _ in range(10): 
        cap.read()
    return cap

def display_markers(id = 0):
    if id is None:
        id = 0
    current_id = id
    cap = setup_camera(current_id)

    extractor = WorkspaceExtractor(aruco_factory(track_time=False))

    sdl2.ext.init()
    window_width, window_height = 1280, 720  
    window = sdl2.ext.Window("Detected Markers (SDL2)", size=(window_width, window_height))
    window.show()
    
    renderer = sdl2.ext.Renderer(window)
    
    print("Початок трансляції SDL2... Закрийте вікно або натисніть ESC для виходу.")

    running = True
    prev_time = time.time()

    while running:
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.repeat != 0:
                    continue
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False
                elif event.key.keysym.sym == sdl2.SDLK_1:
                    print(f"Switching from camera {current_id}...")
                    cap.release()
                    current_id = (current_id + 1) % 10
                    cap = setup_camera(current_id)
                    print(f"Switched to camera {current_id}")

        ret, image = cap.read()
        if not ret: 
            print(f"Camera {current_id} failed, trying next...")
            cap.release()
            current_id = (current_id + 1) % 10
            cap = setup_camera(current_id)
            continue

        markers = extractor.aruco_detector.detect_markers(image)
        
        for marker in markers:
            corners = np.array(marker['corners'], dtype=np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 6)
            center = tuple(map(int, marker['center']))
            cv2.putText(image, f"ID: {marker['id']}", (center[0] + 20, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (window_width, window_height))

        surface = sdl2.SDL_CreateRGBSurfaceFrom(
            image_resized.ctypes.data,
            window_width,
            window_height,
            24, 
            window_width * 3, 
            0xff0000, 0x00ff00, 0x0000ff, 0 
        )

        texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, surface)
        sdl2.SDL_FreeSurface(surface) 

        renderer.clear()
        sdl2.SDL_RenderCopy(renderer.sdlrenderer, texture, None, None)
        renderer.present()
        
        sdl2.SDL_DestroyTexture(texture) 

        curr = time.time()
        fps = 1.0 / (curr - prev_time)
        prev_time = curr

    cap.release()
    sdl2.ext.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous detection and display of custom markers in a video stream.")
    parser.add_argument("--id", type=int, default=0, help="Cams ID (default: 0)")
    args = parser.parse_args()

    display_markers(args.id)