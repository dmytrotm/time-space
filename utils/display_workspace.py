import cv2
import numpy as np
import time
import sdl2
import sdl2.ext
import argparse
import sys
import os
from constants import ZONES_DICT


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processors.workspace_extractor import WorkspaceExtractor
from processors.aruco_detector import aruco_factory 

def setup_camera(camera_id):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    for _ in range(10): 
        cap.read()
    return cap

def display_workspace(id = 0, output_folder = "photos"):
    if id is None:
        id = 0
    current_id = id
    cap = setup_camera(current_id)

    my_zones = ZONES_DICT
    
    extractor = WorkspaceExtractor(aruco_factory(track_time=False), defined_zones=my_zones)

    sdl2.ext.init()
    window_width, window_height = 1280, 720  
    window = sdl2.ext.Window("Extracted Workspace (SDL2)", size=(window_width, window_height))
    window.show()
    
    renderer = sdl2.ext.Renderer(window)
    
    print("Початок трансляції Workspace через SDL2... Закрийте вікно або натисніть ESC для виходу.")

    running = True
    prev_time = time.time()
    fps = 0.0

    image = None
    workspace_image = None

    ws_id = -1
    while running:
        ret, image = cap.read()
        if not ret: 
            # If a camera fails, try the next one or loop back
            print(f"Camera {current_id} failed, trying next...")
            cap.release()
            current_id = (current_id + 1) % 10
            cap = setup_camera(current_id)
            continue

        ws_id, workspace_image = extractor.extract_workspace(image)

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.repeat != 0:
                    continue
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE or event.key.keysym.sym == sdl2.SDLK_7:
                    running = False
                if event.key.keysym.sym == sdl2.SDLK_1:
                    print(f"Switching from camera {current_id}...")
                    cap.release()
                    current_id = (current_id + 1) % 10
                    cap = setup_camera(current_id)
                    print(f"Switched to camera {current_id}")
                if event.key.keysym.sym == sdl2.SDLK_2:
                    timestamp = int(time.time())
                    if workspace_image is not None:
                        folder = f"{output_folder}/workspace_{ws_id}"
                        os.makedirs(folder, exist_ok=True)
                        filename = f"{folder}/{timestamp}_cam{current_id}.jpg"
                        if cv2.imwrite(filename, workspace_image):
                            print(f"Saved workspace from current cam {current_id} to {filename}")
                        else:
                            print(f"Error: Could not save workspace from cam {current_id} to {filename}")
                    else:
                        print(f"No workspace detected on current camera {current_id}")

                if event.key.keysym.sym == sdl2.SDLK_3:
                    timestamp = int(time.time())
                    # Capture from 2 physical cameras (usually 0 and 2 on this system)
                    for cam_id in [0, 2]:
                        t_ws_image = None
                        t_ws_id = -1
                        has_cam = False
                        
                        if cam_id == current_id:
                            t_ws_image = workspace_image
                            t_ws_id = ws_id
                            has_cam = True
                        else:
                            temp_cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                            if temp_cap.isOpened():
                                has_cam = True
                                print(f"Capturing from camera {cam_id}...")
                                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
                                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
                                for _ in range(10):
                                    temp_cap.read()
                                    
                                t_ret, t_image = temp_cap.read()
                                if t_ret:
                                    t_ws_id, t_ws_image = extractor.extract_workspace(t_image)
                                temp_cap.release()
                        
                        if t_ws_image is not None:
                            folder = f"{output_folder}/workspace_{t_ws_id}"
                            os.makedirs(folder, exist_ok=True)
                            filename = f"{folder}/{timestamp}_cam{cam_id}.jpg"
                            if cv2.imwrite(filename, t_ws_image):
                                print(f"Saved workspace from cam {cam_id} to {filename}")
                            else:
                                print(f"Error: Could not save workspace from cam {cam_id} to {filename}")
                        elif has_cam:
                             print(f"No workspace detected on camera {cam_id}")

        if workspace_image is not None:
            display_img = workspace_image
            cv2.putText(display_img, f"Workspace {ws_id} Active | FPS: {fps:.1f}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        else:
            display_img = image.copy()
            cv2.putText(display_img, "WORKSPACE NOT DETECTED", (100, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 8)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (100, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6)

        image_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
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
        fps = 1.0 / (curr - prev_time) if (curr - prev_time) > 0 else 0
        prev_time = curr

    cap.release()
    sdl2.ext.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and display workspace using SDL2.")
    parser.add_argument("--id", type=int, default=0, help="Cams ID")
    parser.add_argument("-o", "--output", type=str, default="photos", help="Parent output folder")

    args = parser.parse_args()

    display_workspace(args.id, args.output)