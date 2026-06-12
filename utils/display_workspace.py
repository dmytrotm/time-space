import cv2
import numpy as np
import time
import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import ZONES_DICT_WS1, ZONES_DICT_WS2, CAMERA_RESOLUTION
from processors.workspace_extractor import WorkspaceExtractor
from processors.aruco_detector import aruco_factory

# ── Camera discovery ───────────────────────────────────────────────────────────
_ISP_PREFIXES = ("pispbe", "rpi-hevc", "bcm2835")


def find_capture_indices():
    """Find real V4L2 capture nodes via sysfs (no device open, instant)."""
    sysfs_base = "/sys/class/video4linux"
    indices = []
    if not os.path.isdir(sysfs_base):
        return indices
    for entry in sorted(os.listdir(sysfs_base)):
        if not entry.startswith("video"):
            continue
        sysfs_path = os.path.join(sysfs_base, entry)
        try:
            with open(os.path.join(sysfs_path, "index")) as f:
                if int(f.read().strip()) != 0:
                    continue
        except (OSError, ValueError):
            continue
        try:
            with open(os.path.join(sysfs_path, "name")) as f:
                dev_name = f.read().strip()
        except OSError:
            continue
        if any(dev_name.lower().startswith(p) for p in _ISP_PREFIXES):
            continue
        try:
            indices.append(int(entry.replace("video", "")))
        except ValueError:
            continue
    return indices


def open_camera(cam_id, width=None, height=None):
    """Open a camera with given (or default) resolution and warmup."""
    w = width or CAMERA_RESOLUTION[0]
    h = height or CAMERA_RESOLUTION[1]
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(8):
        cap.grab()
    return cap


# ── SDL helpers ────────────────────────────────────────────────────────────────

def frame_to_texture(renderer, frame):
    """Convert an OpenCV BGR frame to an SDL texture."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    surface = sdl2.SDL_CreateRGBSurfaceFrom(
        rgb.ctypes.data, w, h, 24, w * 3,
        0x0000FF, 0x00FF00, 0xFF0000, 0
    )
    texture = sdl2.SDL_CreateTextureFromSurface(renderer, surface)
    sdl2.SDL_FreeSurface(surface)
    return texture, w, h


def render_text(renderer, font, message, color=(255, 255, 255)):
    """Render UTF-8 text to an SDL texture."""
    surface = sdlttf.TTF_RenderUTF8_Blended(
        font, message.encode("utf-8"),
        sdl2.SDL_Color(color[0], color[1], color[2])
    )
    texture = sdl2.SDL_CreateTextureFromSurface(renderer, surface)
    w = surface.contents.w
    h = surface.contents.h
    sdl2.SDL_FreeSurface(surface)
    return texture, w, h


def draw_overlay_bar(renderer, font, text, win_w, color=(255, 255, 255)):
    """Draw a semi-transparent bar at the top with overlay text."""
    tex, tw, th = render_text(renderer, font, text, color)
    sdl2.SDL_SetRenderDrawBlendMode(renderer, sdl2.SDL_BLENDMODE_BLEND)
    sdl2.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 170)
    sdl2.SDL_RenderFillRect(renderer, sdl2.SDL_Rect(0, 0, win_w, th + 14))
    sdl2.SDL_RenderCopy(renderer, tex, None, sdl2.SDL_Rect(8, 7, tw, th))
    sdl2.SDL_DestroyTexture(tex)


# ── Workspace extraction ───────────────────────────────────────────────────────

def extract_workspace(image, ext_ws1, ext_ws2):
    """Try WS1 zone dict first, then WS2. Returns (ws_id, ws_image)."""
    ws_id, ws_img = ext_ws1.extract_workspace(image)
    if ws_img is None:
        ws_id, ws_img = ext_ws2.extract_workspace(image)
    return ws_id, ws_img


# ── Main ───────────────────────────────────────────────────────────────────────

def display_workspace(output_folder="photos"):
    # --- Discover cameras ---
    cam_ids = find_capture_indices()
    if not cam_ids:
        print("sysfs scan found nothing, falling back to brute-force 0..9...")
        cam_ids = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                cam_ids.append(i)
            cap.release()
    if not cam_ids:
        print("Error: No cameras found.")
        return
    print(f"Found {len(cam_ids)} camera(s): {cam_ids}")

    # --- ArUco / workspace extractors ---
    ext_ws1 = WorkspaceExtractor(aruco_factory(track_time=False), defined_zones=ZONES_DICT_WS1)
    ext_ws2 = WorkspaceExtractor(aruco_factory(track_time=False), defined_zones=ZONES_DICT_WS2)

    # --- SDL init ---
    sdl2.ext.init()
    sdlttf.TTF_Init()

    display_mode = sdl2.SDL_DisplayMode()
    if sdl2.SDL_GetCurrentDisplayMode(0, display_mode) == 0:
        WIN_W, WIN_H = display_mode.w, display_mode.h
    else:
        WIN_W, WIN_H = 1280, 720
    print(f"Display: {WIN_W}x{WIN_H}")

    window = sdl2.ext.Window("Workspace Display", size=(WIN_W, WIN_H))
    window.show()
    sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
    renderer = sdl2.SDL_CreateRenderer(window.window, -1, sdl2.SDL_RENDERER_ACCELERATED)

    font = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    font_cell = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

    # --- State ---
    current_idx = 0
    cap = open_camera(cam_ids[current_idx])
    show_all = False
    all_caps: dict = {}

    workspace_image = None
    ws_id = -1
    image = None
    running = True
    prev_time = time.time()
    fps = 0.0
    TARGET_FPS = 30
    FRAME_DELAY = int(1000 / TARGET_FPS)

    print("Controls: 1=next cam | 2=save photo | 3=all cams | ESC/7=exit")

    while running:
        frame_start = sdl2.SDL_GetTicks()

        # --- Events ---
        for event in sdl2.ext.get_events():
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            if event.type != sdl2.SDL_KEYDOWN or event.key.repeat != 0:
                continue

            sym = event.key.keysym.sym

            if sym in (sdl2.SDLK_ESCAPE, sdl2.SDLK_7):
                running = False

            elif sym == sdl2.SDLK_1:
                # Next camera
                if show_all:
                    for c in all_caps.values():
                        c.release()
                    all_caps.clear()
                    show_all = False
                else:
                    cap.release()
                current_idx = (current_idx + 1) % len(cam_ids)
                cap = open_camera(cam_ids[current_idx])
                workspace_image = None
                print(f"Camera → {cam_ids[current_idx]}  ({current_idx + 1}/{len(cam_ids)})")

            elif sym == sdl2.SDLK_2:
                # Save current workspace photo
                timestamp = int(time.time())
                if workspace_image is not None:
                    folder = os.path.join(output_folder, f"workspace_{ws_id}")
                    os.makedirs(folder, exist_ok=True)
                    fn = os.path.join(folder, f"{timestamp}_cam{cam_ids[current_idx]}.jpg")
                    if cv2.imwrite(fn, workspace_image):
                        print(f"Saved: {fn}")
                    else:
                        print(f"Error saving: {fn}")
                else:
                    print(f"No workspace detected on cam {cam_ids[current_idx]}")

            elif sym == sdl2.SDLK_3:
                # Toggle show-all grid
                show_all = not show_all
                if show_all:
                    print(f"Show ALL ({len(cam_ids)} cams, 640×480 each)")
                    cap.release()
                    all_caps = {cid: open_camera(cid, 640, 480) for cid in cam_ids}
                else:
                    for c in all_caps.values():
                        c.release()
                    all_caps.clear()
                    cap = open_camera(cam_ids[current_idx])
                    print(f"Single cam mode → {cam_ids[current_idx]}")

        # --- Render ---
        sdl2.SDL_SetRenderDrawColor(renderer, 18, 18, 28, 255)
        sdl2.SDL_RenderClear(renderer)

        if show_all:
            n = len(cam_ids)
            cols = max(1, int(np.ceil(np.sqrt(n))))
            rows = max(1, int(np.ceil(n / cols)))
            cell_w = WIN_W // cols
            cell_h = WIN_H // rows

            for i, cid in enumerate(cam_ids):
                c = all_caps.get(cid)
                if c is None:
                    continue
                ret, frame = c.read()
                if not ret or frame is None:
                    continue

                _, ws_img = extract_workspace(frame, ext_ws1, ext_ws2)
                display_img = ws_img if ws_img is not None else frame
                cell_frame = cv2.resize(display_img, (cell_w, cell_h))

                x = (i % cols) * cell_w
                y = (i // cols) * cell_h

                tex, _, _ = frame_to_texture(renderer, cell_frame)
                sdl2.SDL_RenderCopy(renderer, tex, None, sdl2.SDL_Rect(x, y, cell_w, cell_h))
                sdl2.SDL_DestroyTexture(tex)

                # Cell label
                if font_cell:
                    ok = ws_img is not None
                    label = f"#{i + 1} ID:{cid} | {'WS ACTIVE' if ok else 'NO WS'}"
                    clr = (80, 255, 80) if ok else (255, 80, 80)
                    lt, lw, lh = render_text(renderer, font_cell, label, clr)
                    sdl2.SDL_SetRenderDrawBlendMode(renderer, sdl2.SDL_BLENDMODE_BLEND)
                    sdl2.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 150)
                    sdl2.SDL_RenderFillRect(renderer, sdl2.SDL_Rect(x, y, lw + 12, lh + 8))
                    sdl2.SDL_RenderCopy(renderer, lt, None, sdl2.SDL_Rect(x + 6, y + 4, lw, lh))
                    sdl2.SDL_DestroyTexture(lt)

            if font:
                draw_overlay_bar(renderer, font, f"ALL CAMERAS ({n}) | FPS: {fps:.1f}  [1=single  3=all  ESC=exit]", WIN_W)

        else:
            ret, image = cap.read()
            if not ret or image is None:
                print(f"Camera {cam_ids[current_idx]} failed → switching...")
                cap.release()
                current_idx = (current_idx + 1) % len(cam_ids)
                cap = open_camera(cam_ids[current_idx])
                continue

            ws_id, workspace_image = extract_workspace(image, ext_ws1, ext_ws2)

            if workspace_image is not None:
                display_img = workspace_image.copy()
                overlay_color = (80, 255, 80)
                status = f"Workspace {ws_id} Active"
            else:
                display_img = image.copy()
                cv2.putText(display_img, "WORKSPACE NOT DETECTED",
                            (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                overlay_color = (255, 80, 80)
                status = "WORKSPACE NOT DETECTED"

            render_frame = cv2.resize(display_img, (WIN_W, WIN_H))
            tex, _, _ = frame_to_texture(renderer, render_frame)
            sdl2.SDL_RenderCopy(renderer, tex, None, sdl2.SDL_Rect(0, 0, WIN_W, WIN_H))
            sdl2.SDL_DestroyTexture(tex)

            if font:
                overlay = (
                    f"CAM {current_idx + 1}/{len(cam_ids)}  ID:{cam_ids[current_idx]} | "
                    f"{status} | FPS:{fps:.1f}  [1=next  2=save  3=all  ESC=exit]"
                )
                draw_overlay_bar(renderer, font, overlay, WIN_W, overlay_color)

        sdl2.SDL_RenderPresent(renderer)

        curr = time.time()
        fps = 1.0 / (curr - prev_time) if (curr - prev_time) > 0 else 0
        prev_time = curr

        elapsed = sdl2.SDL_GetTicks() - frame_start
        if elapsed < FRAME_DELAY:
            sdl2.SDL_Delay(FRAME_DELAY - elapsed)

    # --- Cleanup ---
    cap.release()
    for c in all_caps.values():
        c.release()
    if font:
        sdlttf.TTF_CloseFont(font)
    if font_cell:
        sdlttf.TTF_CloseFont(font_cell)
    sdlttf.TTF_Quit()
    sdl2.SDL_DestroyRenderer(renderer)
    sdl2.ext.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display extracted workspaces from all cameras.\n"
                    "  1 = cycle camera  |  2 = save photo  |  3 = show all  |  ESC/7 = exit"
    )
    parser.add_argument("-o", "--output", type=str, default="photos",
                        help="Parent folder for saved photos (default: photos)")
    args = parser.parse_args()
    display_workspace(args.output)