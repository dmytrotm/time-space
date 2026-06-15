import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf
from utils.find_keyboard import find_keyboards_by_name
import json

try:
    from evdev import ecodes
except ImportError:
    class ecodes:
        EV_KEY = 1
        KEY_1 = 2
        KEY_2 = 3
        KEY_4 = 5
        KEY_7 = 4

import time
import threading
from queue import Queue
import cv2
import os
from datetime import datetime
from configs.config import PHOTOS_DIR

# State constants
STATE_START = 0
STATE_LOADING = 1
STATE_ERROR = 2
STATE_SUCCESS = 3
STATE_INIT = 4

_STATE_NAMES = {0: 'START', 1: 'LOADING', 2: 'ERROR', 3: 'SUCCESS', 4: 'INIT'}

class UIManager:
    def __init__(
        self, verification_manager, image_server, window_width=800, window_height=480,
        resource_monitor=None, save_all=False, save_errors=False,
        shared_state=None, shared_frames=None, cmd_queue=None,
    ):
        self.verification_manager = verification_manager
        self.image_server = image_server
        self.resource_monitor = resource_monitor
        self.WIDTH = window_width
        self.HEIGHT = window_height
        self.save_all = save_all
        self.save_errors = save_errors

        self.shared_state = shared_state
        self.shared_frames = shared_frames
        self.cmd_queue = cmd_queue
        self._streaming_active = False

        # Стан для кожного з 2-х робочих місць
        self.states = {1: STATE_START, 2: STATE_START}
        self.error_messages = {1: "", 2: ""}
        self.is_running = {1: False, 2: False}
        self.state_timers = {1: 0, 2: 0} # Таймери для авто-повернення
        self.AUTO_TIMEOUT_MS = 4000 # 4 секунди на показ результату

        self.missing_zones = {1: False, 2: False}

        self.previous_results = {1: None, 2: None}

    def _check_missing_zones(self):
        """Перевіряємо чи є камери для всіх зон обох робочих місць"""
        for ws in [1, 2]:
            ws_cams = self.image_server.workspaces.get(ws, {})
            # Якщо хоча б одна зона None, вважаємо, що не всі камери ініціалізовані
            if ws_cams.get(1) is None or ws_cams.get(2) is None:
                self.missing_zones[ws] = True
            else:
                self.missing_zones[ws] = False

    def render_text(self, renderer, font, message, color=(255, 255, 255)):
        text_surface = sdlttf.TTF_RenderUTF8_Blended(
            font, message.encode("utf-8"), sdl2.SDL_Color(color[0], color[1], color[2])
        )
        texture = sdl2.SDL_CreateTextureFromSurface(renderer, text_surface)
        w = text_surface.contents.w
        h = text_surface.contents.h
        sdl2.SDL_FreeSurface(text_surface)
        return texture, w, h

    def render_centered_text(self, renderer, font, message, y_pos, target_width, color=(255, 255, 255)):
        lines = message.split('\n')
        current_y = y_pos
        total_h = 0
        for line in lines:
            tex, w, h = self.render_text(renderer, font, line, color)
            x_pos = (target_width - w) // 2
            dst = sdl2.SDL_Rect(x_pos, current_y, w, h)
            sdl2.SDL_RenderCopy(renderer, tex, None, dst)
            sdl2.SDL_DestroyTexture(tex)
            current_y += h + 5  # 5px padding between lines
            total_h += h + 5
        return total_h

    def draw_workspace_ui(self, renderer, ws_id, target_width, target_height, font_large, font_small):
        """Малює UI для конкретного робочого місця у його власну текстуру (без поворотів, як звичайно)"""
        sdl2.SDL_SetRenderDrawColor(renderer, 30, 30, 40, 255)
        sdl2.SDL_RenderClear(renderer)
        
        state = self.states[ws_id]
        
        if state == STATE_INIT:
            self.render_centered_text(renderer, font_large, "Initializing...", target_height // 3, target_width, (255, 200, 50))
            self.render_centered_text(renderer, font_small, "Scanning cameras...", target_height // 2, target_width, (200, 200, 200))
            return

        if self.missing_zones[ws_id]:
            self.render_centered_text(renderer, font_small, "Zones not initialized.", target_height // 2 - 20, target_width, (255, 100, 100))
            self.render_centered_text(renderer, font_small, "Check cameras and markers.", target_height // 2 + 20, target_width, (255, 100, 100))
            return
        
        # Назва робочого місця
        self.render_centered_text(renderer, font_small, f"WORKSPACE {ws_id}", 20, target_width, (100, 200, 255))
        
        if state == STATE_START:
            self.render_centered_text(renderer, font_large, "TIME&SPACE", target_height // 3, target_width, (255, 255, 255))
            key_name = "KEY_1" if ws_id == 1 else "KEY_4"
            self.render_centered_text(renderer, font_small, f"Press {key_name} to start", target_height // 2, target_width, (200, 200, 200))
            
        elif state == STATE_LOADING:
            self.render_centered_text(renderer, font_large, "Loading...", target_height // 3, target_width, (255, 255, 0))
            self.render_centered_text(renderer, font_small, "Please wait...", target_height // 2, target_width, (200, 200, 200))
            
        elif state == STATE_ERROR:
            err_msg = self.error_messages[ws_id]
            # Малюємо помилку трохи вище і зберігаємо її загальну висоту
            h = self.render_centered_text(renderer, font_large, err_msg, target_height // 2 - 60, target_width, (255, 100, 100))
            # Малюємо "Auto-returning..." динамічно під текстом помилки
            self.render_centered_text(renderer, font_small, "Auto-returning...", target_height // 2 - 60 + h + 30, target_width, (150, 150, 150))
            
        elif state == STATE_SUCCESS:
            self.render_centered_text(renderer, font_large, "SUCCESS!", target_height // 3, target_width, (50, 255, 50))
            self.render_centered_text(renderer, font_small, "Inspection completed", target_height // 2, target_width, (100, 255, 100))
            self.render_centered_text(renderer, font_small, "Auto-returning...", target_height // 2 + 50, target_width, (150, 150, 150))

    def _update_shared_state(self):
        if self.shared_state is None:
            return
        self.shared_state['ws1_state'] = _STATE_NAMES.get(self.states[1], 'START')
        self.shared_state['ws2_state'] = _STATE_NAMES.get(self.states[2], 'START')
        self.shared_state['ws1_error'] = self.error_messages[1]
        self.shared_state['ws2_error'] = self.error_messages[2]

    def _frame_streaming_thread(self):
        import cv2
        while self._streaming_active:
            try:
                cams = self.image_server.get_assigned_cameras()
                if self.shared_state is not None:
                    self.shared_state['cameras'] = cams
                for cam_id in cams:
                    raw, ws_frame = self.image_server.capture_frame(cam_id)
                    if raw is not None:
                        _, jpg = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        self.shared_frames[f'raw_{cam_id}'] = jpg.tobytes()
                    if ws_frame is not None:
                        _, jpg = cv2.imencode('.jpg', ws_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        self.shared_frames[f'ws_{cam_id}'] = jpg.tobytes()
            except Exception as e:
                print(f"[FrameStream] Error: {e}")
            time.sleep(0.1)

    def start_verification(self, ws_id):
        if not self.is_running[ws_id]:
            self.is_running[ws_id] = True
            threading.Thread(target=self._run_verification_task, args=(ws_id,), daemon=True).start()
            return True
        return False

    def _run_verification_task(self, ws_id):
        try:
            # Знімаємо фото для конкретного робочого місця
            images = self.image_server.take_photos(workspace_id=ws_id)
            if not images or len(images) != 2:
                raise ValueError("Not enough images captured for verification.")
            
            # --save-all: зберігаємо ВСІ сирі фото (і успіх, і помилки)
            if self.save_all:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                os.makedirs("photos", exist_ok=True)
                for i, img in enumerate(images):
                    filename = f"photos/ws{ws_id}_zone{i+1}_{timestamp}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"[WS{ws_id}] Saved raw photo: {filename}")
            
            self.verification_manager.trigger_verification(ws_id, images, save_errors=self.save_errors)
        except Exception as e:
            print(f"[WS{ws_id}] Error starting verification: {e}")
            # Відправляємо помилку в чергу, щоб головний потік її обробив
            self.verification_manager.result_queue.put({
                "status": "DONE",
                "workspace_id": ws_id,
                "success": False,
                "error": "Camera Error"
            })
    def save_previous_result(self, ws_id):
        result = self.previous_results.get(ws_id)
        if result is None:
            print(f"[WS{ws_id}] No previous result to save.")
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(PHOTOS_DIR, exist_ok=True)
        for i, img in enumerate(result.get("raw_images", [])):
            filename = os.path.join(PHOTOS_DIR, f"ws{ws_id}_zone{i+1}_{timestamp}_raw.jpg")
            cv2.imwrite(filename, img)
            print(f"[WS{ws_id}] Saved: {filename}")
        roi_list = result.get("roi_results_list", [])
        json_filename = os.path.join(PHOTOS_DIR, f"ws{ws_id}_{timestamp}_results.json")
        with open(json_filename, "w") as f:
            json.dump({
                "workspace_id": ws_id,
                "success": result.get("success"),
                "error": result.get("error", ""),
                "roi_results": roi_list,
            }, f, indent=2)
        print(f"[WS{ws_id}] Saved JSON: {json_filename}")
    def main_loop(self):
        sdl2.ext.init()
        sdlttf.TTF_Init()

        # Автовизначення розміру дисплея
        display_mode = sdl2.SDL_DisplayMode()
        if sdl2.SDL_GetCurrentDisplayMode(0, display_mode) == 0:
            self.WIDTH = display_mode.w
            self.HEIGHT = display_mode.h
            print(f"Display detected: {self.WIDTH}x{self.HEIGHT}")
        else:
            print(f"Could not detect display, using default: {self.WIDTH}x{self.HEIGHT}")

        font_large = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = sdlttf.TTF_OpenFont(b"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        
        if not font_large or not font_small:
            raise RuntimeError("Fonts are not loaded, please check the paths")
            
        window = sdl2.ext.Window("TIME&SPACE", size=(self.WIDTH, self.HEIGHT),
                                 flags=sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)
        window.show()
        sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
        
        # Вмикаємо підтримку Target Texture
        renderer = sdl2.SDL_CreateRenderer(window.window, -1, sdl2.SDL_RENDERER_TARGETTEXTURE | sdl2.SDL_RENDERER_ACCELERATED)

        keys = Queue()

        import select

        def keyboard_thread():
            while True:
                kbds = find_keyboards_by_name("SayoDevice")
                if not kbds:
                    time.sleep(2)
                    continue
                
                kbds_dict = {dev.fd: dev for dev in kbds}
                print(f"Listening on {len(kbds)} keyboard(s)...")

                try:
                    while True:
                        r, w, x = select.select(kbds_dict.keys(), [], [])
                        for fd in r:
                            for e in kbds_dict[fd].read():
                                if e.type == ecodes.EV_KEY and e.value == 1:  # Key DOWN
                                    keys.put(e.code)
                except Exception as ex:
                    print("Keyboard disconnected or error:", ex)
                    time.sleep(1)

        threading.Thread(target=keyboard_thread, daemon=True).start()

        self._check_missing_zones()

        running = True
        TARGET_FPS = 30
        FRAME_DELAY = int(1000 / TARGET_FPS)
        
        # Розміри однієї половинки (до повороту)
        half_w = self.HEIGHT
        half_h = self.WIDTH // 2
        
        # Створюємо текстури-цілі для обох робочих місць
        tex_target_1 = sdl2.SDL_CreateTexture(renderer, sdl2.SDL_PIXELFORMAT_RGBA8888, sdl2.SDL_TEXTUREACCESS_TARGET, half_w, half_h)
        tex_target_2 = sdl2.SDL_CreateTexture(renderer, sdl2.SDL_PIXELFORMAT_RGBA8888, sdl2.SDL_TEXTUREACCESS_TARGET, half_w, half_h)

        self.states = {1: STATE_INIT, 2: STATE_INIT}

        def init_task():
            print("Starting camera scan while UI displays 'Initializing'...")
            self.image_server._init_cameras()
            self._check_missing_zones()
            self.states[1] = STATE_START
            self.states[2] = STATE_START
            self._update_shared_state()

        threading.Thread(target=init_task, daemon=True).start()

        if self.shared_frames is not None:
            self._streaming_active = True
            threading.Thread(target=self._frame_streaming_thread, daemon=True).start()
        
        running = True
        TARGET_FPS = 30
        FRAME_DELAY = int(1000 / TARGET_FPS)

        while running:
            frame_start = sdl2.SDL_GetTicks()
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    running = False

            # Перевірка результатів від Worker-процесів
            result = self.verification_manager.check_results()
            if result:
                ws_id = result.get("workspace_id", 1)
                self.is_running[ws_id] = False

                if result["success"]:
                    self.states[ws_id] = STATE_SUCCESS
                else:
                    self.states[ws_id] = STATE_ERROR
                    self.error_messages[ws_id] = result.get("error", "Error")
                    if "error_images" in result and self.save_errors:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        # --save-errors: зберігаємо тільки проблемні кропи
                        os.makedirs("photos", exist_ok=True)
                        for error_name, img in result["error_images"].items():
                            filename = f"photos/ws{ws_id}_{timestamp}_{error_name}.jpg"
                            cv2.imwrite(filename, img)
                            print(f"[WS{ws_id}] Saved error image: {filename}")

                # Запускаємо таймер авто-повернення
                self.state_timers[ws_id] = sdl2.SDL_GetTicks()
                self.previous_results[ws_id] = result
                self._update_shared_state()

            # Обробка таймаутів для автоматичного повернення
            current_time = sdl2.SDL_GetTicks()
            state_changed = False
            for ws_id in [1, 2]:
                if self.states[ws_id] in [STATE_SUCCESS, STATE_ERROR]:
                    if current_time - self.state_timers[ws_id] > self.AUTO_TIMEOUT_MS:
                        self.states[ws_id] = STATE_START
                        state_changed = True
            if state_changed:
                self._update_shared_state()

            # Обробка веб-команд (workspace_server)
            if self.cmd_queue is not None:
                while True:
                    try:
                        cmd = self.cmd_queue.get_nowait()
                        if cmd.get('command') == 'TRIGGER':
                            ws_id = cmd.get('workspace_id')
                            if ws_id in (1, 2):
                                key = ecodes.KEY_1 if ws_id == 1 else ecodes.KEY_4
                                keys.put(key)
                    except Exception:
                        break

            # Обробка натискань клавіш
            key_state_changed = False
            while not keys.empty():
                key = keys.get()
                if key == ecodes.KEY_7:
                    running = False
                elif key == ecodes.KEY_1:
                    if self.states[1] in [STATE_START, STATE_ERROR, STATE_SUCCESS] and not self.is_running[1]:
                        if self.missing_zones[1]:
                            print("[WS1] Камери не ініціалізовано. Спроба переініціалізації...")
                            self.image_server.reinit_missing_cameras()
                            self._check_missing_zones()

                        if not self.missing_zones[1]:
                            self.states[1] = STATE_LOADING
                            self.start_verification(1)
                            key_state_changed = True
                elif key == ecodes.KEY_4:
                    if self.states[2] in [STATE_START, STATE_ERROR] and not self.is_running[2]:
                        if self.missing_zones[2]:
                            print("[WS2] Камери не ініціалізовано. Спроба переініціалізації...")
                            self.image_server.reinit_missing_cameras()
                            self._check_missing_zones()

                        if not self.missing_zones[2]:
                            self.states[2] = STATE_LOADING
                            self.start_verification(2)
                            key_state_changed = True
                elif key == ecodes.KEY_2:
                    self.save_previous_result(1)
                elif key == ecodes.KEY_3:
                    self.save_previous_result(2)
                    
            if key_state_changed:
                self._update_shared_state()

            # --- РЕНДЕРИНГ WORKSPACE 1 ---
            sdl2.SDL_SetRenderTarget(renderer, tex_target_1)
            self.draw_workspace_ui(renderer, 1, half_w, half_h, font_large, font_small)
            
            # --- РЕНДЕРИНГ WORKSPACE 2 ---
            sdl2.SDL_SetRenderTarget(renderer, tex_target_2)
            self.draw_workspace_ui(renderer, 2, half_w, half_h, font_large, font_small)
            
            # --- ПОВЕРНЕННЯ НА ГОЛОВНИЙ ЕКРАН ТА КОМПОЗИЦІЯ ---
            sdl2.SDL_SetRenderTarget(renderer, None)
            sdl2.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)
            sdl2.SDL_RenderClear(renderer)

            half_screen_w = self.WIDTH // 2   # 512
            half_screen_h = self.HEIGHT        # 600

            # SDL_RenderCopyEx масштабує до dst_rect, потім обертає навколо центру.
            # Щоб після обертання на 90° контент заповнив half_screen_w × half_screen_h,
            # dst_rect має бути half_screen_h × half_screen_w, зміщений на offset.
            offset_x = (half_screen_w - half_screen_h) // 2  # (512-600)/2 = -44
            offset_y = (half_screen_h - half_screen_w) // 2  # (600-512)/2 = 44

            # Workspace 1 (Ліва половина): Поворот на 90 градусів вправо
            dst_w1 = sdl2.SDL_Rect(offset_x, offset_y, half_screen_h, half_screen_w)
            sdl2.SDL_RenderCopyEx(renderer, tex_target_1, None, dst_w1, 90.0, None, sdl2.SDL_FLIP_NONE)

            # Workspace 2 (Права половина): Поворот на 270 (-90) градусів вліво
            dst_w2 = sdl2.SDL_Rect(half_screen_w + offset_x, offset_y, half_screen_h, half_screen_w)
            sdl2.SDL_RenderCopyEx(renderer, tex_target_2, None, dst_w2, 270.0, None, sdl2.SDL_FLIP_NONE)

            # Відображаємо
            sdl2.SDL_RenderPresent(renderer)

            frame_time = sdl2.SDL_GetTicks() - frame_start
            if frame_time < FRAME_DELAY:
                sdl2.SDL_Delay(FRAME_DELAY - frame_time)

        self._streaming_active = False

        # Очищення
        sdl2.SDL_DestroyTexture(tex_target_1)
        sdl2.SDL_DestroyTexture(tex_target_2)
        sdlttf.TTF_CloseFont(font_large)
        sdlttf.TTF_CloseFont(font_small)
        sdlttf.TTF_Quit()
        sdl2.SDL_DestroyRenderer(renderer)
        sdl2.ext.quit()
