import cv2
import numpy as np
import time
import sdl2
import sdl2.ext
# Решта ваших імпортів (WorkspaceExtractor тощо) залишаються без змін

def display_markers():
    # --- Налаштування камери ---
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

    for _ in range(4): cap.read()

    extractor = WorkspaceExtractor(aruco_factory(track_time=False))

    # --- Ініціалізація SDL2 ---
    sdl2.ext.init()
    window_width, window_height = 1280, 720  # Розмір вікна для відображення
    window = sdl2.ext.Window("Detected Markers (SDL2)", size=(window_width, window_height))
    window.show()
    
    renderer = sdl2.ext.Renderer(window)
    
    print("Початок трансляції SDL2... Закрийте вікно або натисніть ESC для виходу.")

    running = True
    prev_time = time.time()

    while running:
        # 1. Обробка подій SDL2 (вихід)
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False

        ret, image = cap.read()
        if not ret: break

        # 2. Логіка обробки маркерів (як у вашому коді)
        markers = extractor.aruco_detector.detect_markers(image)
        
        for marker in markers:
            corners = np.array(marker['corners'], dtype=np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 6)
            center = tuple(map(int, marker['center']))
            cv2.putText(image, f"ID: {marker['id']}", (center[0] + 20, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        # 3. Підготовка зображення для SDL2
        # Конвертуємо BGR (OpenCV) -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Масштабуємо під розмір вікна SDL
        image_resized = cv2.resize(image_rgb, (window_width, window_height))

        # Створюємо поверхню SDL з масиву numpy
        # Вказуємо pitch (ширина * кількість каналів)
        surface = sdl2.SDL_CreateRGBSurfaceFrom(
            image_resized.ctypes.data,
            window_width,
            window_height,
            24, # Depth
            window_width * 3, # Pitch
            0xff0000, 0x00ff00, 0x0000ff, 0 # RGBA Masks
        )

        # 4. Рендеринг
        texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, surface)
        sdl2.SDL_FreeSurface(surface) # Очищуємо пам'ять поверхні

        renderer.clear()
        sdl2.SDL_RenderCopy(renderer.sdlrenderer, texture, None, None)
        renderer.present()
        
        sdl2.SDL_DestroyTexture(texture) # Очищуємо текстуру після кадру

        # FPS calculation (опціонально в консоль)
        curr = time.time()
        fps = 1.0 / (curr - prev_time)
        prev_time = curr

    cap.release()
    sdl2.ext.quit()