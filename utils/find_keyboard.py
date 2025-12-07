from evdev import InputDevice, categorize, ecodes, list_devices

def find_keyboard_by_name(n="SayoDevice SayoDevice 1x7 RGB"):
    devices = [InputDevice(path) for path in list_devices()]
    
    for dev in devices:
        caps = dev.capabilities()
        
        
        if ecodes.EV_KEY in caps:
            keys = caps[ecodes.EV_KEY]
            if dev.name == n:
                print("🔍 Keyboard is found:")
                print(f"Path: {dev.path}")
                print(f"Name: {dev.name}")
                print(f"Keys: {keys}")
                return dev
    
    print("Keyboard is not found")
    return None

