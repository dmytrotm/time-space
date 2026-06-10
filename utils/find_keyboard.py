try:
    from evdev import InputDevice, categorize, ecodes, list_devices
except ImportError:
    InputDevice = None
    categorize = None
    ecodes = None
    list_devices = lambda: []


def find_keyboards_by_name(n="SayoDevice"):
    devices = [InputDevice(path) for path in list_devices()]
    matched = []
    
    for dev in devices:
        caps = dev.capabilities()
        if ecodes.EV_KEY in caps:
            if n in dev.name:
                print(f"Keyboard endpoint identified: {dev.path} ({dev.name})")
                matched.append(dev)
                
    if not matched:
        print("Keyboard is not found")
        
    return matched
