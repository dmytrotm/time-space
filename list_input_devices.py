#!/usr/bin/env python3
"""
Utility script to list all available input devices.
Run this to find the correct keyboard device name.
"""
from evdev import InputDevice, ecodes, list_devices

print("Available input devices:")
print("=" * 60)

devices = [InputDevice(path) for path in list_devices()]

for dev in devices:
    caps = dev.capabilities()
    print(f"\nPath: {dev.path}")
    print(f"Name: {dev.name}")
    print(f"Physical: {dev.phys}")
    
    # Check if it has keyboard capabilities
    if ecodes.EV_KEY in caps:
        keys = caps[ecodes.EV_KEY]
        print(f"✓ Has keyboard capabilities ({len(keys)} keys)")
    else:
        print("✗ No keyboard capabilities")
    
    print("-" * 60)

print("\nTo use a specific device, update the default name in find_keyboard.py")
print("Or pass the device name to find_keyboard_by_name(name='...')")
