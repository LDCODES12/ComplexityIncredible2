# test_metal.py
import platform

# Verify we're on Apple Silicon
is_apple_silicon = (
    platform.system() == "Darwin" and
    platform.processor() == "arm" and
    platform.machine() == "arm64"
)

print(f"Running on Apple Silicon: {is_apple_silicon}")

# Try to import Metal
try:
    import Metal
    import Foundation
    device = Metal.MTLCreateSystemDefaultDevice()
    if device:
        print(f"Metal working! Device name: {device.name()}")
        print(f"Metal version: {device.supportsFamily_(Metal.MTLGPUFamilyApple7)}")
    else:
        print("Metal device not available")
except ImportError as e:
    print(f"Metal import error: {e}")