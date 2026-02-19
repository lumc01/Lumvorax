import ctypes
import os
import numpy as np

def test_ctypes_loading():
    lib_path = os.path.abspath("liblumvorax_replit.so")
    if not os.path.exists(lib_path):
        print(f"ERROR: {lib_path} not found")
        return False
    
    try:
        lib = ctypes.CDLL(lib_path)
        print(f"SUCCESS: {lib_path} loaded via ctypes")
        
        # Test existence of 3D volume functions
        functions = [
            "vorax_volume3d_validate",
            "vorax_volume3d_normalize",
            "vorax_volume3d_threshold"
        ]
        
        for func in functions:
            if hasattr(lib, func):
                print(f"SUCCESS: Function {func} found in library")
            else:
                print(f"ERROR: Function {func} NOT found")
                return False
        return True
    except Exception as e:
        print(f"ERROR loading library: {e}")
        return False

if __name__ == "__main__":
    success = test_ctypes_loading()
    exit(0 if success else 1)
