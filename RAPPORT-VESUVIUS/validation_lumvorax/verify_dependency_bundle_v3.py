import os
import sys
import argparse

def verify(directory):
    required = [
        "imagecodecs-2026.1.14-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "imageio-2.37.2-py3-none-any.whl",
        "lazy_loader-0.4-py3-none-any.whl",
        "liblumvorax_replit.so",
        "networkx-3.6.1-py3-none-any.whl",
        "numpy-2.4.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "opencv_python-4.13.0.92-cp37-abi3-manylinux_2_28_x86_64.whl",
        "packaging-26.0-py3-none-any.whl",
        "pillow-12.1.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "scikit_image-0.26.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "scipy-1.17.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "tifffile-2026.1.28-py3-none-any.whl",
        "tifffile-2026.2.16-py3-none-any.whl"
    ]
    files = os.listdir(directory)
    missing = [f for f in required if f not in files]
    if missing:
        print(f"Missing files: {missing}")
        return False
    print("All files present.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    if verify(args.dir):
        sys.exit(0)
    else:
        sys.exit(1)
