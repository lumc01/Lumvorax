import ctypes
import os

_lib_paths = [
    "/lib/x86_64-linux-gnu/libstdc++.so.6",
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
]
for _p in _lib_paths:
    if os.path.exists(_p):
        ctypes.CDLL(_p)
        break


def main():
    print("Hello from repl-nix-workspace!")


if __name__ == "__main__":
    main()
