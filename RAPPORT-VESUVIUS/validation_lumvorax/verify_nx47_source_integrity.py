import os
import ast
import hashlib

def verify():
    path = 'nx47_vesu_kernel_v2.py'
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    
    with open(path, 'rb') as f:
        content_bin = f.read()
    
    content_str = content_bin.decode('utf-8')
    sha256 = hashlib.sha256(content_bin).hexdigest()
    tabs = content_str.count('\t')
    
    try:
        ast.parse(content_str)
        ast_ok = True
    except SyntaxError:
        ast_ok = False
        
    print(f"sha256: {sha256}")
    print(f"tabs: {tabs}")
    print(f"ast_ok: {ast_ok}")

if __name__ == "__main__":
    verify()
