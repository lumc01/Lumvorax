import ast
import hashlib
from pathlib import Path

TARGET = Path('nx47_vesu_kernel_v2.py')


def main() -> int:
    if not TARGET.exists():
        print(f'ERROR: missing {TARGET}')
        return 2

    raw = TARGET.read_bytes()
    text = raw.decode('utf-8')
    tabs = text.count('\t')
    lines = text.splitlines()
    fingerprint = hashlib.sha256(raw).hexdigest()

    try:
        ast.parse(text)
        ast_ok = True
        ast_error = ''
    except SyntaxError as e:
        ast_ok = False
        ast_error = f'{type(e).__name__}: {e}'

    print('NX47_SOURCE_INTEGRITY')
    print(f'path={TARGET}')
    print(f'bytes={len(raw)}')
    print(f'lines={len(lines)}')
    print(f'tabs={tabs}')
    print(f'sha256={fingerprint}')
    print(f'ast_ok={ast_ok}')
    if ast_error:
        print(f'ast_error={ast_error}')

    return 0 if ast_ok and tabs == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
