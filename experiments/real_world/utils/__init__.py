from typing import Union, Optional
from pathlib import Path
import sys
import shutil


def mkdir(path: Path, resume=False, overwrite=False) -> None:

    while True:
        if overwrite:
            if path.is_dir():
                print('overwriting directory ({})'.format(path))
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            return
        elif resume:
            print('resuming directory ({})'.format(path))
            path.mkdir(parents=True, exist_ok=True)
            return
        else:
            if path.exists():
                feedback = input('target directory ({}) already exists, overwrite? [Y/r/n] '.format(path))
                ret = feedback.casefold()
            else:
                ret = 'y'
            if ret == 'n':
                sys.exit(0)
            elif ret == 'r':
                resume = True
            elif ret == 'y':
                overwrite = True


def get_root(path: Union[str, Path], name: str = '.root') -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root
