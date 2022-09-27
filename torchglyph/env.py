import subprocess
from datetime import datetime
from uuid import uuid4


def git_describe(*, n: int = 10) -> str:
    try:
        version = subprocess.check_output(f'git describe --tags --dirty --abbrev={n} --always', shell=True)
        return str(version, encoding='utf-8').strip()
    except subprocess.CalledProcessError:
        return 'untracked'


def datetime_now(*, fmt: str = r'%y%m%d-%H%M%S') -> str:
    return datetime.strftime(datetime.now(), fmt).strip()


def uuid(*, g: int = 10, u: int = 8) -> str:
    return f'{git_describe(n=g)}-{datetime_now()}-{uuid4().hex[:u]}'
