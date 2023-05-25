import logging
import sys
from logging import getLogger
from pathlib import Path

import colorlog

from torchglyph import DEBUG

logger = getLogger(__name__)

LOG_TXT = 'log.txt'


def clear_root(*, level: int) -> None:
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
        handler.close()

    return logging.root.setLevel(level=level)


def add_stream_handler(*, level: int, fmt: str) -> None:
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(colorlog.ColoredFormatter(
        fmt='%(log_color)s' + fmt,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'bold_red',
            'ERROR': 'bold_orange',
            'CRITICAL': 'bold_purple',
        },
    ))
    stream_handler.setLevel(level=level)

    return logging.root.addHandler(hdlr=stream_handler)


def add_file_handler(*, out_dir: Path, level: int, fmt: str) -> None:
    file_handler = logging.FileHandler(filename=str(out_dir / LOG_TXT), mode='w', encoding='utf-8')
    file_handler.setFormatter(fmt=logging.Formatter(fmt=fmt))
    file_handler.setLevel(level=level)

    return logging.root.addHandler(hdlr=file_handler)


def init_logger(*, out_dir: Path, rank: int) -> None:
    if rank == 0:
        level = logging.DEBUG if DEBUG else logging.INFO
    else:
        level = logging.DEBUG if DEBUG else logging.WARNING

    clear_root(level=level)

    fmt = '%(asctime)s [{rank}-%(levelname)-s] %(name)s | %(message)s'.format(rank=rank)
    add_stream_handler(level=level, fmt=fmt)
    add_file_handler(out_dir=out_dir, level=level, fmt=fmt)

    return logger.info(' '.join(sys.argv))
