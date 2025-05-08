import os
import platform
import socket
from matplotlib import pyplot as plt
from pathlib import Path

data_dir = (Path.home() / '.cache' / 'torchglyph').resolve()
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

host_name: str = socket.gethostname()
system_name: str = platform.system().lower()

if system_name != 'darwin':
    plt.switch_backend('agg')

DEBUG = os.environ.get('DEBUG', f"{system_name == 'darwin'}").lower() in ('1', 'y', 'yes', 't', 'true')

CHECK = os.environ.get('CHECK', '').strip().lower().split(',')
CHECK_LOG = 'log' in CHECK
CHECK_DATA = 'data' in CHECK
CHECK_EVAL = 'eval' in CHECK
