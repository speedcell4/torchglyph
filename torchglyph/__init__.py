from pathlib import Path

data_dir = (Path.home() / '.torchglyph').expanduser().absolute()
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
