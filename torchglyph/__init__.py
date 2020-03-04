from pathlib import Path

data_path = Path.home() / '.torchglyph'
if not data_path.exists():
    data_path.mkdir(parents=True, exist_ok=True)
