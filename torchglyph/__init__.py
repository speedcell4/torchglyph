from pathlib import Path

usr_data_path = (Path.home() / '.torchglyph').expanduser().absolute()
if not usr_data_path.exists():
    usr_data_path.mkdir(parents=True, exist_ok=True)
