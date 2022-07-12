from torchglyph.proc.abc import Proc


class ToLower(Proc):
    def __call__(self, data: str, **kwargs) -> str:
        return data.lower()


class ToUpper(Proc):
    def __call__(self, data: str, **kwargs) -> str:
        return data.upper()


class ToCapitalize(Proc):
    def __call__(self, data: str, **kwargs) -> str:
        return data.capitalize()
