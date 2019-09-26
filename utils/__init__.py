try:
    from .utils import ddict, get_devices
    from .datasets import load_dataset

except ImportError as e:
    import sys
    print('''Could not import submodules (exact error was: %s).''' % e, file=sys.stderr)


__all__ = [
    'ddict', 'get_devices', 'load_dataset'
]
