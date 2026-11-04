"""Portable training paths; optional local.py overrides these defaults."""
import importlib
from pathlib import Path


class EnvironmentSettings:
    def __init__(self, workspace_dir=None, data_dir=None):
        root = Path(__file__).resolve().parents[3]
        workspace = Path(workspace_dir or root).resolve()
        data = Path(data_dir or root / 'data').resolve()
        self.workspace_dir = str(workspace)
        self.tensorboard_dir = str(workspace / 'tensorboard')
        self.pretrained_networks = str(root / 'pretrained')
        folders = {'lasot': 'lasot', 'got10k': 'got10k/train', 'trackingnet': 'trackingnet',
                   'coco': 'coco', 'vasttrack': 'vasttrack', 'depthtrack': 'depthtrack/train',
                   'lasher': 'lasher/trainingset', 'visevent': 'visevent/train',
                   'tnl2k': 'tnl2k_train', 'otb99': 'otb_lang'}
        for name, folder in folders.items():
            setattr(self, name + '_dir', str(data / folder))
        for name in ['lasot', 'got10k', 'coco', 'trackingnet']:
            setattr(self, name + '_lmdb_dir', str(data / (name + '_lmdb')))


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = Path(__file__).with_name('local.py')
    path.write_text('from .environment import EnvironmentSettings as Defaults\n\n'
                    'class EnvironmentSettings(Defaults):\n'
                    '    def __init__(self):\n'
                    f'        super().__init__({str(workspace_dir)!r}, {str(data_dir)!r})\n')


def env_settings():
    try:
        module = importlib.import_module('lib.train.admin.local')
    except ModuleNotFoundError as exc:
        if exc.name != 'lib.train.admin.local':
            raise
        return EnvironmentSettings()
    return module.EnvironmentSettings()
