"""Portable benchmark and output paths for TGTrack."""
import importlib
from pathlib import Path


class EnvSettings:
    def __init__(self, workspace_dir=None, data_dir=None, save_dir=None):
        root = Path(workspace_dir or Path(__file__).resolve().parents[3]).resolve()
        data = Path(data_dir or root / 'data').resolve()
        save = Path(save_dir or root).resolve()
        output = save / 'test'
        self.prj_dir = str(root)
        self.save_dir = str(save)
        self.results_path = str(output / 'tracking_results')
        self.segmentation_path = str(output / 'segmentation_results')
        self.network_path = str(save / 'checkpoints')
        self.result_plot_path = str(output / 'result_plots')
        for name in ['got10k', 'lasot', 'lasot_extension_subset', 'trackingnet', 'tnl2k',
                     'depthtrack', 'lasher', 'visevent', 'lasot_lmdb', 'got10k_lmdb']:
            setattr(self, name + '_path', str(data / name))
        self.otblang_path = str(data / 'otb_lang')
        self.rgbt234_path = str(data / 'RGB-T234')
        self.got_packed_results_path = str(output / 'submissions/got10k')
        self.got_reports_path = str(output / 'reports/got10k')
        self.tn_packed_results_path = str(output / 'submissions/trackingnet')


def create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir):
    Path(__file__).with_name('local.py').write_text(
        'from .environment import EnvSettings\n\n'
        'def local_env_settings():\n'
        f'    return EnvSettings({str(workspace_dir)!r}, {str(data_dir)!r}, {str(save_dir)!r})\n')


def env_settings():
    try:
        module = importlib.import_module('lib.test.evaluation.local')
    except ModuleNotFoundError as exc:
        if exc.name != 'lib.test.evaluation.local':
            raise
        return EnvSettings()
    return module.local_env_settings()
