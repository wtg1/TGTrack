"""Configure paths for TGTrack training and evaluation."""
import argparse
from pathlib import Path
import _init_paths
from lib.train.admin.environment import create_default_local_file_ITP_train
from lib.test.evaluation.environment import create_default_local_file_ITP_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace_dir', default='.')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--save_dir', default='.')
    args = parser.parse_args()
    workspace, data, save = (str(Path(p).expanduser().resolve())
                             for p in (args.workspace_dir, args.data_dir, args.save_dir))
    create_default_local_file_ITP_train(save, data)
    create_default_local_file_ITP_test(workspace, data, save)
    print('Configured TGTrack training and evaluation paths.')


if __name__ == '__main__':
    main()
