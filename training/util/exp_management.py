import subprocess
import nni


def get_git_info() -> (str, str, bool):
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    git_clean = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip() == ''
    return git_branch, git_commit, git_clean


def get_param_from_nni():
    raw_params = nni.get_next_parameter()
