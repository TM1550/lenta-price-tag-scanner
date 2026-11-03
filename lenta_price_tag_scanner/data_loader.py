"""Explicit, fail-fast DVC materialization and public-source bootstrap."""

import subprocess
from pathlib import Path


def repository_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / ".dvc").is_dir():
            return candidate
    raise FileNotFoundError("Run from a DVC checkout or disable pull_dvc for explicit local files")


def download_data_with_dvc(targets, enabled=True):
    if not enabled:
        return
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        raise ValueError("Specify DVC targets; refusing to silently skip data retrieval")
    from dvc.repo import Repo

    with Repo(str(repository_root())) as repository:
        repository.pull(targets=list(targets))


def download_data(url, output, folder=False):
    """Fetch public originals; prepared/relabelled data remain separate DVC artifacts."""
    import gdown

    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if folder:
        result = gdown.download_folder(url, output=str(destination), quiet=False)
    else:
        result = gdown.download(url, output=str(destination) + "/", fuzzy=True, quiet=False)
    if not result:
        raise RuntimeError("Public download failed; check URL and access")
    return result


def code_version():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode:
        return {"git_commit": "uncommitted", "git_dirty": True}
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"git_commit": result.stdout.strip(), "git_dirty": bool(status.stdout.strip())}
