from __future__ import annotations

from pathlib import Path


def sync_repo(repo_url: str, repo_dir: Path):
    from git import Repo

    if repo_dir.exists():
        repo = Repo(repo_dir)
        repo.remotes.origin.fetch()
        repo.git.checkout("main")
        repo.remotes.origin.pull("main")
        return repo

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    return Repo.clone_from(repo_url, repo_dir)
