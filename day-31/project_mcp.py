"""Read-only project tools. No arbitrary commands or model-supplied paths."""
import json
import os
from pathlib import Path
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day31-project")


def git(project, *args):
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(project), *args],
                                stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Git недоступен или превысил таймаут.") from None
    if result.returncode:
        raise ValueError("Не удалось прочитать Git-контекст проекта.")
    return result.stdout


def project_root():
    root = Path(os.environ["DAY31_PROJECT"]).resolve()
    if not root.is_dir():
        raise ValueError("Каталог проекта отсутствует.")
    return root


def context_for(project):
    project = Path(project).resolve()
    repository = Path(git(project, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    branch = git(project, "branch", "--show-current").decode().strip()
    return {"project": project.name, "repository": repository.name,
            "project_path": project.relative_to(repository).as_posix(),
            "branch": branch or "(detached HEAD)",
            "head": git(project, "rev-parse", "HEAD").decode().strip(),
            "detached": not bool(branch)}


def files_for(project):
    project = Path(project).resolve()
    # ls-files is relative to -C project, including a project nested in a repo.
    files = git(project, "ls-files", "-z", "--", ".").decode("utf-8").split("\0")
    files = sorted(path for path in files if path and not path.startswith("../"))
    return {"files": files[:500], "total": len(files), "truncated": len(files) > 500}


@mcp.tool()
def get_project_context() -> str:
    """Получить текущую Git-ветку и HEAD репозитория выбранного приложения."""
    return json.dumps(context_for(project_root()), ensure_ascii=False)


@mcp.tool()
def list_project_files() -> str:
    """Получить до 500 отслеживаемых Git-файлов выбранного приложения."""
    return json.dumps(files_for(project_root()), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
