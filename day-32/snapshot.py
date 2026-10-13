"""Read immutable Git objects; never execute files from the reviewed project."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess

MAX_FILE = 250_000
MAX_CORPUS = 2_000_000
MAX_DIFF = 80_000
MAX_DIRECT = 140_000
MAX_CHANGED = 20
HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def git(project, *args):
    result = subprocess.run(['git', '--no-pager', '-C', str(project), *args],
                            capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError('Git не смог прочитать коммиты. Нужна полная история base/head.')
    return result.stdout


def revision(project, value):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', value):
        raise ValueError('Base и head должны быть полными 40-символьными SHA коммитов.')
    return git(project, 'rev-parse', '--verify', value + '^{commit}').decode().strip()


def safe_path(path):
    parts = PurePosixPath(path).parts
    if (not parts or path.startswith('/') or '..' in parts
            or any(ord(c) < 32 or ord(c) == 127 for c in path)):
        raise ValueError('Недопустимый путь в Git.')
    return path


def secret_path(path):
    name = PurePosixPath(path).name.lower()
    return (name == '.env' or (name.startswith('.env.') and name != '.env.example')
            or name.endswith(('.pem', '.key', '.p12', '.pfx'))
            or name in {'credentials', 'id_rsa', 'id_ed25519'})


@dataclass(frozen=True)
class Source:
    id: str
    path: str
    revision: str
    kind: str
    start: int
    end: int
    text: str

    def payload(self):
        return vars(self)


def tree(project, sha):
    entries = {}
    for record in git(project, 'ls-tree', '-r', '-z', sha).split(b'\0'):
        if record:
            metadata, raw_path = record.split(b'\t', 1)
            mode, kind, oid = metadata.decode().split()
            path = safe_path(raw_path.decode('utf-8'))
            entries[path] = (mode, kind, oid)
    return entries


def read_blob(project, entry):
    mode, kind, oid = entry
    if kind != 'blob' or mode not in {'100644', '100755'}:
        raise ValueError('Символические ссылки и submodules не поддерживаются в корпусе/изменениях.')
    size = int(git(project, 'cat-file', '-s', oid))
    if size > MAX_FILE:
        raise ValueError('Файл больше лимита 250 КБ; ревью не выполнено.')
    raw = git(project, 'cat-file', 'blob', oid)
    if b'\0' in raw:
        return None
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return None


def kind_for(path):
    parts = PurePosixPath(path).parts
    if any(part.startswith('.') for part in parts) or secret_path(path):
        return None
    if path == 'README.md' or (parts[0] == 'docs' and path.endswith('.md')):
        return 'docs'
    if ((parts[0] in {'src', 'test', 'scripts'} and path.endswith(('.js', '.ts', '.py')))
            or path in {'index.html', 'styles.css', 'package.json'}):
        return 'code'
    return None


def changed_lines(diff):
    locations = {'base': set(), 'head': set()}
    old = new = None
    for line in diff.splitlines():
        match = HUNK.match(line)
        if match:
            old, new = int(match[1]), int(match[3])
        elif old is not None:
            if line.startswith('-'):
                locations['base'].add(old)
                old += 1
            elif line.startswith('+'):
                locations['head'].add(new)
                new += 1
            elif line.startswith(' '):
                old += 1
                new += 1
    return locations


def collect(project, base, head):
    project = Path(project).resolve()
    base, head = revision(project, base), revision(project, head)
    ancestor = git(project, 'merge-base', base, head).decode().strip()
    before, after = tree(project, ancestor), tree(project, head)
    raw = git(project, 'diff', '--no-ext-diff', '--no-textconv', '--no-renames',
              '--name-status', '-z', ancestor, head).split(b'\0')
    changes = []
    for i in range(0, len(raw) - 1, 2):
        status, path = raw[i].decode(), safe_path(raw[i + 1].decode())
        if secret_path(path):
            raise ValueError('PR меняет файл секретов; он не будет передан LLM.')
        changes.append({'status': status, 'path': path})
    if not changes or len(changes) > MAX_CHANGED:
        raise ValueError('Нужен непустой PR с максимум 20 изменёнными файлами.')
    direct, total, diff_size = [], 0, 0
    for change in changes:
        path = change['path']
        patch = git(project, '-c', 'core.quotePath=false', 'diff', '--no-ext-diff',
                    '--no-textconv', '--no-renames', '--unified=3', ancestor, head,
                    '--', path).decode('utf-8', errors='replace')
        diff_size += len(patch.encode())
        if diff_size > MAX_DIFF:
            raise ValueError('Diff больше 80 КБ; автоматическое ревью остановлено без усечения.')
        change['diff'] = patch
        change['lines'] = changed_lines(patch)
        change['sources'] = {}
        for side, entries, sha in [('base', before, ancestor), ('head', after, head)]:
            if path not in entries:
                continue
            text = read_blob(project, entries[path])
            if text is None:
                change.setdefault('skipped', []).append(side + ': binary/non-UTF8')
                continue
            total += len(text.encode())
            if total > MAX_DIRECT:
                raise ValueError('Изменённые файлы base/head больше 140 КБ; ревью остановлено.')
            source = Source(f'C{len(direct) + 1}', path, sha, side, 1,
                            len(text.splitlines()), text)
            direct.append(source)
            change['sources'][side] = source.id
    corpus, corpus_size = [], 0
    for path, entry in sorted(after.items()):
        kind = kind_for(path)
        if kind is None:
            continue
        text = read_blob(project, entry)
        if text is None:
            raise ValueError('Не-UTF8 или бинарный файл внутри текстового RAG-корпуса.')
        corpus_size += len(text.encode())
        if corpus_size > MAX_CORPUS or len(corpus) >= 500:
            raise ValueError('Корпус больше 2 МБ или 500 файлов.')
        corpus.append(Source('', path, head, kind, 1, len(text.splitlines()), text))
    if not any(s.kind == 'docs' for s in corpus) or not any(s.kind == 'code' for s in corpus):
        raise ValueError('Для RAG нужны документация и код в head-коммите.')
    return {'base': base, 'merge_base': ancestor, 'head': head,
            'changes': changes, 'direct': direct, 'corpus': corpus}
