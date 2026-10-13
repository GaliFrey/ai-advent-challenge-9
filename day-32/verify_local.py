"""Real local embeddings and a test-confirmed demo bug, without any LLM call."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from rag import load_model
from review import run, save
from snapshot import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'resources/local-check')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='day32-demo-') as directory:
        root = Path(directory) / 'guitar-trainer'
        subprocess.run(['git', 'clone', '-q', '--no-hardlinks', str(args.project.resolve()), str(root)], check=True)
        def git(*options):
            return subprocess.check_output(['git', '-C', str(root), *options], stderr=subprocess.DEVNULL).decode().strip()
        git('config', 'user.name', 'Day 32 local fixture')
        git('config', 'user.email', 'fixture@example.invalid')
        base = git('rev-parse', 'HEAD')
        target = root / 'src/pitch-match.js'
        original = target.read_text()
        before = 'if (!matching) { since = null; latched = false; }'
        after = 'if (!matching) { latched = false; }'
        if original.count(before) != 1:
            raise ValueError('Исходник изменился; демонстрационная замена больше неприменима.')
        # First confirm this exact baseline test is green; only then insert bug.
        command = ['node', '--test', 'test/pitch-match.test.js']
        baseline = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if baseline.returncode:
            raise RuntimeError('Базовый pitch-match.test.js не прошёл.')
        target.write_text(original.replace(before, after))
        git('add', 'src/pitch-match.js')
        git('commit', '-qm', 'demo: fail to clear interrupted pitch hold')
        head = git('rev-parse', 'HEAD')
        broken = subprocess.run(command, cwd=root, capture_output=True, text=True)
        (output / 'baseline-test.txt').write_text(baseline.stdout + baseline.stderr)
        (output / 'bug-test.txt').write_text(broken.stdout + broken.stderr)
        if broken.returncode == 0:
            raise RuntimeError('Демонстрационный баг не подтверждён тестом.')
        snapshot = collect(root, base, head)
        result = run(snapshot, load_model(), retrieve_only=True)
        save(result, output)
        stats = {'baseline_test_exit': baseline.returncode, 'bug_test_exit': broken.returncode,
                 'base': base, 'head': head, 'rag': result['rag'], 'llm_calls': 0}
        (output / 'verification.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
