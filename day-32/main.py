"""CLI shared by local checks and GitHub Actions."""
import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from rag import load_model
from review import run, save
from snapshot import collect


def main():
    parser = argparse.ArgumentParser(description='AI PR review: Git snapshot + code/docs RAG + DeepSeek')
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'resources' / 'review')
    parser.add_argument('--retrieve-only', action='store_true', help='RAG без платного API-вызова')
    args = parser.parse_args()
    # Only load this tool's local .env, never the reviewed repository's .env.
    load_dotenv(Path(__file__).with_name('.env'))
    try:
        snapshot = collect(args.project, args.base, args.head)
        if not args.retrieve_only and not os.getenv('DEEPSEEK_API_KEY', '').strip():
            raise ValueError('Не задан DEEPSEEK_API_KEY; запрос не выполнен.')
        result = run(snapshot, load_model(), os.getenv('DEEPSEEK_API_KEY', ''),
                     os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'), args.retrieve_only)
        save(result, args.output)
        print(f"Статус: {result['status']}; отчёт: {args.output.resolve() / 'review.md'}")
        if result['status'] == 'failed':
            print('Причина: ' + result.get('error', 'неизвестная ошибка'), file=sys.stderr)
        return 0 if result['status'] in {'accepted', 'retrieved'} else 1
    except (ValueError, RuntimeError, OSError) as error:
        # Preflight/model failures must not leave an old successful report.
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'review.json').unlink(missing_ok=True)
        import html
        (args.output / 'review.md').write_text('# AI-ревью не выполнено\n\n' + html.escape(str(error)) + '\n', encoding='utf-8')
        print('Ревью не выполнено: ' + str(error) + '; см. review.md.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
