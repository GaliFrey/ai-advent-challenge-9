"""CLI: index, /help, /help question, /exit. No dialogue history."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from assistant import ask, save_result
from index import DAY, DEFAULT_PROJECT, build, index_path, load_model, open_fresh

OVERVIEW = "Как устроен проект: какие основные каталоги и компоненты и за что они отвечают?"


def question_from(command):
    parts = command.strip().split(maxsplit=1)
    if not parts or parts[0] != "/help":
        raise ValueError("Используйте /help, /help <вопрос> или /exit.")
    return parts[1] if len(parts) == 2 else OVERVIEW


def show(result, *, details=False):
    print()
    if result["status"] == "accepted":
        print(result["answer"]["answer"])
        by_id = {source["id"]: source for source in result["sources"]}
        shown = set()
        for citation in result["answer"]["citations"]:
            identity = citation["source"]
            if identity in shown:
                continue
            shown.add(identity)
            source = by_id[identity]
            print(f"\n[{identity}] Источник: {source['source']} · {source['section']}")
            excerpt = " ".join(citation["quote"].split())
            if len(excerpt) > 160:
                excerpt = excerpt[:157].rstrip() + "…"
            print(f"Фрагмент чанка: {excerpt}")
    elif result["status"] == "rejected":
        print(f"Ответ отклонён: {result['error']}")
    else:
        print("Только поиск: DeepSeek не вызывался.")
    context = result["mcp"]["context"]
    print(f"\nMCP · ветка: {context['branch']} "
          f"· HEAD: {context['head'][:12]}")
    if not details:
        return
    print(f"Сервер: {result['mcp']['trace']['server']}")
    print(f"Приложение: {context['project']} · репозиторий: {context['repository']} "
          f"· путь: {context['project_path']}")
    files = result["mcp"]["files"]
    print(f"MCP list_project_files: {len(files['files'])}/{files['total']} файлов"
          + (" (список усечён)" if files["truncated"] else ""))
    for path in files["files"]:
        print(f"  {path}")
    print("\nRAG — найденные фрагменты:")
    for source in result["sources"]:
        if source["id"].startswith("S"):
            print(f"[{source['id']}] {source['source']} · {source['section']} "
                  f"· score={source['score']}")
            print(source["text"])
    if result["status"] == "accepted":
        print("\nПроверенные цитаты:")
        for citation in result["answer"]["citations"]:
            print(f"[{citation['source']}] Цитата: {citation['quote']}")


def main(argv=None):
    load_dotenv(DAY / ".env", override=False)
    parser = argparse.ArgumentParser(description="День 31: RAG по README/docs и Git через MCP")
    parser.add_argument("command", nargs="?", help="'/help', '/help <вопрос>' или '/exit'; без аргумента — цикл ввода")
    parser.add_argument("--project", type=Path,
                        default=Path(os.getenv("DAY31_PROJECT") or DEFAULT_PROJECT),
                        help="Каталог приложения (по умолчанию соседний guitar-trainer)")
    parser.add_argument("--index", action="store_true", help="Пересобрать локальный индекс без вызова DeepSeek")
    parser.add_argument("--retrieve-only", action="store_true", help="Показать RAG/MCP без облачного запроса")
    parser.add_argument("--details", action="store_true", help="Показать найденные тексты, список файлов и проверенные цитаты")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    path = index_path(project)
    model = None
    try:
        if args.index:
            info = build(project, path, load_model())
            print(f"Индекс: {info['documents']} документов, {info['chunks']} чанков\n{path}")
            return 0
        if args.command == "/exit":
            return 0
        if args.command is not None:
            question_from(args.command)
        with closing(open_fresh(project, path)):
            pass
        key = os.getenv("DEEPSEEK_API_KEY", "")
        llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        if not args.retrieve_only and not key.strip():
            raise ValueError("Задайте DEEPSEEK_API_KEY в day-31/.env или окружении.")
        print(f"Проект: {project}\nКоманды: /help, /help <вопрос>, /exit")
        while True:
            if args.command is not None:
                command = args.command
            else:
                try:
                    command = input("\n> ").strip()
                except EOFError:
                    return 0
            if command == "/exit":
                return 0
            if not command:
                continue
            try:
                question = question_from(command)
                if model is None:
                    print("Загрузка локальной модели эмбеддингов…")
                    model = load_model()
                print("Поиск документации и чтение MCP…")
                result = ask(question, project, path, model, key, llm_model,
                             retrieve_only=args.retrieve_only)
                saved = save_result(result)
                show(result, details=args.details or args.retrieve_only)
                if args.details or args.retrieve_only or result["status"] == "rejected":
                    print(f"\nДиагностика: {saved}")
                if args.command is not None:
                    return int(result["status"] == "rejected")
            except (ValueError, RuntimeError, OSError) as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                if args.command is not None:
                    return 1
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nВыход.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
