"""Full-screen adaptive Textual chat, task memory and inspectable retrieval."""
import copy
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.text import Text
from tqdm import tqdm
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, Select, Static, TabbedContent, TabPane, TextArea

from evidence import answer_text
from memory import state_text
from pipeline import DAY, Runner, Settings, new_session, load_session

STAGES = {'plan': 'DeepSeek: память и поисковый вопрос', 'index': 'Проверка индекса',
          'embedding_model': 'Загрузка эмбеддингов', 'reranker_model': 'Загрузка reranker',
          'search': 'Поиск по базе', 'rerank': 'Реранкинг', 'answer': 'DeepSeek: ответ с цитатами',
          'validate': 'Проверка данных', 'filter': 'Отбор источников по порогу',
          'judge': 'DeepSeek: оценка сценария', 'save': 'Сохранение',
          'ready': 'Готов'}


class ChatInput(TextArea):
    BINDINGS = [Binding('enter', 'submit', 'Отправить', show=False, priority=True),
                Binding('shift+enter', 'newline', 'Новая строка', show=False, priority=True)]

    def action_submit(self):
        self.app.action_send()

    def action_newline(self):
        self.insert('\n')


class ChatApp(App):
    TITLE = 'День 25 · RAG-чат и память задачи'
    CSS_PATH = 'tui.tcss'
    BINDINGS = [('ctrl+q', 'quit', 'Выход'), ('ctrl+enter', 'send', 'Отправить')]

    def __init__(self, session_path=None, *, runner_factory=Runner):
        # Retrieval runs in threads in this single-process app. Avoid tqdm's
        # lazy multiprocessing lock while Textual redirects terminal streams.
        tqdm.set_lock(RLock())
        super().__init__()
        load_dotenv(DAY / '.env', override=False)
        self.path = Path(session_path) if session_path else None
        self.session = load_session(self.path) if self.path else new_session()
        self.runner_factory, self.runner = runner_factory, None
        self.running = False
        self.stop = Event()
        self.selected = len(self.session['turns']) - 1
        self.chunk_rows = []
        self.stage, self.started = 'ready', None
        self.theme = 'textual-dark'
        self.replay = bool(self.session.get('validation_only'))

    def compose(self) -> ComposeResult:
        yield Static('ДЕНЬ 25  /  RAG + ДИАЛОГ + ПАМЯТЬ ЗАДАЧИ', id='brand')
        with Horizontal(id='toolbar'):
            yield Select([], prompt='Сохранённый диалог', id='sessions')
            yield Button('Открыть', id='open')
            yield Button('Новый', id='new')
            yield Button('Остановить', id='stop', disabled=True)
        yield Static('Одна реплика: до 2 вызовов DeepSeek', id='cost')
        yield Static(id='status')
        with TabbedContent(id='tabs'):
            with TabPane('Диалог', id='dialog-tab'):
                with Horizontal(id='workspace'):
                    with VerticalScroll(id='chat-scroll'):
                        yield Static(id='chat', markup=False)
                    with VerticalScroll(id='memory-scroll'):
                        yield Static(id='memory', markup=False)
            with TabPane('Источники и поиск', id='sources-tab'):
                yield Select([], prompt='Выберите реплику', id='turns')
                yield Static(id='query', markup=False)
                with Horizontal(id='retrieval'):
                    yield DataTable(id='chunks', cursor_type='row')
                    with VerticalScroll(id='chunk-scroll'):
                        yield Static(id='chunk', markup=False)
                with VerticalScroll(id='evidence-scroll'):
                    yield Static(id='evidence', markup=False)
            with TabPane('Память подробно', id='memory-tab'):
                with VerticalScroll():
                    yield Static(id='memory-detail', markup=False)
            with TabPane('Проверка сценария', id='checks-tab'):
                yield Static('Оценка LLM не гарантирует правильность. Клик или Enter на строке показывает оценку ниже.', id='assessment-note')
                yield Button('Источники выбранного хода', id='inspect-sources', disabled=True)
                yield DataTable(id='summary', cursor_type='row')
                with VerticalScroll(id='assessment-scroll'):
                    yield Static(id='assessment', markup=False)
            with TabPane('Настройки', id='settings-tab'):
                with VerticalScroll(id='settings-scroll'):
                    yield Static('Настройки применяются к следующим репликам; история сохраняет прежнюю выдачу.')
                    yield Static('Число кандидатов · K до реранкинга', classes='setting-title')
                    yield Input(value=str(self.session['settings']['top_k_before']), id='before')
                    yield Static('Общий максимум фрагментов, передаваемых reranker для оценки. '
                                 'Целое число от 1 до 100; по умолчанию 20. '
                                 'Для составного вопроса лимит делится между поисковыми темами. '
                                 'Больше кандидатов — больше работы для reranker.', classes='setting-help')
                    yield Static('Число фрагментов в ответе · итоговый K', classes='setting-title')
                    yield Input(value=str(self.session['settings']['top_k_after']), id='after')
                    yield Static('Максимум фрагментов, передаваемых модели после реранкинга и фильтрации. '
                                 'От 1 до числа кандидатов; по умолчанию 5. '
                                 'При сравнении сохраняется фрагмент каждой поисковой темы, если он прошёл порог и K достаточно. '
                                 'Если порог прошли меньше фрагментов, модель получит меньше.', classes='setting-help')
                    yield Static('Минимальная оценка reranker · порог', classes='setting-title')
                    yield Input(value=str(self.session['settings']['rerank_threshold']), id='threshold')
                    yield Static('Фрагменты с оценкой ниже порога исключаются из контекста. '
                                 'Число от 0 до 1; по умолчанию 0.10. Более высокий порог строже. '
                                 'Если ни один фрагмент не прошёл, чат откажется от ответа без генерации. '
                                 'Оценка не является вероятностью правильного ответа; порог 0.10 экспериментальный.',
                                 classes='setting-help')
        with Horizontal(id='composer'):
            yield ChatInput(placeholder='Введите сообщение. Enter — отправить, Shift+Enter — новая строка.', id='input')
            yield Button('Отправить', id='send', variant='primary')
        yield Footer()

    def on_mount(self):
        self.reset_chunk_table()
        self.query_one('#summary', DataTable).add_columns('Ход', 'Вопрос', 'Статус', 'Источники', 'Оценка LLM')
        self.refresh_sessions()
        self.render_session()
        self.set_interval(.5, self.tick)
        self.query_one('#input').focus()

    def on_resize(self, event):
        if not self.query('#memory-scroll'):
            return
        narrow = event.size.width < 120
        self.query_one('#memory-scroll').display = not narrow
        self.query_one('#chat-scroll').styles.width = '100%' if narrow else '2fr'
        if self.query('#chunks'):
            self.render_turn()

    def reset_chunk_table(self):
        table = self.query_one('#chunks', DataTable)
        table.clear(columns=True)
        # Two equal panes, with TabPane padding and room for the scrollbar.
        available = max(48, (self.size.width - 4) // 2)
        table.add_column('№', width=3)
        table.add_column('Источник / раздел', width=max(8, available - 41))
        table.add_column('Score', width=5)
        table.add_column('Отбор', width=24)
        return table

    def refresh_sessions(self):
        paths = sorted((DAY / 'sessions').glob('*.json'), reverse=True)
        options = []
        for path in paths:
            try:
                data = load_session(path)
                options.append((f"{data['title'][:45]} · {path.stem[-20:]}", str(path)))
            except (ValueError, KeyError, TypeError, OSError):
                continue
        self.query_one('#sessions', Select).set_options(options)

    def sync_settings(self):
        for widget, field in (('before', 'top_k_before'), ('after', 'top_k_after'),
                              ('threshold', 'rerank_threshold')):
            self.query_one('#' + widget, Input).value = str(self.session['settings'][field])

    def tick(self):
        if not self.query('#status'):
            return
        tokens = sum(out.get('usage', {}).get('total_tokens', 0) for t in self.session['turns']
                     for out in (t.get('plan_output', {}), t.get('answer_output', {}),
                                 t.get('assessment', {}).get('output', {})))
        elapsed = f' · {time.monotonic() - self.started:.0f} с' if self.running else ''
        key = '' if os.getenv('DEEPSEEK_API_KEY', '').strip() else ' · НЕТ КЛЮЧА'
        tag = ' · СИНТЕТИЧЕСКАЯ ЛОКАЛЬНАЯ ПРОВЕРКА' if self.session.get('validation_only') else ''
        self.query_one('#status', Static).update(Text(
            f"{self.session['title']} · {STAGES.get(self.stage, self.stage)}{elapsed}{key}{tag}\n"
            f"Реплик: {len(self.session['turns'])} · Токенов: {tokens} · Файл: {self.path.name if self.path else 'ещё не сохранён'}"))

    def render_session(self, select_latest=True):
        turns = self.session['turns']
        if select_latest or not 0 <= self.selected < len(turns):
            self.selected = len(turns) - 1
        parts = []
        for t in turns:
            answer = answer_text(t.get('response')) if t['status'] != 'running' else 'Обработка…'
            for source in (t.get('response') or {}).get('sources', []):
                answer = answer.replace(f"[{source['source_id']}]", f"[{t['number']}:{source['source_id']}]")
            parts += [f"ВЫ · {t['number']}\n{t['question']}", 'АССИСТЕНТ\n' + answer]
            sources = (t.get('response') or {}).get('sources', [])
            parts.append('ИСТОЧНИКИ\n' + ('\n'.join(
                f"[{t['number']}:{s['source_id']}] {s['source']} · {s['section']}" for s in sources)
                or ('Подтверждающие источники не найдены' if t.get('response') else 'Ответ ещё не принят')))
            if t['status'] in ('failed', 'cancelled', 'invalid'):
                parts.append(f"Статус: {t['status']} · {t.get('error', t.get('checks', {}))}")
        self.query_one('#chat', Static).update('\n\n'.join(parts) or 'Задайте цель и первый вопрос по README репозитория.')
        state = self.session['state']
        if 0 <= self.selected < len(turns):
            state = turns[self.selected].get('state_after', turns[self.selected]['state_before'])
        self.query_one('#memory', Static).update(state_text(state))
        details = state_text(state) + '\n\nПРОИСХОЖДЕНИЕ\n' + '\n\n'.join(
            f"Реплика {e['turn']} · {e['text']}\nЦитата пользователя: {e['quote']}"
            for entries in state.values() for e in entries)
        self.query_one('#memory-detail', Static).update(details)
        selector = self.query_one('#turns', Select)
        with selector.prevent(Select.Changed):
            selector.set_options([(f"{t['number']} · {t['question'][:100]}", i) for i, t in enumerate(turns)])
            if turns:
                selector.value = self.selected
        self.query_one('#inspect-sources', Button).disabled = not turns
        summary = self.query_one('#summary', DataTable)
        with summary.prevent(DataTable.RowHighlighted, DataTable.RowSelected):
            summary.clear()
            for i, t in enumerate(turns):
                judge = t.get('assessment', {})
                verdict = judge.get('result', {}).get('verdict', judge.get('status', '—'))
                summary.add_row(str(t['number']), t['question'][:90], t['status'],
                                str(len((t.get('response') or {}).get('sources', []))), verdict, key=str(i))
            if turns:
                summary.move_cursor(row=self.selected)
        self.render_turn()
        if select_latest:
            self.query_one('#chat-scroll', VerticalScroll).scroll_end(animate=False)
        self.tick()

    def render_turn(self):
        if not 0 <= self.selected < len(self.session['turns']):
            return
        t = self.session['turns'][self.selected]
        plan = t.get('plan', {})
        searches = t.get('searches', plan.get('searches', []))
        search_text = '\n'.join(f"Поиск {i}: {s['query']} · {s['source'] or 'весь корпус'}"
                                for i, s in enumerate(searches, 1)) or f"Поиск: {plan.get('query', '—')}"
        self.query_one('#query', Static).update(f"Реплика {t['number']}: {t['question']}\nСамостоятельный вопрос: {plan.get('resolved_question', '—')}\n{search_text}")
        self.chunk_rows = t.get('candidates', [])
        table = self.reset_chunk_table()
        for i, c in enumerate(self.chunk_rows):
            table.add_row(str(i + 1), Text(f"{c['source']}\n{c['section']}"),
                          f"{c['rerank_score']:.3f}", c['decision'], key=str(i), height=None)
        self.show_chunk(0)
        response = t.get('response') or {}
        quotes = []
        for claim in response.get('answer', []):
            quotes.append(claim['text'])
            for c in claim['citations']:
                source = next(s for s in response['sources'] if s['source_id'] == c['source_id'])
                quotes.append(f"[{t['number']}:{c['source_id']}] {source['source']} · {source['section']} · {source['chunk_id']}\n«{c['quote']}»")
        self.query_one('#evidence', Static).update('ЦИТАТЫ И ПРОВЕРКИ\n' + '\n\n'.join(quotes) + '\n' + str(t.get('checks', {})))
        judge = t.get('assessment', {})
        self.query_one('#assessment', Static).update('ОЦЕНКА LLM\n' + str(judge.get('result', judge.get('status', 'Не выполнялась'))) +
            '\n\nДиагностика: ' + str(t.get('error', {})) + '\n' + str(t.get('checks', {})))

    def show_chunk(self, index):
        if not self.chunk_rows:
            self.query_one('#chunk', Static).update('Нет найденных фрагментов')
            return
        c = self.chunk_rows[index]
        self.query_one('#chunk', Static).update(f"{c['source']}\n{c['section']} · {c['chunk_id']}\n\n{c['text']}")

    @on(DataTable.RowSelected, '#chunks')
    def chunk_selected(self, event):
        self.show_chunk(int(event.row_key.value))

    @on(Select.Changed, '#turns')
    def turn_changed(self, event):
        if not isinstance(event.value, int) or not 0 <= event.value < len(self.session['turns']):
            return
        self.select_turn(int(event.value))

    def select_turn(self, index):
        self.selected = index
        self.render_turn()
        t = self.session['turns'][self.selected]
        state = t.get('state_after', t['state_before'])
        self.query_one('#memory', Static).update(state_text(state))
        self.query_one('#memory-detail', Static).update(state_text(state) + '\n\n' + '\n\n'.join(
            f"Реплика {e['turn']}: {e['quote']}" for entries in state.values() for e in entries))

    @on(DataTable.RowSelected, '#summary')
    @on(DataTable.RowHighlighted, '#summary')
    def summary_selected(self, event):
        event.stop()
        if self.query_one('#tabs', TabbedContent).active != 'checks-tab' or event.row_key is None:
            return
        index = int(event.row_key.value)
        selector = self.query_one('#turns', Select)
        with selector.prevent(Select.Changed):
            selector.value = index
        self.select_turn(index)

    @on(Button.Pressed)
    def pressed(self, event):
        identity = event.button.id
        if identity == 'send':
            self.action_send()
        elif identity == 'inspect-sources':
            self.query_one('#tabs', TabbedContent).active = 'sources-tab'
            self.query_one('#turns', Select).focus()
        elif identity == 'stop':
            self.stop.set()
            self.stage = 'Остановка после текущего запроса'
        elif identity == 'new':
            self.session, self.path, self.selected = new_session(), None, -1
            self.replay = False
            self.runner = None
            self.sync_settings()
            self.query_one('#input', TextArea).load_text('')
            self.query_one('#chunks', DataTable).clear()
            for name in ('query', 'chunk', 'evidence', 'assessment'):
                self.query_one('#' + name, Static).update('')
            self.render_session()
        elif identity == 'open':
            value = self.query_one('#sessions', Select).value
            if isinstance(value, str):
                try:
                    self.path = Path(value)
                    self.session = load_session(self.path)
                    self.replay = bool(self.session.get('validation_only'))
                    self.runner = None
                    self.sync_settings()
                    self.render_session()
                except Exception as error:
                    self.notify(f'Не удалось открыть: {type(error).__name__}', severity='error')

    def action_send(self):
        question = self.query_one('#input', TextArea).text.strip()
        if question:
            if self.replay:
                self.notify('Для живого общения создайте новый диалог: этот отчёт синтетический.', severity='warning')
                return
            self.start(question=question)

    def start(self, question):
        if self.running:
            return
        if not os.getenv('DEEPSEEK_API_KEY', '').strip():
            self.notify('Задайте DEEPSEEK_API_KEY в day-25/.env', severity='error')
            self.stage = 'Запуск остановлен: НЕТ КЛЮЧА'
            self.tick()
            return
        try:
            settings = Settings(int(self.query_one('#before', Input).value), int(self.query_one('#after', Input).value),
                                float(self.query_one('#threshold', Input).value))
        except ValueError:
            self.notify('Проверьте K и порог от 0 до 1', severity='error')
            return
        from dataclasses import asdict
        self.session['settings'] = asdict(settings)
        self.running, self.started = True, time.monotonic()
        self.stop.clear()
        self.set_busy(True)
        if not self.path:
            self.path = DAY / 'sessions' / ('chat-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
        if self.runner is None:
            self.runner = self.runner_factory(key=os.getenv('DEEPSEEK_API_KEY', ''),
                                              model=os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'))
        self.execute(question)

    def set_busy(self, value):
        for name in ('send', 'new', 'open', 'sessions', 'before', 'after', 'threshold', 'input'):
            self.query_one('#' + name).disabled = value
        self.query_one('#stop').disabled = not value

    def receive(self, session, stage):
        self.session, self.stage = session, stage
        self.render_session(select_latest=not self.inspecting_turn())

    def inspecting_turn(self):
        return self.selected >= 0 and self.query_one('#tabs', TabbedContent).active in (
            'checks-tab', 'sources-tab', 'memory-tab')

    @work(exclusive=True)
    async def execute(self, question):
        def notify(session, stage):
            self.call_from_thread(self.receive, copy.deepcopy(session), stage)

        session, runner, path = self.session, self.runner, self.path

        def run():
            return runner.send(session, question, path, notify, self.stop.is_set)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='day25-rag')
        try:
            await asyncio.get_running_loop().run_in_executor(executor, run)
        except Exception as error:
            self.notify(f'Ошибка: {type(error).__name__}', severity='error')
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.finished()

    def finished(self):
        self.running = False
        self.stage = 'ready'
        self.set_busy(False)
        self.query_one('#input', TextArea).load_text('')
        self.refresh_sessions()
        self.render_session(select_latest=not self.inspecting_turn())

    def action_quit(self):
        if self.running:
            self.stop.set()
            self.notify('Сначала остановите выполнение. Текущий запрос может ждать до 90 секунд.')
        else:
            self.exit()
