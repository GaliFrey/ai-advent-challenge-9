"""Fullscreen comparison, source inspection and shared persistent memory."""
import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from dotenv import load_dotenv
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Footer, Input, Label, Select, Static, Switch, TabbedContent, TabPane, TextArea

from clients import CloudClient, DocsClient, LocalClient
from core import DAY, Settings, display_answer, load_session, new_session, save_session
from pipeline import Runner
from ollama import OllamaManager


class QuestionInput(TextArea):
    BINDINGS = [Binding('enter', 'submit', show=False, priority=True),
                Binding('shift+enter', 'newline', show=False, priority=True)]

    def action_submit(self):
        self.app.action_send()

    def action_newline(self):
        self.insert('\n')


STATUS_LABELS = {'pending': 'Ожидание', 'running': 'Формируется', 'complete': 'Завершено',
    'error': 'Ошибка', 'invalid': 'Проверка не пройдена', 'incomplete': 'Ответ не завершён',
    'cancelled': 'Отменено', 'interrupted': 'Прервано', 'disabled': 'Выключено',
    'skipped': 'Пропущено', 'no_sources': 'Источников нет', 'attention': 'Нужна проверка'}


def status_label(value):
    return STATUS_LABELS.get(value, 'Нет ответа')


def memory_text(memory):
    questions = memory.get('user_questions', [])
    answers = memory.get('prior_answers', [])
    history = []
    for i, question in enumerate(questions):
        text = f'{i + 1}. Пользователь: {question}'
        previous = answers[i] if i < len(answers) else None
        if previous:
            text += f"\nМодель {previous['model']} ({status_label(previous['status'])}): {previous['answer']}"
            if previous.get('gaps'):
                text += '\nПробелы: ' + str(previous['gaps'])
        history.append(text)
    return ('Заметки: ' + (memory.get('notes') or 'не заданы') + '\n\nПредыдущие вопросы:\n'
            + ('\n\n'.join(history) or 'Предыдущих вопросов нет.')
            + f"\n\nНе вошло в контекст: {memory.get('excluded_questions', 0)} вопросов.")


def seconds(value):
    return '—' if value is None else f'{value:.2f} с'


class RagApp(App):
    TITLE = 'День 28 · WebTutor RAG'
    CSS_PATH = 'tui.tcss'
    BINDINGS = [('ctrl+q', 'quit', 'Выход'), ('ctrl+enter', 'send', 'Отправить'), ('escape', 'stop', 'Стоп')]

    def __init__(self, *, directory=None, session_path=None, runner=None, ollama=None):
        super().__init__()
        self.directory = Path(directory or DAY / 'sessions')
        self.session = load_session(session_path) if session_path else new_session()
        if not session_path:
            self.session['settings']['cloud_model'] = os.getenv('DEEPSEEK_MODEL', 'deepseek-flash')
        self.runner = runner or Runner(DocsClient(), LocalClient(), CloudClient(os.getenv('DEEPSEEK_API_KEY', '')))
        self.ollama = ollama or OllamaManager()
        self.ollama_task = None
        self.ollama_operation = None
        self._ollama_closing = False
        self.ollama_state = {'available': False, 'reachable': False, 'owned': False, 'models': [], 'running': [], 'error': ''}
        self.generation_task = None
        self.index = len(self.session['turns']) - 1
        self.dirty = False
        self.started = None
        self.phase = 'Готов'
        self.last_save = 0
        self.theme = 'textual-dark'
        self._ui_ready = False
        self._compact = False
        self._turn_options = None
        self._source_signature = None
        self._rendered_index = None

    def compose(self) -> ComposeResult:
        yield Static('', id='brand', markup=False)
        with Horizontal(id='toolbar'):
            yield Select([], prompt='Сессии', id='sessions')
            yield Button('Открыть', id='open')
            yield Button('Новая', id='new')
            yield Button('Сохранить', id='save')
            yield Button('Стоп', id='stop', disabled=True)
        with Horizontal(id='turnbar'):
            yield Select([], prompt='Ход диалога', id='turns')
            yield Button('Повтор на том же контексте', id='repeat', disabled=True)
        yield Static('Готов', id='status', markup=False)
        with TabbedContent(id='tabs'):
            with TabPane('Ответы', id='answers-tab'):
                with Horizontal(id='answers'):
                    for name, label in [('local', 'ЛОКАЛЬНАЯ · OLLAMA'), ('cloud', 'ОБЛАЧНАЯ · DEEPSEEK')]:
                        with Vertical(classes='answer-column'):
                            yield Static(label, classes='heading')
                            yield Static(id=name+'-metrics', classes='metrics', markup=False)
                            with VerticalScroll(id=name+'-scroll', classes='answer-scroll'):
                                yield Static('', id=name+'-answer')
                                with Collapsible(title='Диагностика ответа', collapsed=True):
                                    yield Static('', id=name+'-diagnostics', markup=False)
            with TabPane('Источники и поиск', id='sources-tab'):
                with VerticalScroll(id='source-panel'):
                    yield Static(id='search-details', markup=False)
                    yield Select([], prompt='Прочитанный источник', id='sources')
                    yield Static(id='source-text', markup=False)
                    with Collapsible(title='Диагностика поиска', collapsed=True):
                        yield Static(id='search-diagnostics', markup=False)
            with TabPane('Память', id='memory-tab'):
                with VerticalScroll():
                    yield Static('Общие заметки · требования и подтверждённые уточнения.\n'
                        'Сохраняются кнопкой сверху, при отправке и выходе.')
                    yield TextArea(id='notes', soft_wrap=True)
                    yield Static(id='memory-details', markup=False)
                    with Collapsible(title='Диагностика памяти', collapsed=True):
                        yield Static(id='memory-diagnostics', markup=False)
            with TabPane('Ollama', id='ollama-tab'):
                with VerticalScroll(id='ollama-scroll'):
                    yield Static('Состояние ещё не проверено', id='ollama-status', markup=False)
                    with Horizontal(id='ollama-buttons'):
                        yield Button('Запустить', id='ollama-start', variant='primary')
                        yield Button('Остановить', id='ollama-stop', disabled=True)
                        yield Button('Обновить список', id='ollama-refresh')
                    yield Label('Установленная локальная модель · для следующих запросов')
                    yield Select([], prompt='Список появится после запуска сервера', id='ollama-model')
                    yield Static(id='ollama-selected', markup=False)
                    yield Static(id='ollama-model-details', markup=False)
                    yield Static('МОДЕЛИ В ПАМЯТИ', classes='heading')
                    yield Static(id='ollama-loaded', markup=False)
                    yield Static('Запуск только по кнопке. Модель загрузится при первом вопросе.\n'
                        'При выходе сервер, запущенный этим TUI, останавливается.\n'
                        'Сервер из другого терминала остаётся под твоим управлением.')
            with TabPane('Настройки', id='settings-tab'):
                with VerticalScroll(id='settings-scroll'):
                    yield Label('Локальная модель')
                    yield Input(id='local_model')
                    yield Label('Облачная модель')
                    yield Input(id='cloud_model')
                    for name, label in [('cloud', 'Сравнивать с облаком · один платный вызов на ход'),
                                        ('planner', 'Локальный планировщик · память и до 3 запросов'),
                                        ('thinking', 'Thinking локального ответа · расходует лимит генерации')]:
                        with Horizontal(classes='switch-row'):
                            yield Label(label)
                            yield Switch(id=name)
                    for name, label in [('temperature', 'Температура · 0–2'),
                                        ('num_ctx', 'Контекст Ollama · 4096–32768'),
                                        ('num_predict', 'Лимит генерации · 256–8192, не больше половины контекста'),
                                        ('top_k', 'Число читаемых источников · 1–8'),
                                        ('max_chars', 'Лимит текста каждого источника · 1000–12000 символов')]:
                        yield Label(label)
                        yield Input(id=name)
                    yield Label('Корпус MCP · docs объединяет Datex и Portal')
                    yield Select([('Datex + Portal', 'docs'), ('Datex', 'datex'), ('Portal', 'portal'),
                                  ('SQL/XML-схема', 'schema')], allow_blank=False, id='source')
                    yield Static('Облако выключено по умолчанию. Ключ берётся только из .env/окружения.\n'
                        'Поиск и планировщик локальные; источники общие, история ответов у каждой модели своя.\n'
                        'Проверка цитат не доказывает предметную правильность ответа.')
        with Horizontal(id='composer'):
            yield QuestionInput(placeholder='Вопрос о WebTutor · Enter — отправить · Shift+Enter — новая строка', id='question')
            yield Button('Отправить', variant='primary', id='send')
        yield Footer()

    def on_mount(self):
        self._ui_ready = True
        self.update_layout(self.size)
        self.load_fields()
        self.refresh_sessions()
        self.refresh_turns()
        self.load_turn_details()
        self.render_turn()
        self.set_interval(.2, self.tick)
        self.set_interval(3, self.poll_ollama)
        self.begin_ollama('refresh')
        self.query_one('#question').focus()

    def load_fields(self):
        for key, value in self.session['settings'].items():
            widget = self.query_one('#' + key)
            if isinstance(widget, Switch):
                widget.value = value
            elif isinstance(widget, Select):
                widget.value = value
            else:
                widget.value = str(value)
        self.query_one('#notes', TextArea).load_text(self.session['notes'])
        self.render_ollama()
        self.render_brand()

    def render_brand(self):
        local = self.query_one('#local_model', Input).value
        cloud = self.query_one('#cloud_model', Input).value
        comparison = f'{local} ↔ {cloud}' if self.query_one('#cloud', Switch).value else f'{local} · облако выключено'
        self.query_one('#brand', Static).update('ДЕНЬ 28 · WebTutor · ' + comparison)

    @on(Input.Changed, '#cloud_model')
    @on(Switch.Changed, '#cloud')
    def comparison_changed(self):
        self.render_brand()

    def update_layout(self, size):
        self._compact = size.width < 100 or size.height < 32
        self.screen.set_class(self._compact, 'compact')
        self.query_one('#repeat', Button).label = 'Повтор' if self._compact else 'Повтор на том же контексте'

    def on_resize(self, event):
        if self._ui_ready:
            self.update_layout(event.size)
            self.render_turn()

    def read_settings(self):
        values = {}
        for key in asdict(Settings()):
            widget = self.query_one('#' + key)
            value = widget.value
            if key in ('num_ctx', 'num_predict', 'top_k', 'max_chars'):
                value = int(value)
            elif key == 'temperature':
                value = float(value)
            values[key] = value
        return Settings(**values).validate()

    def capture(self, *, settings=True):
        self.session['notes'] = self.query_one('#notes', TextArea).text
        if settings:
            self.session['settings'] = asdict(self.read_settings())

    def persist(self):
        try:
            save_session(self.session, self.directory)
            self.last_save = time.monotonic()
            return True
        except OSError:
            self.phase = 'Ошибка сохранения сессии; проверьте доступ к каталогу.'
            self.query_one('#status', Static).update(self.phase)
            return False

    def refresh_sessions(self):
        options = []
        for path in sorted(self.directory.glob('*.json'), reverse=True):
            try:
                data = load_session(path)
                options.append((data['title'][:35] + ' · ' + path.stem, str(path)))
            except (ValueError, TypeError, KeyError, OSError):
                continue
        self.query_one('#sessions', Select).set_options(options)

    def refresh_turns(self):
        select = self.query_one('#turns', Select)
        options = [(f"{i+1}. {t['question'][:70]} · {status_label(t['status'])}", i)
                   for i, t in enumerate(self.session['turns'])]
        with select.prevent(Select.Changed):
            if options != self._turn_options:
                select.set_options(options)
                self._turn_options = options
            if self.index >= 0:
                select.value = self.index
            else:
                select.clear()

    def selected(self):
        return self.session['turns'][self.index] if 0 <= self.index < len(self.session['turns']) else None

    def render_turn(self):
        turn = self.selected()
        self.render_brand()
        for name in ('local', 'cloud'):
            result = turn['results'][name] if turn else {}
            metrics = result.get('metrics', {})
            speed = metrics.get('tokens_per_second')
            label = status_label(result.get('status'))
            if result.get('status') == 'running':
                label += ' · ещё не проверено'
            checks = result.get('checks')
            if checks:
                label += ' · ' + ('цитаты прошли' if checks['passed'] else 'цитаты/формат не прошли')
            information = (f"{result.get('model', '')} · {label}\n"
                f"Время: {seconds(metrics.get('wall_seconds'))} · первый текст: {seconds(metrics.get('first_content_seconds'))}\n"
                f"Токены вход/выход: {metrics.get('input_tokens') if metrics.get('input_tokens') is not None else '—'}/{metrics.get('output_tokens') if metrics.get('output_tokens') is not None else '—'}"
                + (f' · {speed:.1f} ток./с' if speed is not None else ''))
            if self._compact:
                information = f"{result.get('model', '')} · {status_label(result.get('status'))}\nВремя: {seconds(metrics.get('wall_seconds'))}"
            if result.get('error'):
                information += '\n' + result['error']
            self.query_one('#' + name + '-metrics', Static).update(information)
            text = display_answer(result) if turn else 'Задайте вопрос. Здесь появится ответ и его метрики.'
            scroll = self.query_one('#' + name + '-scroll', VerticalScroll)
            follow = scroll.scroll_y >= scroll.max_scroll_y - 1
            self.query_one('#' + name + '-answer', Static).update(RichMarkdown(text))
            self.query_one('#' + name + '-diagnostics', Static).update(json.dumps(
                {k: result.get(k) for k in ('raw', 'thinking', 'metrics', 'checks', 'error')}, ensure_ascii=False, indent=2))
            if self._rendered_index != self.index:
                scroll.scroll_home(animate=False)
            elif follow and result.get('status') == 'running':
                scroll.scroll_end(animate=False)
        self._rendered_index = self.index
        if turn:
            plan = turn.get('plan', {})
            queries = plan.get('queries', [])
            details = f"Вопрос: {turn['question']}\n"
            if plan.get('question') and plan['question'] != turn['question']:
                details += 'Вопрос для поиска: ' + plan['question'] + '\n'
            details += '\nПоисковые запросы:\n' + ('\n'.join(f'• {q}' for q in queries) or 'Поиск ещё не выполнен.')
            details += f"\n\nПрочитано источников: {len(turn['sources'])} · поиск: {seconds(turn.get('retrieval_seconds'))}"
            if turn.get('repeat_of'):
                details += f"\nПовтор хода {turn['repeat_of']} на сохранённом контексте."
            self.query_one('#search-details', Static).update(details)
            contexts = turn.get('provider_contexts')
            if contexts:
                text = '\n\n'.join(label + '\n\n' + memory_text(contexts[name]['memory'])
                                   for name, label in (('local', 'ЛОКАЛЬНАЯ МОДЕЛЬ'), ('cloud', 'ОБЛАЧНАЯ МОДЕЛЬ')))
            else:
                memory = turn.get('context', {}).get('memory', turn.get('memory', {}))
                text = memory_text(memory)
            self.query_one('#memory-details', Static).update('ПАМЯТЬ ВЫБРАННОГО ХОДА\n\n' + text)
            self.query_one('#memory-diagnostics', Static).update(json.dumps(
                {'snapshot': turn.get('memory'), 'sent': contexts or turn.get('context', {}).get('memory')}, ensure_ascii=False, indent=2))
            self.query_one('#search-diagnostics', Static).update(json.dumps(
                {k: turn.get(k) for k in ('context', 'provider_contexts', 'server', 'searches', 'candidates')}, ensure_ascii=False, indent=2))
        else:
            self.query_one('#search-details', Static).update('Источников ещё нет.')
            self.query_one('#memory-details', Static).update('История пуста. Каждая модель будет видеть свои предыдущие ответы.')
            self.query_one('#memory-diagnostics', Static).update('')
            self.query_one('#search-diagnostics', Static).update('')
        can_repeat = bool(turn and turn.get('messages') and turn.get('sources'))
        self.query_one('#repeat', Button).disabled = not can_repeat or self.running() or self.ollama_operation in ('start', 'stop')

    def load_turn_details(self):
        self.refresh_sources(force=True)

    def refresh_sources(self, force=False):
        turn = self.selected()
        sources = turn['sources'] if turn else []
        signature = (self.index, tuple((s['source_id'], s['ref'], s.get('chunk_id'), s['title'], s['incomplete']) for s in sources))
        if not force and signature == self._source_signature:
            return
        self._source_signature = signature
        select = self.query_one('#sources', Select)
        with select.prevent(Select.Changed):
            select.set_options([(f"{s['source_id']} · {s['corpus']} · {s['title']}" + (' · НЕПОЛНЫЙ' if s['incomplete'] else ''), i)
                                for i, s in enumerate(sources)])
            self.query_one('#source-text', Static).update('')
            if sources:
                select.value = 0
                self.show_source(0)
        self.query_one('#source-panel', VerticalScroll).scroll_home(animate=False)

    def show_source(self, index):
        turn = self.selected()
        if turn and 0 <= index < len(turn['sources']):
            source = turn['sources'][index]
            self.query_one('#source-text', Static).update(
                f"{source['source_id']} · {source['title']}\nРаздел: {source['heading']}\n"
                + ('Материал усечён; часть документа не вошла в контекст.' if source['incomplete'] else 'Материал прочитан без усечения.') + f"\n\n{source['text']}")
            if source.get('budget_truncated'):
                self.query_one('#source-text', Static).update(
                    f"{source['source_id']} · {source['title']}\nРаздел: {source['heading']}\n"
                    f"Сокращён по бюджету: прочитано {source['original_text_bytes']} байт, "
                    f"отправлено {len(source['text'].encode())} байт. Полный текст сохранён в сессии.\n\n{source['text']}")

    @on(Select.Changed, '#turns')
    def choose_turn(self, event):
        if not isinstance(event.value, int) or event.value == self.index:
            return
        if self.running():
            self.capture(settings=False)
        else:
            self.capture(settings=False)
            if not self.persist():
                return
        self.index = int(event.value)
        self.load_turn_details()
        self.render_turn()

    @on(Select.Changed, '#sources')
    def choose_source(self, event):
        if isinstance(event.value, int):
            self.show_source(int(event.value))

    def running(self):
        return self.generation_task is not None and not self.generation_task.done()

    def busy(self, value):
        for name in ('send', 'open', 'new', 'sessions', 'repeat', 'notes'):
            self.query_one('#' + name).disabled = value
        for widget in self.query('#settings-scroll Input, #settings-scroll Switch, #settings-scroll Select'):
            widget.disabled = value
        self.query_one('#stop', Button).disabled = not (value and self.started is not None)
        self.sync_ollama_controls()

    def changed(self):
        self.refresh_turns()
        self.refresh_sources()
        self.dirty = True

    def set_phase(self, value):
        self.phase = value

    def tick(self):
        if self.dirty:
            self.render_turn()
            self.dirty = False
            if time.monotonic() - self.last_save >= 1:
                self.persist()
        if self.started:
            self.query_one('#status', Static).update(f'{self.phase} · {time.perf_counter()-self.started:.1f} с')

    def sync_ollama_controls(self):
        managing = self.ollama_task is not None and not self.ollama_task.done()
        blocked = self.running() or managing
        state = self.ollama_state
        self.query_one('#ollama-start', Button).disabled = blocked or state['reachable'] or self.ollama.owned
        self.query_one('#ollama-stop', Button).disabled = blocked or not self.ollama.owned
        self.query_one('#ollama-refresh', Button).disabled = managing
        self.query_one('#ollama-model', Select).disabled = blocked or not state['models']

    def render_ollama(self):
        state = self.ollama_state
        if self.ollama_operation == 'start':
            status = 'Запускается…'
        elif self.ollama_operation == 'stop':
            status = 'Останавливается…'
        elif state['available']:
            status = 'Доступна · ' + ('сервер запущен этим TUI' if self.ollama.owned else 'внешний сервер')
        elif state['reachable']:
            status = 'HTTP доступен, API моделей не готово'
        else:
            status = 'Остановлена' if not self.ollama.owned else 'Процесс запущен, ожидание HTTP'
        if state['error']:
            status += '\n' + state['error']
        self.query_one('#ollama-status', Static).update(status + '\n127.0.0.1:11434')
        selected = self.query_one('#local_model', Input).value
        self.query_one('#ollama-selected', Static).update('Выбрана для RAG: ' + selected)
        models = state['models']
        options = [(m['name'], m['name']) for m in models]
        picker = self.query_one('#ollama-model', Select)
        if getattr(self, '_ollama_options', None) != options:
            picker.set_options(options)
            self._ollama_options = options
        if selected in [m['name'] for m in models] and picker.value != selected:
            picker.value = selected
        elif selected not in [m['name'] for m in models] and isinstance(picker.value, str):
            picker.clear()
        model = next((m for m in models if m['name'] == selected), None)
        if model:
            details = model.get('details', {})
            size = model.get('size')
            size_text = f"{size / 1024**3:.2f} ГиБ" if isinstance(size, (int, float)) else '—'
            detail_text = f"На диске: {size_text} · параметры: {details.get('parameter_size', '—')} · квантование: {details.get('quantization_level', '—')}"
        else:
            detail_text = 'Список моделей доступен после запуска сервера.' if not state['available'] else 'Выбранной модели нет в списке установленных локальных моделей.'
        self.query_one('#ollama-model-details', Static).update(detail_text)
        loaded = []
        for model in state['running']:
            if not isinstance(model, dict):
                continue
            vram = model.get('size_vram')
            vram_text = f'{vram / 1024**3:.2f} ГиБ' if isinstance(vram, (int, float)) else '—'
            loaded.append(f"{model.get('name', '—')} · GPU: {vram_text} · контекст: {model.get('context_length', '—')}")
        self.query_one('#ollama-loaded', Static).update('\n'.join(loaded) or 'Модели ещё не загружены.')
        self.sync_ollama_controls()

    @on(Select.Changed, '#ollama-model')
    def choose_ollama_model(self, event):
        if not isinstance(event.value, str) or self.running() or self.ollama_operation in ('start', 'stop'):
            return
        if event.value not in [m['name'] for m in self.ollama_state['models']]:
            return
        self.query_one('#local_model', Input).value = event.value
        self.session['settings']['local_model'] = event.value
        self.capture(settings=False)
        self.persist()
        self.render_ollama()

    @on(Input.Changed, '#local_model')
    def local_model_changed(self):
        self.render_ollama()
        self.render_brand()

    def poll_ollama(self):
        if self.query_one('#tabs', TabbedContent).active == 'ollama-tab':
            self.begin_ollama('refresh')

    def begin_ollama(self, operation):
        if self.ollama_task is not None and not self.ollama_task.done():
            return
        if operation != 'refresh' and self.running():
            return
        self.ollama_operation = operation
        self.ollama_task = asyncio.create_task(self.manage_ollama(operation))
        if operation != 'refresh':
            self.busy(True)
        self.render_ollama()

    async def manage_ollama(self, operation):
        try:
            if operation == 'start':
                self.ollama_state = await self.ollama.start(self.read_settings())
            elif operation == 'stop':
                await self.ollama.stop()
                self.ollama_state = await self.ollama.snapshot()
            else:
                self.ollama_state = await self.ollama.snapshot()
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError) as error:
            self.ollama_state['error'] = str(error)
        except Exception:
            self.ollama_state['error'] = 'Не удалось выполнить действие Ollama.'
        finally:
            self.ollama_operation = None
            # Inside the task it is not yet done; release controls before returning.
            self.ollama_task = None
            if not self._ollama_closing:
                self.busy(self.running())
                self.render_ollama()

    async def cleanup_ollama(self):
        self._ollama_closing = True
        task = self.ollama_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.ollama.stop()

    async def on_unmount(self):
        await self.cleanup_ollama()

    def action_send(self, repeat=None):
        if self.running() or self.ollama_operation in ('start', 'stop'):
            return
        question = self.query_one('#question', TextArea).text.strip()
        if repeat is not None:
            previous = self.session['turns'][repeat]
            question = previous['question']
        if not question:
            return
        try:
            self.capture()
            settings = self.read_settings()
            if settings.cloud and isinstance(self.runner.cloud, CloudClient) and not self.runner.cloud.key.strip():
                raise ValueError('Облако включено, но DEEPSEEK_API_KEY не задан. Заполните .env или выключите облако.')
            if not self.persist():
                return
        except (ValueError, TypeError) as error:
            self.query_one('#status', Static).update(str(error))
            return
        self.query_one('#question', TextArea).load_text('')
        self.started = time.perf_counter()
        self.busy(True)
        self.index = len(self.session['turns'])
        self.generation_task = asyncio.create_task(self.generate(question, settings, repeat))

    async def generate(self, question, settings, repeat):
        try:
            await self.runner.run(self.session, question, settings, self.changed, self.set_phase, repeat)
        except asyncio.CancelledError:
            pass
        finally:
            self.capture(settings=False)
            self.persist()
            self.started = None
            self.busy(False)
            self.refresh_turns()
            self.refresh_sessions()
            self.load_turn_details()
            self.render_turn()
            self.begin_ollama('refresh')
            turn = self.session['turns'][-1]
            self.query_one('#status', Static).update(f"Ход {len(self.session['turns'])}: {status_label(turn['status'])} · "
                f"{seconds(turn.get('total_seconds'))}" + (' · ' + turn['error'] if turn.get('error') else ''))

    def action_stop(self):
        if self.running():
            self.phase = 'Остановка'
            self.generation_task.cancel()

    async def action_quit(self):
        if self.running():
            self.generation_task.cancel()
            await self.generation_task
        self.capture(settings=False)
        if self.persist():
            await self.cleanup_ollama()
            self.exit()

    @on(Button.Pressed)
    def button(self, event):
        name = event.button.id
        if name in ('ollama-start', 'ollama-stop', 'ollama-refresh'):
            self.begin_ollama(name.removeprefix('ollama-'))
        elif name == 'send':
            self.action_send()
        elif name == 'stop':
            self.action_stop()
        elif name == 'repeat' and self.selected() and self.selected().get('messages'):
            self.action_send(repeat=self.index)
        elif name in ('save', 'new', 'open'):
            try:
                self.capture()
                if not self.persist():
                    return
                if name == 'new':
                    settings = self.session['settings']
                    self.session = new_session()
                    self.session['settings'] = settings
                elif name == 'open':
                    path = self.query_one('#sessions', Select).value
                    if not isinstance(path, str):
                        raise ValueError('Выберите сессию.')
                    # The picker only contains files in this day’s sessions directory.
                    if Path(path).resolve().parent != self.directory.resolve():
                        raise ValueError('Сессия вне каталога дня 28.')
                    self.session = load_session(path)
                else:
                    self.refresh_sessions()
                    self.query_one('#status', Static).update('Сессия сохранена.')
                    return
                self.index = len(self.session['turns']) - 1
                self.load_fields()
                self.refresh_turns()
                self.load_turn_details()
                self.render_turn()
                self.query_one('#status', Static).update('Готов')
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.query_one('#status', Static).update(f'Не удалось выполнить действие: {error}')


def main():
    parser = argparse.ArgumentParser(description='WebTutor MCP RAG: общие источники, отдельная история Ollama и DeepSeek')
    parser.add_argument('--session', type=Path, help='Открыть JSON-сессию')
    args = parser.parse_args()
    load_dotenv(DAY / '.env', override=False)
    RagApp(session_path=args.session).run()


if __name__ == '__main__':
    main()
