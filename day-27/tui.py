"""Single-screen local chat with persistent sessions and live streaming."""
import argparse
import asyncio
import time
from dataclasses import asdict
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Input, Select, Static, Switch, TextArea

from chat import DAY, OllamaClient, Settings, load_session, new_session, prepare_messages, save_session


class ChatInput(TextArea):
    BINDINGS = [Binding('enter', 'submit', show=False, priority=True),
                Binding('shift+enter', 'newline', show=False, priority=True)]

    def action_submit(self):
        self.app.action_send()

    def action_newline(self):
        self.insert('\n')


def number(value, suffix=''):
    return '—' if value is None else f'{value:.2f}{suffix}'


class ChatApp(App):
    TITLE = 'День 27 · Локальный чат'
    CSS_PATH = 'tui.tcss'
    BINDINGS = [('ctrl+q', 'quit', 'Выход'), ('ctrl+enter', 'send', 'Отправить'),
                ('escape', 'stop', 'Остановить')]

    def __init__(self, *, session_path=None, directory=None, client=None):
        super().__init__()
        self.directory = Path(directory) if directory is not None else DAY / 'sessions'
        self.session = load_session(session_path) if session_path else new_session()
        self.client = client or OllamaClient()
        self.generation_task = None
        self.started = None
        self.dirty = False
        self.phase = 'Готов'
        self.theme = 'textual-dark'

    def compose(self) -> ComposeResult:
        yield Static('ДЕНЬ 27  /  ЛОКАЛЬНЫЙ ЧАТ · OLLAMA · 127.0.0.1:11434', id='brand')
        with Horizontal(id='toolbar'):
            yield Select([], prompt='Сохранённые сессии', id='sessions')
            yield Button('Открыть', id='open')
            yield Button('Новая сессия', id='new')
            yield Button('Остановить', id='stop', disabled=True)
        yield Static('Готов', id='status', markup=False)
        with Horizontal(id='workspace'):
            with VerticalScroll(id='chat-scroll'):
                yield Static('', id='chat', markup=False)
            with VerticalScroll(id='sidebar'):
                yield Static('НАСТРОЙКИ · следующий запрос', classes='heading')
                yield Static('Модель')
                yield Input(id='model')
                with Horizontal(id='think-row'):
                    yield Static('Thinking')
                    yield Switch(id='thinking')
                yield Static('Temperature · 0–2')
                yield Input(id='temperature')
                yield Static('Контекст · 2048–32768 токенов')
                yield Input(id='num_ctx')
                yield Static('Лимит генерации · включая thinking')
                yield Input(id='num_predict')
                yield Static('ПОСЛЕДНИЙ ЗАПРОС', classes='heading')
                yield Static(id='metrics', markup=False)
                yield Static('ПАМЯТЬ СЕССИИ', classes='heading')
                yield Static(id='memory', markup=False)
                yield Static('СУММА ЗА СЕССИЮ', classes='heading')
                yield Static(id='totals', markup=False)
                yield Static('Вход суммируется с повторной историей. '
                             'Генерация включает thinking.', id='note')
        with Horizontal(id='composer'):
            yield ChatInput(placeholder='Enter — отправить · Shift+Enter — новая строка', id='input')
            yield Button('Отправить', id='send', variant='primary')
        yield Footer()

    def on_mount(self):
        self.load_settings()
        self.refresh_sessions()
        self.render_chat()
        self.render_stats()
        self.set_interval(.1, self.tick)
        self.query_one('#input').focus()

    def load_settings(self):
        for name in ('model', 'temperature', 'num_ctx', 'num_predict'):
            self.query_one('#' + name, Input).value = str(self.session['settings'][name])
        self.query_one('#thinking', Switch).value = self.session['settings']['thinking']

    def settings(self):
        try:
            return Settings(model=self.query_one('#model', Input).value.strip(),
                thinking=self.query_one('#thinking', Switch).value,
                temperature=float(self.query_one('#temperature', Input).value),
                num_ctx=int(self.query_one('#num_ctx', Input).value),
                num_predict=int(self.query_one('#num_predict', Input).value)).validate()
        except (ValueError, TypeError) as error:
            raise ValueError(f'Проверьте настройки: {error}') from error

    def refresh_sessions(self):
        options = []
        for path in sorted(self.directory.glob('*.json'), reverse=True):
            try:
                data = load_session(path)
                options.append((f"{data['title'][:32]} · {path.stem}", str(path)))
            except (ValueError, KeyError, TypeError, OSError):
                continue
        self.query_one('#sessions', Select).set_options(options)

    def persist(self):
        try:
            save_session(self.session, self.directory)
            return True
        except OSError as error:
            self.query_one('#status', Static).update(f'Ошибка сохранения сессии: {error}')
            return False

    def render_chat(self):
        scroll = self.query_one('#chat-scroll', VerticalScroll)
        follow = scroll.scroll_y >= scroll.max_scroll_y - 1
        text = Text()
        if not self.session['turns']:
            text.append('Локальная модель готова к диалогу.\n\n', style='bold cyan')
            text.append('История сохраняется автоматически. Открывайте прежние сессии в верхней панели.\n')
        for index, turn in enumerate(self.session['turns'], 1):
            text.append(f'ВЫ · {index}\n', style='bold cyan')
            text.append(turn['user'] + '\n\n')
            if turn['thinking']:
                text.append('РАССУЖДЕНИЯ\n', style='bold yellow')
                text.append(turn['thinking'] + '\n\n', style='dim')
            text.append(f"{turn['settings']['model']} · {turn['status']}\n", style='bold green')
            text.append(turn['content'] + '\n')
            if turn.get('error'):
                text.append(turn['error'] + '\n', style='red')
            if turn['metrics'].get('done_reason') == 'length':
                text.append('Достигнут лимит генерации; ответ может быть неполным.\n', style='yellow')
            text.append('\n' + '─' * 32 + '\n\n', style='dim')
        self.query_one('#chat', Static).update(text)
        if follow:
            scroll.scroll_end(animate=False)

    def render_stats(self):
        turn = self.session['turns'][-1] if self.session['turns'] else None
        metrics = turn['metrics'] if turn else {}
        elapsed = time.perf_counter() - self.started if self.started else metrics.get('wall_seconds')
        self.query_one('#metrics', Static).update(
            f"Вход: {metrics.get('input_tokens', '—')} ток.\n"
            f"Генерация: {metrics.get('output_tokens', '—')} ток.\n"
            f"Скорость: {number(metrics.get('tokens_per_second'), ' ток./с')}\n"
            f"Первый фрагмент: {number(metrics.get('first_fragment_seconds'), ' с')}\n"
            f"Начало ответа: {number(metrics.get('first_content_seconds'), ' с')}\n"
            f"Полное время: {number(elapsed, ' с')}\n"
            f"Загрузка: {number(metrics.get('load_seconds'), ' с')}")
        memory = turn.get('context', {}) if turn else {}
        self.query_one('#memory', Static).update(
            f"Ходов сохранено: {len(self.session['turns'])}\n"
            f"Прошлых ходов в запросе: {memory.get('included', '—')}\n"
            f"Старых ходов исключено: {memory.get('excluded', '—')}\n"
            f"Оценка входа: {memory.get('estimated_bytes', '—')} байт\n"
            'Оценка по UTF-8, не токенизатор.\n'
            'Ошибки и отмены не входят в контекст.')
        completed = [t for t in self.session['turns'] if t['metrics']]
        self.query_one('#totals', Static).update(
            f"Запросов с метриками: {len(completed)}\n"
            f"Вход: {sum(t['metrics'].get('input_tokens') or 0 for t in completed)} ток.\n"
            f"Генерация: {sum(t['metrics'].get('output_tokens') or 0 for t in completed)} ток.\n"
            f"Время: {sum(t['metrics'].get('wall_seconds') or 0 for t in completed):.2f} с")

    def tick(self):
        if self.started:
            if self.dirty:
                self.render_chat()
                self.dirty = False
            self.render_stats()
            self.query_one('#status', Static).update(f'{self.phase} · {time.perf_counter() - self.started:.1f} с')

    def busy(self, value):
        for name in ('send', 'open', 'new', 'sessions', 'model', 'temperature', 'num_ctx', 'num_predict', 'thinking'):
            self.query_one('#' + name).disabled = value
        self.query_one('#stop', Button).disabled = not value

    def action_send(self):
        if self.generation_task and not self.generation_task.done():
            return
        question = self.query_one('#input', TextArea).text.strip()
        if not question:
            return
        try:
            settings = self.settings()
            messages, context = prepare_messages(self.session, question, settings)
        except ValueError as error:
            self.query_one('#status', Static).update(str(error))
            return
        self.session['settings'] = asdict(settings)
        if not self.session['turns']:
            self.session['title'] = question[:60]
        turn = {'user': question, 'content': '', 'thinking': '', 'status': 'running',
                'settings': asdict(settings), 'context': context, 'metrics': {}}
        self.session['turns'].append(turn)
        if not self.persist():
            self.session['turns'].pop()
            return
        self.query_one('#input', TextArea).load_text('')
        self.started, self.phase = time.perf_counter(), 'Ожидание модели'
        self.busy(True)
        self.render_chat()
        self.generation_task = asyncio.create_task(self.generate(turn, messages, settings))

    async def generate(self, turn, messages, settings):
        def update(content, thinking):
            turn['content'] += content
            turn['thinking'] += thinking
            self.phase = 'Отвечает' if content else 'Рассуждает'
            self.dirty = True
        status = 'Готов'
        try:
            turn['metrics'] = await self.client.generate(messages, settings, update)
            turn['status'] = 'complete' if turn['content'] else 'empty'
            if not turn['content']:
                turn['error'] = 'Модель не выдала текст ответа. Возможно, весь лимит потрачен на thinking.'
            status = 'Лимит генерации достигнут' if turn['metrics']['done_reason'] == 'length' else ('Ответ завершён' if turn['content'] else 'Нет текста ответа')
        except asyncio.CancelledError:
            turn['status'] = 'cancelled'
            turn['error'] = 'Остановлено. Неполный ответ сохранён, в контекст не передаётся.'
            status = 'Остановлено'
        except Exception as error:
            turn['status'], turn['error'] = 'error', str(error)
            status = str(error)
        finally:
            self.started = None
            self.dirty = False
            self.busy(False)
            self.render_chat()
            self.render_stats()
            self.query_one('#status', Static).update(status)
            self.persist()
            self.refresh_sessions()
            self.query_one('#input').focus()

    def action_stop(self):
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

    async def action_quit(self):
        self.action_stop()
        if self.generation_task:
            await self.generation_task
        try:
            self.session['settings'] = asdict(self.settings())
        except ValueError:
            pass
        if self.session['turns'] and not self.persist():
            return
        self.exit()

    @on(Button.Pressed)
    def button_pressed(self, event):
        action = event.button.id
        if action == 'send':
            self.action_send()
        elif action == 'stop':
            self.action_stop()
        elif action in ('new', 'open') and not (self.generation_task and not self.generation_task.done()):
            try:
                if self.session['turns']:
                    self.session['settings'] = asdict(self.settings())
                    if not self.persist():
                        return
                if action == 'new':
                    settings = asdict(self.settings())
                    session = new_session()
                    session['settings'] = settings
                else:
                    path = self.query_one('#sessions', Select).value
                    if path is Select.BLANK:
                        raise ValueError('Выберите сессию.')
                    session = load_session(path)
                self.session = session
                self.load_settings()
                self.render_chat()
                self.render_stats()
                self.refresh_sessions()
                self.query_one('#input', TextArea).load_text('')
                self.query_one('#status', Static).update('Сессия открыта' if action == 'open' else 'Новая сессия')
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.query_one('#status', Static).update(str(error))


def main():
    parser = argparse.ArgumentParser(description='Локальный TUI-чат с Ollama. Сессии не входят в Git.')
    parser.add_argument('--session', type=Path, help='Открыть сохранённый JSON сессии')
    args = parser.parse_args()
    try:
        ChatApp(session_path=args.session).run()
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Ошибка: {error}\n')


if __name__ == '__main__':
    main()
