"""Sequential before/after comparison and the shared batch experiment in Textual."""
import argparse
import asyncio
import copy
import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, Markdown,
                             Select, Static, Switch, TabbedContent, TabPane, TextArea)

from cases import load_cases
from core import DAY, Settings, display_answer, load_report, profiles, status_label
from engine import Engine, safe_error
from judge import Judge
from main import configured_judge, report_text
from ollama import OllamaManager


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def profile_details(profile):
    if not profile:
        return 'Параметры появятся при запуске профиля.'
    settings = profile.get('settings', {})
    prompt = profile.get('prompt')
    prompt_label = {'original': 'Исходный', 'compact': 'Короткий'}.get(prompt, prompt or 'н/д')
    thinking = settings.get('thinking')
    thinking_label = 'включены' if thinking is True else 'выключены' if thinking is False else 'н/д'
    return (
        f'Модель: {settings.get("local_model", "н/д")}\n'
        f'Prompt: {prompt_label} | Рассуждения: {thinking_label}\n'
        f'Температура: {settings.get("temperature", "н/д")}\n'
        f'Контекст: {settings.get("num_ctx", "н/д")} токенов | '
        f'Лимит ответа: {settings.get("num_predict", "н/д")} токенов'
    )


class OptimizeApp(App):
    TITLE = 'День 29 — оптимизация WebTutor RAG'
    CSS_PATH = 'tui.tcss'
    BINDINGS = [('ctrl+enter', 'compare', 'Сравнить'), ('escape', 'cancel', 'Отмена'), ('ctrl+q', 'quit', 'Выход')]

    def __init__(self, engine=None, report=None, manager=None):
        super().__init__()
        self.options = profiles()
        self.engine = engine or Engine(judge=configured_judge())
        self.engine.notify = self.phase
        self.engine.changed = self.mark_dirty
        self.manager = manager or OllamaManager()
        self.report = report
        self.generation_task = None
        self.selected = None
        self.dirty = True
        self.rendered_rows = 0
        self.case_values = load_cases()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane('Сравнение', id='compare-tab'):
                yield Static('Новый запуск: слева настройки первого профиля, справа — второго.', id='setup-hint')
                with Horizontal(id='profile-editors'):
                    for side, title, preset in [('before', 'Первый профиль', 'baseline'),
                                                ('after', 'Второй профиль', 'compact-prompt')]:
                        editor = Vertical(id=side + '-editor', classes='profile-editor')
                        editor.border_title = title
                        with editor:
                            selector = Select([(key, key) for key in self.options], value=preset,
                                              allow_blank=False, id=side)
                            selector.border_title = 'Пресет'
                            yield selector
                            with Horizontal(classes='parameter-row'):
                                for name, value, label, kind in [('temperature', '0', 'Температура', 'number'),
                                                                 ('context', '16384', 'Контекст', 'integer')]:
                                    field = Input(value, id=side + '-' + name, type=kind)
                                    field.border_title = label
                                    yield field
                            with Horizontal(classes='thinking-row'):
                                limit = Input('3072', id=side + '-limit', type='integer')
                                limit.border_title = 'Лимит ответа'
                                yield limit
                                yield Label('Рассуждения')
                                yield Switch(side == 'before', id=side + '-thinking')
                with Horizontal(id='judge-controls'):
                    yield Label('DeepSeek: 2 вызова')
                    yield Switch(value=True, id='judge-enabled')
                with Horizontal(id='case-controls'):
                    yield Select([('Ручной вопрос', 'manual')] + [(c['id'], c['id']) for c in self.case_values],
                                 value='manual', allow_blank=False, id='case')
                    yield Button('Сравнить', id='compare', variant='primary')
                    yield Button('Пакетный эксперимент', id='experiment')
                    yield Button('Отмена', id='cancel')
                yield TextArea(id='question')
                yield Static('Ожидание. Генерации идут последовательно; источники общие.', id='phase')
                yield Static('Результат выбранного запуска появится здесь.', id='result-caption')
                with Horizontal(id='answers'):
                    with VerticalScroll(classes='answer-pane'):
                        yield Static('Первый профиль', id='before-title', classes='answer-title')
                        yield Static(profile_details({}), id='before-profile', classes='profile-details', markup=False)
                        yield Static('', id='before-metrics')
                        yield Markdown('Ожидание ответа…', id='before-answer')
                    with VerticalScroll(classes='answer-pane'):
                        yield Static('Второй профиль', id='after-title', classes='answer-title')
                        yield Static(profile_details({}), id='after-profile', classes='profile-details', markup=False)
                        yield Static('', id='after-metrics')
                        yield Markdown('Ожидание ответа…', id='after-answer')
            with TabPane('Источники и память'):
                yield Select([('Datex + Portal', 'docs'), ('Datex', 'datex'), ('Portal', 'portal'), ('SQL/XML', 'schema')],
                             value='docs', allow_blank=False, id='source')
                yield Label('Заметки для нового ручного вопроса:')
                yield TextArea(id='notes')
                yield TextArea(read_only=True, id='sources')
            with TabPane('Отчёт'):
                yield DataTable(id='comparisons', cursor_type='row')
                with Horizontal():
                    yield Input(placeholder='Путь к сохранённому JSON-отчёту', id='report-path')
                    yield Button('Открыть', id='open-report')
                yield TextArea(read_only=True, id='summary')
            with TabPane('Диагностика'):
                yield TextArea(read_only=True, id='diagnostics')
            with TabPane('Ollama'):
                with Horizontal():
                    yield Button('Обновить', id='refresh-ollama')
                    yield Button('Запустить', id='start-ollama')
                    yield Button('Остановить свой сервер', id='stop-ollama')
                yield TextArea(read_only=True, id='ollama-state')
        yield Footer()

    async def on_mount(self):
        self.query_one('#comparisons', DataTable).add_columns('№', 'Случай', 'Повтор', 'Первый профиль', 'Второй профиль', 'Оценка')
        self.set_interval(.15, self.refresh_result)
        self.apply_profile('before')
        self.apply_profile('after')
        if self.report:
            self.engine.report = self.report
            self.select_winner()
        self.refresh_result()
        if self.report is None:
            await self.refresh_ollama()
        else:
            self.query_one('#ollama-state', TextArea).load_text('Отчёт открыт без API. Для состояния сервера нажмите «Обновить».')

    def on_resize(self, event):
        self.screen.set_class(event.size.width < 110, 'compact')

    def select_winner(self):
        chosen = self.report.get('winner') or self.report.get('selected_profile')
        if chosen:
            self.options['selected'] = copy.deepcopy(chosen)
            self.query_one('#after', Select).set_options([(key, key) for key in self.options])
            self.query_one('#after', Select).value = 'selected'

    def phase(self, text):
        if self.is_mounted:
            self.query_one('#phase', Static).update(text)

    def mark_dirty(self):
        self.dirty = True

    def apply_profile(self, side):
        value = self.options[self.query_one('#' + side, Select).value]['settings']
        for widget, key in [('temperature', 'temperature'), ('context', 'num_ctx'), ('limit', 'num_predict')]:
            self.query_one('#' + side + '-' + widget, Input).value = str(value[key])
        self.query_one('#' + side + '-thinking', Switch).value = value['thinking']

    def on_select_changed(self, event: Select.Changed):
        if event.select.id in ('before', 'after') and event.value in self.options:
            self.apply_profile(event.select.id)
        elif event.select.id == 'case' and event.value != 'manual' and event.value is not Select.BLANK:
            case = next(c for c in self.case_values if c['id'] == event.value)
            self.query_one('#question', TextArea).load_text(case['question'])

    def chosen_profiles(self):
        selected = []
        for side, title in [('before', 'Первый профиль'), ('after', 'Второй профиль')]:
            profile = copy.deepcopy(self.options[self.query_one('#' + side, Select).value])
            try:
                profile['settings'].update(temperature=float(self.query_one('#' + side + '-temperature', Input).value),
                                           num_ctx=int(self.query_one('#' + side + '-context', Input).value),
                                           num_predict=int(self.query_one('#' + side + '-limit', Input).value),
                                           thinking=self.query_one('#' + side + '-thinking', Switch).value)
                Settings(**profile['settings']).validate()
            except ValueError as error:
                raise ValueError(title + ': ' + str(error)) from error
            selected.append(profile)
        return tuple(selected)

    def set_busy(self, value):
        profile_fields = tuple(side + '-' + name for side in ('before', 'after')
                               for name in ('temperature', 'context', 'limit', 'thinking'))
        for identifier in ('compare', 'experiment', 'before', 'after', 'case', 'judge-enabled',
                           'start-ollama', 'stop-ollama', 'open-report', 'source') + profile_fields:
            self.query_one('#' + identifier).disabled = value
        self.query_one('#cancel', Button).disabled = not value

    def action_compare(self):
        self.start_job(False)

    def start_job(self, batch):
        if self.generation_task and not self.generation_task.done():
            return
        try:
            before, after = self.chosen_profiles()
            identifier = self.query_one('#case', Select).value
            question = self.query_one('#question', TextArea).text
            notes = self.query_one('#notes', TextArea).text
            source = self.query_one('#source', Select).value
            if not batch and identifier == 'manual' and not question.strip():
                raise ValueError('Введите вопрос.')
            if isinstance(self.engine.judge, Judge):
                self.engine.judge = configured_judge(not self.query_one('#judge-enabled', Switch).value)
            if batch and not self.engine.judge.available:
                raise ValueError('Пакетный выбор требует включённого судьи и DEEPSEEK_API_KEY.')
            self.selected = None
            self.rendered_rows = 0
            self.query_one('#comparisons', DataTable).clear()
            self.set_busy(True)
            async def work():
                try:
                    if batch:
                        report = await self.engine.experiment()
                    else:
                        case = next(c for c in self.case_values if c['id'] == identifier) if identifier != 'manual' else (
                            await self.engine.retrieve_manual(question, notes, source))
                        report = await self.engine.manual(case, before, after)
                    self.report = report
                    self.phase('Завершено: ' + status_label(report['status']) + '. Отчёт сохранён.')
                    self.select_winner()
                except asyncio.CancelledError:
                    self.phase('Отменено. Частичный отчёт сохранён, если генерация уже началась.')
                except Exception as error:
                    self.phase(safe_error(error))
                finally:
                    self.report = self.engine.report
                    self.mark_dirty()
                    self.refresh_result()
                    self.set_busy(False)
            self.generation_task = asyncio.create_task(work())
        except (ValueError, TypeError) as error:
            self.phase(safe_error(error))

    def action_cancel(self):
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()

    async def refresh_ollama(self):
        snapshot = await self.manager.snapshot()
        self.query_one('#ollama-state', TextArea).load_text(pretty(snapshot))
        self.query_one('#stop-ollama', Button).disabled = not self.manager.owned

    async def on_button_pressed(self, event: Button.Pressed):
        identifier = event.button.id
        if identifier == 'compare':
            self.action_compare()
        elif identifier == 'experiment':
            self.start_job(True)
        elif identifier == 'cancel':
            self.action_cancel()
        elif identifier == 'open-report':
            try:
                self.report = load_report(self.query_one('#report-path', Input).value)
                self.engine.report = self.report
                self.selected, self.rendered_rows = None, 0
                self.query_one('#comparisons', DataTable).clear()
                self.select_winner()
                self.mark_dirty()
                self.refresh_result()
                self.phase('Отчёт открыт без API-вызовов.')
            except Exception as error:
                self.phase(safe_error(error))
        elif identifier in ('refresh-ollama', 'start-ollama', 'stop-ollama'):
            try:
                if identifier == 'start-ollama':
                    await self.manager.start(Settings())
                elif identifier == 'stop-ollama':
                    await self.manager.stop()
                await self.refresh_ollama()
            except Exception as error:
                self.phase(safe_error(error))

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        if event.data_table.id == 'comparisons':
            self.selected = int(event.row_key.value)
            self.mark_dirty()
            self.refresh_result()

    def refresh_result(self):
        if not self.dirty or not self.is_mounted:
            return
        self.dirty = False
        report = self.engine.report or self.report
        if not report:
            return
        comparisons = report['comparisons']
        table = self.query_one('#comparisons', DataTable)
        for index, comparison in enumerate(comparisons):
            values = (str(index), comparison['case_id'], str(comparison['repeat']+1),
                      status_label(comparison.get('before', {}).get('status', 'pending')),
                      status_label(comparison.get('after', {}).get('status', 'pending')),
                      status_label(comparison.get('decision', {}).get('status', 'ожидание')))
            if index >= self.rendered_rows:
                table.add_row(*values, key=str(index))
                self.rendered_rows += 1
            else:
                for column, value in zip(table.columns, values):
                    table.update_cell(str(index), column, value)
        self.query_one('#summary', TextArea).load_text(report_text(report))
        if not comparisons:
            return
        index = self.selected if self.selected is not None and self.selected < len(comparisons) else len(comparisons)-1
        comparison = comparisons[index]
        case = next(c for c in report['cases'] if c['id'] == comparison['case_id'])
        self.query_one('#result-caption', Static).update(
            f'Результат №{index}: {case["id"]}, повтор {comparison["repeat"]+1}. Профили — из этого запуска.')
        for side, title in [('before', 'Первый профиль'), ('after', 'Второй профиль')]:
            result = comparison.get(side, {})
            profile = result.get('profile', {})
            self.query_one('#' + side + '-title', Static).update(title + ': ' + profile.get('id', 'ожидание'))
            self.query_one('#' + side + '-profile', Static).update(profile_details(profile))
            metrics = result.get('metrics', {})
            labels = [('wall_seconds', 'время, с'), ('tokens_per_second', 'токенов/с'),
                      ('first_content_seconds', 'первый текст, с')]
            text = ' | '.join(f'{label}: {metrics[key]:.2f}' for key, label in labels if isinstance(metrics.get(key), (int, float)))
            for key, label in [('max_ram_bytes', 'RAM'), ('max_gpu_bytes', 'GPU')]:
                value = metrics.get(key)
                text += f' | {label}: {value/1024**2:.0f} MiB' if value is not None else f' | {label}: н/д'
            self.query_one('#' + side + '-metrics', Static).update(status_label(result.get('status', 'pending')) + '\n' + text)
            self.query_one('#' + side + '-answer', Markdown).update(display_answer(result))
        self.query_one('#sources', TextArea).load_text(pretty({'question': case['question'], 'sources': case['sources'],
                                                              'memory': case['memory'], 'criteria': case.get('criteria')}))
        self.query_one('#diagnostics', TextArea).load_text(pretty(comparison))
        self.query_one('#report-path', Input).value = str((self.engine.directory or DAY / 'reports') / (report['id'] + '.json'))

    async def on_unmount(self):
        self.action_cancel()
        if self.generation_task:
            await self.generation_task
        await self.manager.stop()


def main():
    parser = argparse.ArgumentParser(description='День 29: сравнение и оптимизация профилей Ollama')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    OptimizeApp(report=load_report(args.report) if args.report else None).run()


if __name__ == '__main__':
    main()
