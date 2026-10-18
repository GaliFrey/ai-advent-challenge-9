import asyncio
import json
import logging
from .storage import utcnow
from .tools import local_time
from .requests import Requests
from .profile import load_profile, profile_text, timezone_name, zone_label
from .receipts import confirmed_text
from .presentation import public_text, notification_text
from .telegram import TelegramError
from .voice import VoiceError

log = logging.getLogger(__name__)
HELP = ('Я сохраняю памятки, события и напоминания.\n\n'
        'Запомни: сегодня Оля обещала не пить газировку.\n'
        'Когда Оля обещала не пить газировку?\n'
        'Напомни через 5 минут проверить духовку.\n'
        'Каждый день в 09:00 напоминай выпить витамины.\n'
        'По будням в 18:00 напоминай закончить работу.\n'
        'Каждого 1, 10 и 20 числа в 10:00 напоминай проверить фильтр.\n'
        'Запомни день рождения Оли 20 октября.\n'
        'Покажи календарь на октябрь.\n'
        'Включи месячную сводку дней рождения.\n'
        'Какие у меня напоминания?\n'
        'Перенеси напоминание про духовку на завтра в 18:00.\n'
        'Отмени напоминание про духовку.\n'
        'Удали памятку про обещание Оли.\n\n'
        '/stats — расход токенов и время API.\n'
        '/profile — постоянные настройки.\n'
        '/new — сразу начать новую беседу.\n'
        '/request — незавершённое поручение; /cancel — остановить обработку и отменить черновик.\n'
        'Можно также просто общаться. Время — по постоянному профилю. '
        'Общее время уведомлений — 09:00; время встречи уточняется отдельно. После простоя досылка зависит от срока и начала события.')


class Runtime:
    def __init__(self, store, telegram, agent, owner, clock=utcnow, *, processing_timeout=120, voice=None):
        self.store, self.telegram, self.agent = store, telegram, agent
        self.owner, self.clock = owner, clock
        self.processing_timeout = processing_timeout
        self.voice = voice

    def authorized(self, update):
        msg = update.get('message') or update.get('callback_query', {}).get('message', {})
        sender = update.get('message', {}).get('from') or update.get('callback_query', {}).get('from', {})
        return (sender.get('id') == self.owner and msg.get('chat', {}).get('type') == 'private'
                and msg.get('chat', {}).get('id') == self.owner)

    async def poll_once(self):
        updates = await self.telegram.updates(self.store.get_meta('offset', 0))
        for update in updates:
            # Store every update before acknowledging its offset, including ignored updates.
            self.store.enqueue(update, self.clock())
            callback = update.get('callback_query')
            if callback:
                try:
                    await self.telegram.acknowledge(callback['id'])
                except Exception:
                    log.warning('Не удалось подтвердить кнопку; результат останется в очереди')

    async def process_once(self):
        job = self.store.next_job()
        if not job:
            return False
        await self._process_job(job)
        return True

    async def _process_job(self, job, *, interrupted=False):
        update = json.loads(job['payload'])
        if not self.authorized(update):
            with self.store.db:
                self.store.db.execute("UPDATE jobs SET status='ignored' WHERE id=?", (job['id'],))
            return True
        callback = update.get('callback_query')
        if callback:
            data = callback.get('data', '')
            with self.store.atomic():
                # The transaction contains both snooze and receipt: crash-safe callback replay.
                if data.startswith('snooze:') and data[7:].isdigit():
                    result = self.store.snooze(int(data[7:]), self.clock())
                else:
                    result = 'Неизвестная кнопка.'
                self.store.queue_message(f"callback:{job['id']}", result, now=self.clock())
                self.store.db.execute("UPDATE jobs SET status='done' WHERE id=?", (job['id'],))
            return True
        if not self.store.bind_job(job['id']):
            return True
        text = update['message'].get('text')
        is_voice = 'voice' in update['message']
        # A transcription is ordinary user speech, not a Telegram slash command.
        command = '' if is_voice else (text or '').strip()
        requests = Requests(self.store)
        if command == '/profile':
            self.store.complete_command(job['id'], profile_text(load_profile(self.store)), self.clock())
            return True
        if command == '/request':
            self.store.complete_command(job['id'], requests.describe(), self.clock())
            return True
        if command == '/cancel' or (command.casefold() in ('отмена', 'отмени поручение') and (interrupted or requests.active() or self.store.get_meta('pending'))):
            with self.store.atomic():
                result = requests.cancel(self.clock())
                reply = result['error']['message'] if 'error' in result else 'Незавершённое поручение отменено.'
                if interrupted:
                    reply = 'Текущая обработка остановлена. ' + ('Незавершённых поручений нет.' if 'error' in result else reply)
                self.store.complete_command(job['id'], reply, self.clock())
            return True
        if command.split()[:1] == ['/new']:
            self.store.new_command(job['id'], command, self.owner, self.clock())
            return True
        if not text and not is_voice:
            reply = ('Принимаю текстовые и голосовые сообщения.' if self.voice is not None
                     else 'Пока принимаю только текстовые сообщения.')
            self.store.finish(job['id'], '[нетекстовое сообщение]', reply, self.clock())
        elif command == '/stats':
            self.store.complete_command(job['id'], self.store.stats_text(), self.clock())
        elif command and command.split()[0] in ('/start', '/help'):
            self.store.complete_command(job['id'], HELP, self.clock())
        else:
            try:
                await self._run_agent(job, text or '[голосовое сообщение]')
            except asyncio.CancelledError:
                raise
            except VoiceError as exc:
                self.store.complete_command(job['id'], str(exc), self.clock())
            except Exception as exc:
                # Never print exceptions containing request URLs, headers or personal messages.
                log.warning('Обработка не завершена: %s', type(exc).__name__)
                fallback = ('Не удалось завершить обработку. Некоторые действия могли уже сохраниться. '
                            'Проверь список памяток или напоминаний перед повторной просьбой.')
                receipt = confirmed_text(self.store, job['id'], 'Обработка остальной части запроса не завершена.')
                actual_text = json.loads(job['payload'])['message'].get('text') or text or '[голосовое сообщение]'
                self.store.finish(job['id'], actual_text, receipt or fallback, self.clock(), presented=bool(receipt))
        return True

    def _next_control(self, active_id):
        # Only local commands may overtake a running generation. Ordinary
        # messages keep their order and never run two agents on one session.
        queued_message = False
        for row in self.store.db.execute("SELECT * FROM jobs WHERE status='pending' AND id!=? AND checkpoint IS NULL ORDER BY id", (active_id,)):
            job = dict(row)
            update = json.loads(job['payload'])
            if not self.authorized(update) or 'message' not in update:
                continue
            if 'voice' in update['message']:
                queued_message = True
                continue
            command = (update['message'].get('text') or '').strip()
            # Start the new session after preceding ordinary messages.
            if command.split()[:1] == ['/new'] and queued_message:
                continue
            if (command in ('/stats', '/profile', '/request', '/cancel')
                    or command.casefold() in ('отмена', 'отмени поручение')
                    or command.split()[:1] in (['/new'], ['/help'], ['/start'])):
                return job, command
            queued_message = True
        return None

    async def _stop_agent(self, task, job, text, reason, *, control_job=None):
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Cancellation happens outside the transaction. Persist the completed
        # job and its local control command together, without a restart gap.
        with self.store.atomic():
            reply = confirmed_text(self.store, job['id'], reason)
            actual_text = json.loads(job['payload'])['message'].get('text') or text
            self.store.finish(job['id'], actual_text, reply or reason, self.clock(), presented=bool(reply))
            if control_job is not None:
                await self._process_job(control_job, interrupted=True)

    async def _run_agent(self, job, text):
        task = None
        message = json.loads(job['payload'])['message']
        preparing = 'voice' in message and not message.get('text')
        deadline = asyncio.get_running_loop().time() + self.processing_timeout
        try:
            while True:
                if task is not None and task.done():
                    await task
                    if preparing:
                        task = None
                        preparing = False
                        # Inspect controls again before the agent can apply tools.
                        continue
                    return
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    await self._stop_agent(task, job, text,
                        'Превышен лимит времени обработки. Уже сохранённые действия не отменены. '
                        'Незавершённое поручение можно проверить через /request или отменить через /cancel.')
                    return
                # Inspect persisted commands before launching/resuming the agent:
                # a restored checkpoint may apply tools without a network wait.
                control = self._next_control(job['id'])
                if control:
                    control_job, command = control
                    stopping = (command == '/cancel' or command.casefold() in ('отмена', 'отмени поручение')
                                or command == '/new')
                    if stopping:
                        await self._stop_agent(task, job, text,
                            'Обработка текущего сообщения остановлена. Уже сохранённые действия не отменены.',
                            control_job=control_job)
                        return
                    await self._process_job(control_job)
                    continue
                if task is None:
                    task = asyncio.create_task(self._prepare_voice(job) if preparing else self.agent.process(job))
                await asyncio.wait({task}, timeout=min(.05, remaining))
        finally:
            # Shutdown preserves the checkpoint; explicit cancellation and the
            # deadline above complete the job so it cannot resume after restart.
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _prepare_voice(self, job):
        message = json.loads(job['payload'])['message']
        if self.voice is None:
            raise VoiceError('Приём голосовых выключен. Пришли текстовое сообщение.')
        config = self.voice.config
        audio = await self.telegram.download_voice(message['voice'], config.max_bytes, config.max_seconds)
        text = await self.voice.transcribe(audio)
        self.store.save_transcription(job, text)

    def schedule_once(self):
        now = self.clock()
        def render(r):
            schedule = self.store.db.execute('SELECT timezone FROM schedules WHERE id=?', (r['schedule_id'],)).fetchone() if r['schedule_id'] else None
            event = self.store.db.execute('SELECT e.timezone,o.start_date,o.start_time FROM events e JOIN event_occurrences o ON o.event_id=e.id WHERE o.id=?',(r['event_occurrence_id'],)).fetchone() if r['event_occurrence_id'] else None
            standalone=self.store.db.execute("SELECT timezone FROM schedules WHERE id=? AND local_date IS NOT NULL",(r['standalone_id'],)).fetchone() if r['standalone_id'] else None
            zone = event['timezone'] if event else standalone['timezone'] if standalone else schedule['timezone'] if schedule else timezone_name(self.store)
            delayed = '\nДоставлено после срока.' if now - r['due_at'] > 60 else ''
            event_text = ('\nСобытие: '+event['start_date']+(' '+event['start_time'] if event['start_time'] else ' (весь день)')) if event else ''
            return f"⏰ Напоминание\n{r['content']}\nСрок: {local_time(r['due_at'], zone)} ({zone_label(zone)}){event_text}{delayed}"
        self.store.prepare_reminders(now, render)

    async def deliver_once(self):
        row = self.store.next_outbox(self.clock())
        if not row:
            return False
        try:
            button = row['reminder_id'] if row['key'].endswith(':last') else None
            if row['reminder_id'] is not None:
                first_part = row['key'].endswith((':0000:part', ':0000:last'))
                text = notification_text(row['text'], first_part)
            elif row['presentation_done']:
                # The complete answer was formatted before chunking. A second
                # pass cannot see quotes that span Telegram messages.
                text = row['text']
            else:
                text = public_text(row['text'])
            if not self.store.start_delivery(row['key']):
                return True
            result = await self.telegram.send(self.owner, text, button)
            if (not isinstance(result, dict) or type(result.get('message_id')) is not int
                    or result['message_id'] <= 0):
                raise TelegramError(rejected=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning('Доставка отложена: %s', type(exc).__name__)
            self.store.retry(row['key'], self.clock(), getattr(exc, 'retry_after', None),
                             rejected=isinstance(exc, TelegramError) and exc.rejected and not row['delivery_started'])
        else:
            self.store.sent(row['key'], result['message_id'], self.clock())
        return True

    async def polling(self):
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning('Получение сообщений отложено: %s', type(exc).__name__)
                await asyncio.sleep(5)

    async def working(self):
        while True:
            if not await self.process_once():
                await asyncio.sleep(.25)

    async def scheduling(self):
        while True:
            self.schedule_once()
            await asyncio.sleep(1)

    async def delivering(self):
        while True:
            if not await self.deliver_once():
                await asyncio.sleep(.25)

    async def run(self):
        async with asyncio.TaskGroup() as group:
            for task in (self.polling, self.working, self.scheduling, self.delivering):
                group.create_task(task())
