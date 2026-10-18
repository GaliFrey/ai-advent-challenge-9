from urllib.parse import quote
from .voice import VoiceError


class TelegramError(Exception):
    def __init__(self, code=0, retry_after=None, *, rejected=True):
        super().__init__(f'Telegram API error {code}')
        self.code, self.retry_after = code, retry_after
        self.rejected = rejected


class Telegram:
    def __init__(self, client, token):
        self.client = client
        self.base = 'https://api.telegram.org/bot' + token + '/'
        self.file_base = 'https://api.telegram.org/file/bot' + token + '/'

    async def download_voice(self, voice, max_bytes, max_seconds):
        if (not isinstance(voice, dict) or not isinstance(voice.get('file_id'), str)
                or not voice['file_id']):
            raise VoiceError('Некорректное голосовое сообщение.')
        if voice.get('file_size', 0) > max_bytes or voice.get('duration', 0) > max_seconds:
            raise VoiceError('Голосовое превышает лимит размера или длительности.')
        try:
            info = await self.call('getFile', file_id=voice['file_id'])
            path = info.get('file_path', '')
            if (not isinstance(path, str) or not path or path.startswith('/')
                    or any(p in ('', '.', '..') for p in path.split('/'))):
                raise VoiceError('Не удалось получить файл голосового.')
            if info.get('file_size', 0) > max_bytes:
                raise VoiceError('Голосовое превышает допустимый размер.')
            audio = bytearray()
            async with self.client.stream('GET', self.file_base + quote(path, safe='/'), timeout=40) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if len(audio) + len(chunk) > max_bytes:
                        raise VoiceError('Голосовое превышает допустимый размер.')
                    audio.extend(chunk)
            if not audio:
                raise VoiceError('Голосовое сообщение пустое.')
            return bytes(audio)
        except VoiceError:
            raise
        except Exception:
            raise VoiceError('Не удалось скачать голосовое из Telegram. Попробуй позже.') from None

    async def call(self, method, **payload):
        response = await self.client.post(self.base + method, json=payload, timeout=40)
        try:
            result = response.json()
        except ValueError:
            raise TelegramError(response.status_code, rejected=False) from None
        if not result.get('ok'):
            raise TelegramError(result.get('error_code', response.status_code),
                                result.get('parameters', {}).get('retry_after'), rejected=result.get('ok') is False)
        return result['result']

    async def updates(self, offset):
        return await self.call('getUpdates', offset=offset, timeout=25,
                               allowed_updates=['message', 'callback_query'])

    async def send(self, chat_id, text, reminder_id=None):
        payload = {'chat_id': chat_id, 'text': text}
        if reminder_id is not None:
            payload['reply_markup'] = {'inline_keyboard': [[{
                'text': 'Отложить на 10 минут', 'callback_data': f'snooze:{reminder_id}'}]]}
        return await self.call('sendMessage', **payload)

    async def acknowledge(self, callback_id, text=None):
        payload = {'callback_query_id': callback_id}
        if text:
            payload['text'] = text
        return await self.call('answerCallbackQuery', **payload)
