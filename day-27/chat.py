"""Session storage, bounded conversational history and local Ollama streaming."""
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

DAY = Path(__file__).resolve().parent
SYSTEM = 'Ты полезный собеседник. Отвечай по-русски, если пользователь не попросил другой язык.'


@dataclass
class Settings:
    model: str = 'qwen3:14b'
    thinking: bool = False
    temperature: float = 0.7
    num_ctx: int = 8192
    num_predict: int = 2048

    def validate(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('Укажите модель, например qwen3:14b.')
        if type(self.thinking) is not bool:
            raise ValueError('Thinking должен быть включён или выключен.')
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError('Temperature: число от 0 до 2.')
        if type(self.num_ctx) is not int or not 2048 <= self.num_ctx <= 32768:
            raise ValueError('Контекст: целое число от 2048 до 32768.')
        if type(self.num_predict) is not int or not 1 <= self.num_predict <= self.num_ctx // 2:
            raise ValueError('Лимит генерации: от 1 до половины контекста.')
        return self


def new_session():
    return {'version': 1, 'id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid4().hex[:8],
            'created_at': datetime.now(timezone.utc).isoformat(), 'title': 'Новый диалог',
            'settings': asdict(Settings()), 'turns': []}


def save_session(session, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (session['id'] + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path


def load_session(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('version') != 1 or not isinstance(data.get('turns'), list):
        raise ValueError('Неподдерживаемый формат сессии.')
    if not isinstance(data.get('id'), str) or Path(data['id']).name != data['id'] or data['id'] in ('', '.', '..'):
        raise ValueError('Некорректный идентификатор сессии.')
    Settings(**data['settings']).validate()
    for turn in data['turns']:
        if not isinstance(turn.get('user'), str) or not isinstance(turn.get('content'), str):
            raise ValueError('Повреждённое сообщение сессии.')
        if not isinstance(turn.get('thinking'), str) or not isinstance(turn.get('metrics'), dict):
            raise ValueError('Повреждённые данные ответа.')
        if turn.get('status') == 'running':
            turn['status'] = 'interrupted'
            turn['error'] = 'Приложение закрыто до завершения ответа.'
    return data


def prepare_messages(session, question, settings):
    """Conservative byte budget; not an exact tokenizer for arbitrary models."""
    settings.validate()
    messages = [{'role': 'system', 'content': SYSTEM}]
    pairs = [[{'role': 'user', 'content': t['user']}, {'role': 'assistant', 'content': t['content']}]
             for t in session['turns'] if t['status'] == 'complete' and t['content']]
    total = len(pairs)
    # Reserve template overhead and the maximum generation, including thinking.
    budget = settings.num_ctx - settings.num_predict - 512
    def size(items):
        return sum(len(m['content'].encode('utf-8')) + 32 for m in items)
    current = {'role': 'user', 'content': question}
    if size(messages + [current]) > budget:
        raise ValueError('Сообщение слишком длинное для выбранного контекста. Сократите его или увеличьте контекст.')
    while pairs and size(messages + [m for pair in pairs for m in pair] + [current]) > budget:
        pairs.pop(0)
    messages += [m for pair in pairs for m in pair] + [current]
    return messages, {'included': len(pairs), 'excluded': total - len(pairs),
                      'budget': budget, 'estimated_bytes': size(messages)}


def metrics_from_response(data, wall, first, first_content):
    count, duration = data.get('eval_count'), data.get('eval_duration')
    return {'input_tokens': data.get('prompt_eval_count'), 'output_tokens': count,
            'tokens_per_second': count * 1e9 / duration if count is not None and duration else None,
            'wall_seconds': wall, 'first_fragment_seconds': first,
            'first_content_seconds': first_content,
            'load_seconds': data.get('load_duration', 0) / 1e9,
            'generation_seconds': duration / 1e9 if duration is not None else None,
            'server_seconds': data.get('total_duration', 0) / 1e9,
            'done_reason': data.get('done_reason', 'unknown')}


class OllamaClient:
    def __init__(self, *, transport=None):
        self.transport = transport

    async def generate(self, messages, settings, update):
        settings.validate()
        body = {'model': settings.model.strip(), 'messages': messages, 'stream': True,
                'think': settings.thinking, 'options': {'temperature': settings.temperature,
                'num_ctx': settings.num_ctx, 'num_predict': settings.num_predict}}
        started, first, first_content = time.perf_counter(), None, None
        final = None
        try:
            async with httpx.AsyncClient(base_url='http://127.0.0.1:11434', trust_env=False,
                    timeout=httpx.Timeout(300, connect=5), transport=self.transport) as client:
                async with client.stream('POST', '/api/chat', json=body) as response:
                    if response.status_code != 200:
                        await response.aread()
                        if response.status_code == 404:
                            raise ValueError('Модель не найдена. Скачайте её через ollama pull или измените имя.')
                        raise ValueError(f'Ollama вернула HTTP {response.status_code}: {response.text[:300]}')
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get('error'):
                            raise ValueError(f"Ollama: {data['error']}")
                        message = data.get('message', {})
                        content, thinking = message.get('content', ''), message.get('thinking', '')
                        elapsed = time.perf_counter() - started
                        if (content or thinking) and first is None:
                            first = elapsed
                        if content and first_content is None:
                            first_content = elapsed
                        if content or thinking:
                            update(content, thinking)
                        if data.get('done'):
                            final = data
                            break
        except httpx.ConnectError as error:
            raise ValueError('Нет соединения с Ollama на 127.0.0.1:11434. Запустите ollama serve.') from error
        except httpx.TimeoutException as error:
            raise ValueError('Ollama не ответила вовремя (300 с без данных).') from error
        except (json.JSONDecodeError, httpx.HTTPError) as error:
            raise ValueError(f'Ошибка потока Ollama: {error}') from error
        if final is None:
            raise ValueError('Поток оборвался без итоговых метрик; ответ не считается завершённым.')
        return metrics_from_response(final, time.perf_counter() - started, first, first_content)
