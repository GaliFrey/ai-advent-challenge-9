"""Local stdio MCP, Ollama adapter and opt-in cloud streaming."""
import asyncio
import importlib.util
import json
import os
import time
import tomllib
from datetime import timedelta
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core import DAY, ANSWER_SCHEMA, PLAN_SCHEMA

_spec = importlib.util.spec_from_file_location('day28_ollama', DAY.parent / 'day-27/chat.py')
_local = importlib.util.module_from_spec(_spec)
# dataclasses resolve their declaring module through sys.modules.
import sys
sys.modules[_spec.name] = _local
_spec.loader.exec_module(_local)


class LocalClient:
    def __init__(self, transport=None):
        self.transport = transport

    async def generate(self, messages, settings, update, *, planning=False):
        settings.validate()
        body = {'model': settings.local_model, 'messages': messages, 'stream': True,
                'think': False if planning else settings.thinking,
                'format': PLAN_SCHEMA if planning else ANSWER_SCHEMA,
                'options': {'temperature': 0 if planning else settings.temperature,
                            'num_ctx': settings.num_ctx,
                            'num_predict': 512 if planning else settings.num_predict}}
        start, first, first_content, final = time.perf_counter(), None, None, None
        try:
            async with httpx.AsyncClient(base_url='http://127.0.0.1:11434', trust_env=False,
                    timeout=httpx.Timeout(300, connect=5), transport=self.transport) as client:
                async with client.stream('POST', '/api/chat', json=body) as response:
                    if response.status_code != 200:
                        raise ValueError(f'Ollama HTTP {response.status_code}; проверьте локальную модель и сервер.')
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get('error'):
                            raise ValueError('Ollama вернула ошибку генерации.')
                        message = data.get('message', {})
                        content, thinking = message.get('content') or '', message.get('thinking') or ''
                        elapsed = time.perf_counter() - start
                        if (content or thinking) and first is None:
                            first = elapsed
                        if content and first_content is None:
                            first_content = elapsed
                        if content or thinking:
                            update(content, thinking)
                        if data.get('done'):
                            final = data
                            break
        except httpx.ConnectError:
            raise ValueError('Нет соединения с Ollama на 127.0.0.1:11434. Запустите ollama serve.') from None
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
            raise ValueError('Ошибка соединения или формата потока Ollama.') from None
        if final is None:
            raise ValueError('Поток Ollama оборвался без итоговых метрик.')
        return _local.metrics_from_response(final, time.perf_counter() - start, first, first_content)


class CloudClient:
    def __init__(self, key='', transport=None):
        self.key, self.transport = key, transport

    async def generate(self, messages, settings, update):
        if not self.key.strip():
            raise ValueError('Задайте DEEPSEEK_API_KEY в day-28/.env или окружении.')
        payload = {'model': settings.cloud_model, 'messages': messages, 'stream': True,
                   'stream_options': {'include_usage': True}, 'temperature': settings.temperature,
                   'max_tokens': settings.num_predict, 'thinking': {'type': 'disabled'},
                   'response_format': {'type': 'json_object'}}
        started, first, finish, usage, done = time.perf_counter(), None, None, {}, False
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10), trust_env=False,
                    transport=self.transport) as client:
                async with client.stream('POST', 'https://api.deepseek.com/chat/completions',
                        headers={'Authorization': f'Bearer {self.key}'}, json=payload) as response:
                    if response.status_code != 200:
                        raise ValueError(f'DeepSeek HTTP {response.status_code}; повторов нет.')
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):
                            continue
                        value = line[5:].strip()
                        if value == '[DONE]':
                            done = True
                            break
                        data = json.loads(value)
                        if data.get('error'):
                            raise ValueError('DeepSeek вернул ошибку в потоке.')
                        if data.get('usage'):
                            usage = data['usage']
                        for choice in data.get('choices', []):
                            delta = choice.get('delta', {})
                            content, thinking = delta.get('content') or '', delta.get('reasoning_content') or ''
                            if content and first is None:
                                first = time.perf_counter() - started
                            if content or thinking:
                                update(content, thinking)
                            if choice.get('finish_reason'):
                                finish = choice['finish_reason']
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
            # Do not leak provider bodies, headers or credentials into sessions.
            raise ValueError('Ошибка соединения или формата потока DeepSeek; повторов нет.') from None
        if not done or finish is None:
            raise ValueError('Поток DeepSeek оборвался без завершения.')
        wall = time.perf_counter() - started
        return {'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
                'wall_seconds': wall, 'first_content_seconds': first, 'done_reason': finish,
                'tokens_per_second': None}


def mcp_configuration():
    command = os.getenv('WEBTUTOR_MCP_COMMAND')
    if command:
        args = json.loads(os.getenv('WEBTUTOR_MCP_ARGS', '[]'))
    else:
        # Read just the server block; never load credentials or unrelated MCP settings.
        config = Path.home() / '.codex/config.toml'
        lines, inside = [], False
        if config.exists():
            with config.open(encoding='utf-8') as file:
                for line in file:
                    if line.strip() == '[mcp_servers.webtutor-docs]':
                        inside = True
                        continue
                    if inside and line.lstrip().startswith('['):
                        break
                    if inside:
                        lines.append(line)
        data = tomllib.loads(''.join(lines))
        command, args = data.get('command'), data.get('args', [])
    if not isinstance(command, str) or not command or not isinstance(args, list) or any(
            not isinstance(arg, str) for arg in args):
        raise ValueError('Настройте webtutor-docs в Codex либо WEBTUTOR_MCP_COMMAND/ARGS в .env.')
    env = {name: value for name, value in os.environ.items() if name.startswith('WEBTUTOR_')
           and name not in ('WEBTUTOR_MCP_COMMAND', 'WEBTUTOR_MCP_ARGS')}
    return StdioServerParameters(command=command, args=args, env=env)


def decode_tool(result):
    if result.isError:
        raise ValueError('MCP вернул ошибку инструмента.')
    if result.structuredContent:
        return result.structuredContent
    text = '\n'.join(c.text for c in result.content if c.type == 'text')
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('MCP вернул неожиданный формат.')
    return data


class DocsClient:
    @asynccontextmanager
    async def connect(self):
        with open(os.devnull, 'w') as errors:
            async with stdio_client(mcp_configuration(), errlog=errors) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def call(self, session, name, args):
        return decode_tool(await session.call_tool(name, args, read_timeout_seconds=timedelta(seconds=90)))

    async def retrieve(self, queries, settings, notify=lambda text: None):
        async with self.connect() as session:
            status = await self.call(session, 'webtutor_docs_status', {})
            if status.get('cache', {}).get('state') != 'ready':
                raise ValueError('Локальный индекс MCP не готов. Подготовьте его отдельно.')
            searches, cards, seen = [], [], set()
            for query in queries:
                notify('Поиск: ' + query)
                response = await self.call(session, 'webtutor_search',
                    {'query': query, 'source': settings.source, 'limit': settings.top_k})
                searches.append(response)
                expanded = []
                for card in response.get('results', []):
                    expanded.append(card)
                    if settings.source == 'docs':
                        for other in card.get('alsoIn', []):
                            expanded.append({**other, 'heading': '', 'matchedBy': 'alsoIn'})
                response['read_candidates'] = expanded
                for card in expanded:
                    key = (card['ref'], card.get('chunkId'))
                    if key not in seen:
                        seen.add(key)
                        cards.append(card)
            # Round-robin coverage of queries; no second independent search per model.
            selected, selected_keys = [], set()
            for rank in range(settings.top_k):
                for search in searches:
                    results = search.get('read_candidates', [])
                    if rank < len(results):
                        card = results[rank]
                        key = (card['ref'], card.get('chunkId'))
                        if key not in selected_keys and len(selected) < settings.top_k:
                            selected_keys.add(key)
                            selected.append(card)
            sources = []
            for card in selected:
                notify('Чтение: ' + card.get('title', card['ref']))
                pages, incomplete = await self.read_card(session, card, settings)
                # Structured sections preserve warnings, parameters and origin.
                text = source_text(pages) if settings.source != 'schema' else json.dumps(pages, ensure_ascii=False, indent=2)
                if len(text) > settings.max_chars:
                    text = text[:settings.max_chars] + '\n[Материал усечён по лимиту контекста]'
                    incomplete = True
                sources.append({'source_id': f'S{len(sources)+1}', 'ref': card['ref'],
                    'title': card.get('title', ''), 'corpus': card.get('corpus', ''),
                    'chunk_id': card.get('chunkId'), 'heading': card.get('heading', ''),
                    'text': text, 'incomplete': incomplete, 'pages': pages})
            # pages kept in archive, not duplicated in generation context.
            return {'server': status, 'searches': searches, 'candidates': cards, 'sources': sources}

    async def read_card(self, session, card, settings):
        pages, incomplete = [], False
        if settings.source == 'schema':
            name = card.get('objectName')
            if not name:
                raise ValueError('MCP не вернул objectName для объекта схемы.')
            for section in ('overview', 'object_columns', 'catalog_columns', 'xml_fields'):
                args = {'object_name': name, 'section': section, 'limit': 20}
                for _ in range(2):
                    page = await self.call(session, 'webtutor_schema_read', args)
                    pages.append(page)
                    if page.get('complete', True):
                        break
                    cursor = page.get('nextCursor')
                    if cursor is None:
                        incomplete = True
                        break
                    args['cursor'] = cursor
                if not page.get('complete', True):
                    incomplete = True
        else:
            args = {'ref': card['ref'], 'sections': [], 'max_chars': settings.max_chars,
                    'include_links': True}
            if card.get('chunkId'):
                args['chunk_id'] = card['chunkId']
            for _ in range(3):
                page = await self.call(session, 'webtutor_read', args)
                pages.append(page)
                incomplete |= bool(page.get('truncated') or page.get('truncatedFields'))
                if not page.get('hasMore'):
                    break
                cursor = page.get('nextCursor')
                if cursor is None:
                    incomplete = True
                    break
                args['cursor'] = cursor
            incomplete |= bool(page.get('hasMore'))
        return pages, incomplete


def source_text(pages):
    sections = []
    for page in pages:
        if isinstance(page.get('summary'), dict):
            lead = page['summary'].get('lead')
            if lead:
                sections.append('Описание:\n' + lead)
        for name in ('syntax', 'parameters', 'returns', 'warnings'):
            value = page.get(name)
            if value:
                sections.append(name + ':\n' + (value if isinstance(value, str) else
                    json.dumps(value, ensure_ascii=False, indent=2)))
        if page.get('body'):
            sections.append(page['body'])
    return '\n\n'.join(sections)
