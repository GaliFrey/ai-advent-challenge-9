from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from mcp import Client, StdioServerParameters
import agent
from servers import Server


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_transfers_ids_and_saves_exact_snapshot_through_mcp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dbpath = root / 'logins.sqlite3'
            now = datetime.now(timezone.utc).isoformat()
            with closing(sqlite3.connect(dbpath)) as db, db:
                db.executescript('CREATE TABLE logins(cursor TEXT, occurred_at TEXT, username TEXT, ip TEXT, method TEXT); CREATE TABLE metadata(key TEXT, value TEXT);')
                db.executemany('INSERT INTO logins VALUES(?,?,?,?,?)', [('1',now,'alice','192.0.2.1','publickey'),('2',now,'alice','192.0.2.2','publickey'),('3',now,'bob','192.0.2.1','publickey')])
                db.execute('INSERT INTO metadata VALUES(?,?)', ('last_success', now))
            target = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name('server.py'))], env={'DAY19_DB_PATH':str(dbpath),'DAY19_REPORT_DIR':str(root / 'reports')})
            async with Client(target, mode='legacy') as client:
                tools = tuple((await client.list_tools()).tools)
                self.assertEqual({t.name for t in tools}, {'get_login_events','analyze_login_events','save_report'})
                @asynccontextmanager
                async def connect(_):
                    yield client
                count = 0
                async def completion(http, key, messages, offered):
                    nonlocal count
                    count += 1
                    if count == 1:
                        name, args = 'get_login_events', {'hours':24}
                    elif count == 2:
                        snapshot = json.loads(messages[-1]['content'])
                        self.assertEqual(snapshot['event_count'], 3)
                        # Change source after step 1: analysis must retain the original snapshot.
                        with closing(sqlite3.connect(dbpath)) as db, db:
                            db.execute('DELETE FROM logins')
                        name, args = 'analyze_login_events', {'snapshot_id':snapshot['snapshot_id']}
                    elif count == 3:
                        report = json.loads(messages[-1]['content'])
                        self.assertEqual(report['total'], 3)
                        self.assertEqual(report['by_user'], {'alice':2,'bob':1})
                        self.assertEqual(report['unique_ips'], 2)
                        name, args = 'save_report', {'report_id':report['report_id'],'filename':'test.md'}
                    else:
                        saved = json.loads(messages[-1]['content'])
                        self.assertIn('local_path', saved)
                        content = Path(saved['path']).read_bytes()
                        self.assertEqual(hashlib.sha256(content).hexdigest(), saved['sha256'])
                        self.assertIn('Всего входов: 3', content.decode())
                        self.assertEqual(Path(saved['path']).stat().st_mode & 0o777, 0o600)
                        return {'choices':[{'message':{'role':'assistant','content':'Отчёт сохранён на VM.'}}]}
                    return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':str(count),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}
                async def download(saved):
                    return {'local_path':saved['path'], 'bytes':saved['bytes'], 'sha256':saved['sha256']}
                traces = []
                with patch.object(agent,'connect',connect), patch.object(agent,'api_key',return_value='test'), patch.object(agent,'completion',completion), patch.object(agent,'download_report',download):
                    answer, history = await agent.run_turn(Server('test','local',()),tools,[],'Сохрани отчёт',lambda label,data:traces.append((label,data)))
                self.assertEqual([data['name'] for label,data in traces if label=='MCP CALL'], ['get_login_events','analyze_login_events','save_report'])
                report_id = next(data['data']['report_id'] for label,data in traces if label=='MCP RESULT' and data['name']=='analyze_login_events')
                dated = agent.result_data(await client.call_tool('save_report', {'report_id':report_id}))
                self.assertRegex(Path(dated['path']).name, r'^ssh-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.md$')
                self.assertEqual(Path(dated['path']).read_bytes(), (root / 'reports/test.md').read_bytes())
                for name, args in [('save_report' ,{'report_id':report_id,'filename':'../escape.md'}),('save_report',{'report_id':report_id,'filename':'test.md'}),('analyze_login_events',{'snapshot_id':'invented'}),('get_login_events',{'hours':0})]:
                    self.assertTrue((await client.call_tool(name,args)).is_error)
                empty = agent.result_data(await client.call_tool('get_login_events',{'hours':1}))
                empty_report = agent.result_data(await client.call_tool('analyze_login_events',{'snapshot_id':empty['snapshot_id']}))
                self.assertEqual(empty_report['total'],0)
                self.assertEqual(empty_report['by_ip'],{})

    async def test_mcp_error_stops_agent_before_save(self):
        from mcp import types
        class FakeClient:
            async def call_tool(self, name, args):
                return types.CallToolResult(is_error=True,content=[types.TextContent(type='text',text='Unknown snapshot')])
        @asynccontextmanager
        async def connect(_):
            yield FakeClient()
        async def completion(*args):
            return {'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'1','type':'function','function':{'name':'analyze_login_events','arguments':'{"snapshot_id":"bad"}'}}]}}]}
        tool = types.Tool(name='analyze_login_events', input_schema={'type':'object'})
        with patch.object(agent,'connect',connect), patch.object(agent,'api_key',return_value='test'), patch.object(agent,'completion',completion):
            with self.assertRaisesRegex(RuntimeError,'Unknown snapshot'):
                await agent.run_turn(Server('test','local',()),(tool,),[],'report',lambda *_:None)

if __name__ == '__main__':
    unittest.main()
