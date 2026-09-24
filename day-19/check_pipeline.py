"""Run the deployed MCP pipeline; --llm also verifies autonomous model calls."""
from __future__ import annotations
import argparse
import asyncio
import json
from agent import initial_history, result_data, run_turn
from mcp_client import connect, discover_tools
from servers import SERVERS
from trace_log import SessionLog
from pathlib import Path
from report_download import download_report

async def check(use_llm: bool) -> None:
    server = SERVERS[0]
    discovery = await discover_tools(server)
    expected = ['get_login_events', 'analyze_login_events', 'save_report']
    assert {t.name for t in discovery.tools} == set(expected)
    log = SessionLog(Path(__file__).parent / 'sessions')
    downloads = []
    calls = []
    results = {}
    def trace(label, data):
        log.append(label, data, server.name)
        if label == 'DOWNLOAD COMPLETE':
            downloads.append(data)
        if label == 'MCP CALL':
            calls.append(data['name'])
        if label == 'MCP RESULT':
            results[data['name']] = data['data']
    if use_llm:
        answer, _ = await run_turn(server, discovery.tools, initial_history(server),
            'Подготовь и сохрани отчёт об SSH-входах за последние сутки.', trace)
        trace('ANSWER', answer)
    else:
        async with connect(server) as client:
            args = {'hours':24}
            for name in expected:
                trace('MCP CALL', {'name':name,'arguments':args})
                result = await client.call_tool(name,args)
                if result.is_error:
                    raise RuntimeError(str(result_data(result)))
                data = result_data(result)
                trace('MCP RESULT',{'name':name,'data':data})
                if name == expected[0]:
                    args = {'snapshot_id':data['snapshot_id']}
                elif name == expected[1]:
                    args = {'report_id':data['report_id']}
    if not use_llm:
        trace('DOWNLOAD COMPLETE', await download_report(results['save_report']))
    assert downloads
    assert calls == expected, calls
    snapshot, report, saved = (results[name] for name in expected)
    assert snapshot['snapshot_id'] == report['snapshot_id'] == saved['snapshot_id']
    assert snapshot['events_sha256'] == report['events_sha256']
    assert report['report_id'] == saved['report_id']
    assert snapshot['event_count'] == report['total'] == sum(report['by_user'].values()) == sum(report['by_ip'].values())
    print(json.dumps({'mode':'llm' if use_llm else 'direct','calls':calls,'total':report['total'],'saved':saved,'download':downloads[-1],'session':str(log.path)},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm',action='store_true',help='Make paid DeepSeek calls to test autonomous orchestration')
    asyncio.run(check(parser.parse_args().llm))
