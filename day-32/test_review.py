import json
import io
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

import httpx
import numpy as np

from provider import complete
from rag import retrieve
from review import markdown, prepare, run, save, validate
from snapshot import collect, revision
from verify_report import check
import main as cli


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}


class FakeModel:
    tokenizer = FakeTokenizer()

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            words = text.lower()
            vector = np.array([1, words.count('pitch'), words.count('hold'),
                               words.count('architecture'), words.count('test')], dtype=float)
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)


class RepositoryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Review Test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.write('README.md', '# Trainer\nArchitecture pitch hold must be continuous.\n')
        self.write('docs/audio.md', '# Audio\nRejected pitch frames interrupt hold.\n')
        self.write('src/pitch.js', 'export const hold = 200;\nexport const reliable = true;\n')
        self.write('src/app.js', 'import { hold } from "./pitch.js";\n// pitch test hold\n')
        self.write('test/pitch.test.js', '// test pitch hold rejects unreliable frames\n')
        self.base = self.commit()
        self.write('src/pitch.js', 'export const hold = 0;\nexport const reliable = true;\n')
        self.head = self.commit()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL).decode().strip()

    def write(self, path, content):
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding='utf-8')

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def snapshot(self):
        return collect(self.root, self.base, self.head)

    def finding(self):
        snapshot = self.snapshot()
        payload, sources, stats = prepare(snapshot, FakeModel())
        direct = next(s for s in sources if s.kind == 'head' and s.path == 'src/pitch.js')
        doc = next(s for s in sources if s.kind == 'docs')
        answer = {'summary': 'Обнаружена ошибка удержания.', 'findings': [
            {'category': 'bug', 'priority': 'P2', 'file': 'src/pitch.js', 'side': 'head',
             'line': 1, 'description': 'Нулевое удержание принимает одиночный кадр; сохраните hold 200.',
             'evidence': [{'source': direct.id, 'quote': 'export const hold = 0;'},
                          {'source': doc.id, 'quote': doc.text}]}]}
        return snapshot, sources, answer

    def test_dirty_worktree_does_not_change_snapshot(self):
        self.write('src/pitch.js', 'uncommitted secret-like unrelated text')
        snapshot = self.snapshot()
        self.assertEqual(snapshot['merge_base'], self.base)
        self.assertEqual(snapshot['direct'][1].text, 'export const hold = 0;\nexport const reliable = true;\n')
        self.assertEqual(snapshot['changes'][0]['lines'], {'base': {1}, 'head': {1}})

    def test_rag_selects_docs_and_related_code(self):
        snapshot = self.snapshot()
        sources, associations, count = retrieve(snapshot, FakeModel())
        self.assertGreater(count, 0)
        self.assertEqual({s.kind for s in sources}, {'docs', 'code'})
        self.assertTrue(associations['src/pitch.js'])
        self.assertNotIn('src/pitch.js', {s.path for s in sources})
        self.assertTrue(all(s.revision == self.head for s in sources))

    def test_new_deleted_renamed_and_space_paths(self):
        self.write('src/new file.js', 'export const added = true;\n')
        self.git('rm', 'src/app.js')
        self.git('mv', 'test/pitch.test.js', 'test/renamed.test.js')
        self.head = self.commit()
        changes = {c['path']: c for c in self.snapshot()['changes']}
        self.assertEqual(changes['src/new file.js']['lines']['head'], {1})
        self.assertEqual(changes['src/app.js']['lines']['base'], {1, 2})
        self.assertEqual(changes['test/renamed.test.js']['status'], 'A')
        self.assertEqual(changes['test/pitch.test.js']['status'], 'D')

    def test_diverged_base_uses_merge_base(self):
        original_head = self.head
        self.git('checkout', '-qb', 'other', self.base)
        self.write('src/unrelated.js', 'const x = 1;\n')
        base_tip = self.commit()
        snapshot = collect(self.root, base_tip, original_head)
        self.assertEqual(snapshot['base'], base_tip)
        self.assertEqual(snapshot['merge_base'], self.base)
        self.assertEqual([c['path'] for c in snapshot['changes']], ['src/pitch.js'])

    def test_symlink_not_followed(self):
        self.git('rm', 'src/pitch.js')
        (self.root / 'src/pitch.js').symlink_to('/etc/passwd')
        self.head = self.commit()
        with self.assertRaisesRegex(ValueError, 'Символические'):
            self.snapshot()

    def test_secrets_and_non_corpus_files_excluded(self):
        self.write('.env', 'NOT_A_REAL_KEY=fixture\n')
        self.write('PLAN.md', 'not indexed')
        self.write('node_modules/foo.js', 'not indexed')
        self.head = self.commit()
        with self.assertRaisesRegex(ValueError, 'секретов'):
            self.snapshot()
        self.base = self.head
        self.write('src/pitch.js', 'const pitch = 1;\n')
        self.head = self.commit()
        paths = {s.path for s in self.snapshot()['corpus']}
        self.assertFalse(paths & {'.env', 'PLAN.md', 'node_modules/foo.js'})

    def test_oversized_diff_fails_before_provider(self):
        with patch('snapshot.MAX_DIFF', 20):
            with self.assertRaisesRegex(ValueError, 'Diff больше'):
                self.snapshot()

    def test_invalid_revision_and_empty_change(self):
        for bad in ['HEAD', '--help', 'a' * 39, 'g' * 40]:
            with self.assertRaises(ValueError):
                revision(self.root, bad)
        with self.assertRaisesRegex(ValueError, 'непустой'):
            collect(self.root, self.base, self.base)

    def test_binary_explicitly_reported(self):
        (self.root / 'picture.bin').write_bytes(b'\0\x01\x02')
        self.head = self.commit()
        snapshot = self.snapshot()
        change = next(c for c in snapshot['changes'] if c['path'] == 'picture.bin')
        self.assertEqual(change['skipped'], ['head: binary/non-UTF8'])
        result = run(snapshot, FakeModel(), retrieve_only=True)
        self.assertIn('picture.bin — head: binary/non-UTF8', markdown(result))

    def test_accept_grounded_finding_and_empty_review(self):
        snapshot, sources, answer = self.finding()
        self.assertEqual(validate(json.dumps(answer), snapshot, sources), answer)
        answer['findings'] = []
        self.assertEqual(validate(json.dumps(answer), snapshot, sources)['findings'], [])

    def test_reject_unchanged_or_fabricated_locations(self):
        snapshot, sources, answer = self.finding()
        for field, value in [('line', 2), ('line', True), ('file', 'src/app.js'), ('side', 'invalid')]:
            candidate = json.loads(json.dumps(answer))
            candidate['findings'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate(json.dumps(candidate), snapshot, sources)

    def test_reject_fake_quote_unknown_source_wrong_side(self):
        snapshot, sources, answer = self.finding()
        for evidence in [[{'source': 'C999', 'quote': 'anything'}],
                         [{'source': 'C2', 'quote': 'not present'}],
                         [{'source': 'C1', 'quote': 'export const hold = 200;'}]]:
            candidate = json.loads(json.dumps(answer))
            candidate['findings'][0]['evidence'] = evidence
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate(json.dumps(candidate), snapshot, sources)

    def test_deleted_line_can_be_reviewed_on_base(self):
        snapshot, sources, answer = self.finding()
        f = answer['findings'][0]
        f['side'] = 'base'
        f['evidence'] = [{'source': 'C1', 'quote': 'export const hold = 200;'}]
        self.assertEqual(validate(json.dumps(answer), snapshot, sources), answer)

    def test_retrieve_only_never_calls_llm(self):
        def forbidden(*args):
            self.fail('LLM called in retrieve-only')
        result = run(self.snapshot(), FakeModel(), retrieve_only=True, completion=forbidden)
        self.assertEqual(result['status'], 'retrieved')
        self.assertTrue(result['payload']['sources'])

    def test_rejected_answer_not_presented_as_success(self):
        result = run(self.snapshot(), FakeModel(), completion=lambda *a: ('not json', {}))
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('answer', result)
        self.assertIn('Ревью не выполнено', markdown(result))
        output = self.root / 'output'
        save(result, output)
        saved = json.loads((output / 'review.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertNotIn('snapshot', saved)
        self.assertEqual(saved['rejected_response'], 'not json')

    def test_model_quote_is_replaced_with_source_text(self):
        snapshot, sources, answer = self.finding()
        quote = 'export const hold = 999;\n::error::untrusted text'
        answer['findings'][0]['evidence'][0]['quote'] = quote
        content = json.dumps(answer)
        result = run(snapshot, FakeModel(), completion=lambda *a: (content, {'usage': {'total_tokens': 10}}))
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['model_response'], content)
        evidence = result['answer']['findings'][0]['evidence'][0]
        self.assertEqual(evidence['source'], 'C2')
        self.assertNotEqual(evidence['quote'], quote)
        self.assertIn(evidence['quote'], next(s.text for s in result['sources'] if s.id == 'C2'))
        save(result, self.root / 'output')
        report = json.loads((self.root / 'output/review.json').read_text())
        self.assertEqual(report['model_response'], content)
        self.assertEqual(check(self.root, report), result['answer'])

    def test_source_ids_only_resolve_and_include_changed_lines(self):
        snapshot, sources, answer = self.finding()
        for evidence in answer['findings'][0]['evidence']:
            evidence.pop('quote')
        result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(answer), {}))
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['payload']['changed_lines']['src/pitch.js'], {'head': [1], 'base': [1]})
        for evidence in result['answer']['findings'][0]['evidence']:
            source = next(s for s in result['sources'] if s.id == evidence['source'])
            self.assertIn(evidence['quote'], source.text)
        self.assertEqual(check(self.root, result), result['answer'])

    def test_resolved_mode_still_rejects_invalid_locations_and_wrong_side(self):
        snapshot, sources, answer = self.finding()
        for evidence in answer['findings'][0]['evidence']:
            evidence.pop('quote')
        for invalid in ('line', 'side'):
            candidate = json.loads(json.dumps(answer))
            if invalid == 'line':
                candidate['findings'][0]['line'] = 2
            else:
                candidate['findings'][0]['evidence'][0]['source'] = 'C1'
            result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(candidate), {}))
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn('answer', result)
            self.assertIn('rejected_response', result)

    def test_unknown_source_diagnostics(self):
        snapshot, sources, answer = self.finding()
        answer['findings'][0]['evidence'][0]['source'] = 'R999'
        result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(answer), {}))
        self.assertEqual(result['validation_error']['reason'], 'unknown_source')
        self.assertEqual(result['validation_error']['source'], 'R999')
        self.assertIn('R999', result['error'])

    def test_rejected_response_redacts_key(self):
        snapshot, sources, answer = self.finding()
        key = 'fixture-secret-"with-quote'
        answer['summary'] += key
        answer['findings'][0]['evidence'][0]['source'] = key
        result = run(snapshot, FakeModel(), key=key, completion=lambda *a: (json.dumps(answer), {}))
        save(result, self.root / 'output')
        # JSON representation escapes quotation marks, so also inspect decoded fields.
        saved = json.loads((self.root / 'output/review.json').read_text())
        self.assertNotIn(key, saved['error'])
        self.assertNotIn(key, saved['validation_error']['source'])
        self.assertNotIn(key, json.loads(saved['rejected_response'])['summary'])

    def test_cli_reports_failure_reason_and_nonzero_exit(self):
        output = self.root / 'cli-output'
        with patch('sys.argv', ['main.py', '--project', str(self.root), '--base', self.base,
                              '--head', self.head, '--output', str(output)]), \
                patch('main.load_dotenv'), patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'fixture-key'}), \
                patch('main.load_model', return_value=FakeModel()), \
                patch('main.run', return_value={'status': 'failed', 'error': 'fixture validation error'}), \
                patch('main.save') as saved, redirect_stdout(io.StringIO()) as stdout, \
                redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(cli.main(), 1)
        saved.assert_called_once()
        self.assertIn('Статус: failed', stdout.getvalue())
        self.assertIn('Причина: fixture validation error', stderr.getvalue())

    def test_markdown_escapes_html_and_persists_exact_context(self):
        snapshot, sources, answer = self.finding()
        answer['summary'] = '<script>alert(1)</script>'
        result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(answer), {}))
        self.assertIn('&lt;script&gt;', markdown(result))
        self.assertNotIn('<script>', markdown(result))
        save(result, self.root / 'output')
        saved = json.loads((self.root / 'output/review.json').read_text())
        self.assertEqual(saved['payload']['head'], self.head)
        self.assertNotIn('DEEPSEEK_API_KEY', json.dumps(saved))

    def test_saved_report_revalidated_against_git(self):
        snapshot, sources, answer = self.finding()
        result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(answer), {}))
        self.assertEqual(check(self.root, result), result['answer'])
        result['payload']['changes'][0]['diff'] += 'invented change'
        with self.assertRaisesRegex(ValueError, 'Diff'):
            check(self.root, result)

    def test_saved_report_rejects_fabricated_rag_text(self):
        snapshot, sources, answer = self.finding()
        result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(answer), {}))
        source = next(s for s in result['payload']['sources'] if s['id'].startswith('R'))
        source['text'] = 'fabricated source fragment'
        with self.assertRaisesRegex(ValueError, 'RAG-фрагмент'):
            check(self.root, result)

    def test_saved_report_rejects_fabricated_changed_lines(self):
        snapshot, sources, answer = self.finding()
        result = run(snapshot, FakeModel(), completion=lambda *a: (json.dumps(answer), {}))
        result['payload']['changed_lines']['src/pitch.js']['head'] = [999]
        with self.assertRaisesRegex(ValueError, 'Допустимые строки'):
            check(self.root, result)


class ProviderCase(unittest.TestCase):
    def test_exact_payload_and_one_call(self):
        calls = []
        def handler(request):
            calls.append(request)
            payload = json.loads(request.content)
            self.assertIn('sources', json.loads(payload['messages'][1]['content']))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"summary":"ok","findings":[]}'}}], 'usage': {'total_tokens': 3}})
        content, metadata = complete({'sources': []}, 'test-key', 'test-model', httpx.MockTransport(handler))
        self.assertEqual(len(calls), 1)
        self.assertEqual(metadata['usage']['total_tokens'], 3)
        self.assertIn('findings', content)

    def test_errors_no_retry_no_key_in_message(self):
        for response in [httpx.Response(401, text='test-key'),
                         httpx.Response(429, text='test-key'),
                         httpx.Response(200, json={'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}),
                         httpx.Response(200, json={'choices': []})]:
            calls = []
            def handler(request):
                calls.append(request)
                return response
            with self.subTest(response=response), self.assertRaises(RuntimeError) as caught:
                complete({}, 'test-key', 'test-model', httpx.MockTransport(handler))
            self.assertEqual(len(calls), 1)
            self.assertNotIn('test-key', str(caught.exception))

    def test_network_error_and_missing_key(self):
        def handler(request):
            raise httpx.ConnectError('test-key', request=request)
        with self.assertRaisesRegex(RuntimeError, 'Сетевая'):
            complete({}, 'test-key', 'test-model', httpx.MockTransport(handler))
        with self.assertRaises(ValueError):
            complete({}, '', 'test-model', httpx.MockTransport(handler))


if __name__ == '__main__':
    unittest.main()
