import json
import unittest

from judge import validate
import test_pipeline as fixtures
from test_pipeline import completion


class JudgeContractTests(unittest.TestCase):
    def test_rejects_malformed_missing_fields_and_inconsistent_verdict(self):
        for raw in ('not json', '[]', '{}', json.dumps({"verdict": "pass", "checks":
                    {"support": False, "coverage": True, "abstention": True}, "explanation": "Несовпадение"}),
                    json.dumps({"verdict": "pass", "checks": {"support": 1, "coverage": True, "abstention": True},
                                "explanation": "Число вместо bool"})):
            data, errors = validate(raw)
            self.assertIsNone(data)
            self.assertTrue(errors)


class JudgePipelineTests(unittest.TestCase):
    run_case = fixtures.PipelineTests.run_case

    def test_failure_verdict_preserves_answer(self):
        def complete(messages, key, model):
            if 'response' in json.loads(messages[1]['content']):
                return {'answer': json.dumps({'verdict': 'fail', 'checks': {'support': False, 'coverage': True,
                        'abstention': True}, 'explanation': 'Цитата не подтверждает утверждение.'}), 'usage': {}, 'llm_seconds': 0}
            return completion(messages, key, model)
        report = self.run_case(completion=complete)
        item = report['items'][0]
        self.assertEqual(report['status'], 'complete')
        self.assertTrue(item['checks']['passed'])
        self.assertIsNotNone(item['response'])
        self.assertEqual(item['judge']['assessment']['verdict'], 'fail')


    def test_malformed_assessment_is_invalid_and_answer_is_saved(self):
        def complete(messages, key, model):
            if 'response' in json.loads(messages[1]['content']):
                return {'answer': 'not json', 'usage': {}, 'llm_seconds': 0}
            return completion(messages, key, model)
        report = self.run_case(completion=complete)
        self.assertEqual(report['status'], 'invalid')
        item = report['items'][0]
        self.assertIsNotNone(item['response'])
        self.assertIsNone(item['judge']['assessment'])
        self.assertEqual(item['judge']['status'], 'invalid')


    def test_network_failure_preserves_generated_answer(self):
        def complete(messages, key, model):
            if 'response' in json.loads(messages[1]['content']):
                raise RuntimeError('secret-error')
            return completion(messages, key, model)
        report = self.run_case(completion=complete)
        self.assertEqual(report['error']['stage'], 'judge')
        self.assertEqual(report['items'][0]['judge']['status'], 'failed')
        self.assertIsNotNone(report['items'][0]['response'])
        self.assertNotIn('secret-error', json.dumps(report))
