import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from personal_assistent.agent import Agent
from personal_assistent.search import INDEX_VERSION, index_values, match_query, related_query
from personal_assistent.storage import Store
from personal_assistent.tools import ToolRunner
from test_assistant import NOW, Model, answer, call, update


class NoteSearchTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.store.enqueue(update('Сохрани памятку'), NOW)
        self.runner = ToolRunner(self.store)
        self.job = self.store.next_job()
        self.counter = 0

    def tearDown(self):
        self.store.db.close()

    def tool(self, name, **args):
        self.counter += 1
        result = self.runner.run(call(name, args, str(self.counter)), self.job, NOW)
        self.assertNotIn('error', result)
        return result

    def save(self, content):
        return self.tool('create_note', content=content)['id']

    def ids(self, query, **page):
        return [r['id'] for r in self.tool('search_notes', query=query, **page)['items']]

    def test_russian_noun_forms_find_original_note(self):
        identity = self.save('Я заменил фильтр')
        for query in ('фильтр', 'фильтра', 'фильтром', 'фильтры', 'фильтров'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [identity])

    def test_russian_verb_forms_match_in_both_directions(self):
        first = self.save('Я заменил фильтр')
        second = self.save('Мы заменили картридж')
        for query in ('заменить', 'заменила', 'заменили'):
            with self.subTest(query=query):
                self.assertEqual(set(self.ids(query)), {first, second})

    def test_declined_names_find_notes(self):
        olga = self.save('Оля обещала проверить духовку')
        maria = self.save('Маша забрала посылку')
        for query in ('Оле', 'Олю', 'Олей'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [olga])
        self.assertEqual(self.ids('Маше'), [maria])
        self.assertEqual(self.ids('Анне'), [])

    def test_case_yo_and_decomposed_unicode_share_search_representation(self):
        identity = self.save('Ёжик живёт в саду')
        for query in ('ежика', 'ЁЖИКОМ', 'Е\u0308ЖИКА'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [identity])

    def test_stressed_russian_words_remain_whole_and_match_inflections(self):
        identity = self.save('Я заменил фи\u0301льтр')
        for query in ('фильтр', 'фильтра', 'фи\u0301льтром'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [identity])
        unrelated = self.save('фи и льтр')
        self.assertNotIn(unrelated, self.ids('фи\u0301льтр'))

    def test_casefold_combining_mark_does_not_split_latin_place_names(self):
        identity = self.save('Встреча в İstanbul')
        unrelated = self.save('I')
        for query in ('istanbul', 'İSTANBUL', 'i\u0307stanbul'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [identity])
                self.assertNotIn(unrelated, self.ids(query))

    def test_combining_accent_normalization_does_not_merge_cyrillic_short_i_with_i(self):
        identity = self.save('Йога')
        self.assertEqual(self.ids('Йога'), [identity])
        self.assertEqual(self.ids('И\u0306ога'), [identity])
        self.assertEqual(self.ids('Иога'), [])

    def test_original_content_and_source_are_preserved(self):
        content = 'Оля: «Ёжик — заказ №42, ID АБ-123».\nЗаменили фильтр.'
        identity = self.save(content)
        found = self.tool('search_notes', query='фильтров')['items'][0]
        self.assertEqual(found['id'], identity)
        self.assertEqual(found['content'], content)
        self.assertEqual(found['original'], 'Сохрани памятку')

    def test_latin_numbers_and_mixed_product_codes_remain_literal(self):
        identity = self.save('Картридж HP LaserJet 1020 с кодом м6')
        for query in ('HP', 'laserjet', '1020', 'М6'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [identity])
        for query in ('102', 'H', 'м7', 'LaserJets'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [])

    def test_exact_word_match_has_higher_weight_than_stem_only_match(self):
        inflected = self.save('Фильтра')
        exact = self.save('Фильтр')
        self.assertEqual(self.ids('фильтр'), [exact, inflected])
        self.assertEqual(self.ids('фильтра'), [inflected, exact])

    def test_function_words_do_not_add_unrelated_results(self):
        target = self.save('Заменил фильтр')
        self.save('Я поговорил с Олей о поездке')
        self.save('Что было в коробке и на столе')
        self.assertEqual(self.ids('Что я делал с фильтром?'), [target])

    def test_function_word_only_query_does_not_search_arbitrary_notes(self):
        self.save('Я был в парке и на почте')
        result = self.tool('search_notes', query='Что я в и на?')
        self.assertEqual(result['items'], [])
        self.assertIn('list_notes', result['hint'])

    def test_negative_and_time_words_are_still_searchable(self):
        target = self.save('Кофе не пью; без сахара; после обеда; перед прогулкой')
        self.save('Кофе и сахар')
        for query in ('не', 'без', 'после', 'перед'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [target])

    def test_function_words_are_filtered_as_whole_words_not_stems(self):
        target = self.save('Радиатор Котова')
        self.assertEqual(self.ids('радиатор Котова'), [target])
        self.assertEqual(self.ids('ради Котова'), [target])

    def test_equivalent_query_forms_do_not_multiply_stem_relevance(self):
        inflected = self.save('Фильтров')
        exact = self.save('Духовка')
        expected = [exact, inflected]
        self.assertEqual(self.ids('фильтр духовка'), expected)
        self.assertEqual(self.ids('фильтр фильтра фильтром духовка'), expected)
        self.assertEqual(self.ids('фильтр фильтра фильтром фильтры духовка'), expected)

    def test_related_vocabulary_finds_independent_phrasings(self):
        for content, query in (
                ('Счёт за электроэнергию пришёл в среду', 'свет'),
                ('Ребёнок засыпает под музыку', 'со сном'),
                ('Рубашки передал благотворительному фонду', 'одежда'),
                ('Часы забрал из мастерской', 'ремонтировать'),
                ('Каникулы проведём в деревне', 'отдыхать')):
            with self.subTest(query=query):
                identity = self.save(content)
                self.assertIn(identity, self.ids(query))

    def test_direct_hits_suppress_expansion_on_every_page(self):
        self.save('Электроэнергия')
        direct = [self.save('Свет ' + str(i)) for i in range(7)]
        first = self.tool('search_notes', query='свет', limit=50)
        self.assertEqual([r['id'] for r in first['items']], direct[:5])
        self.assertEqual(first['total'], 7)
        self.assertEqual(first['next_offset'], 5)
        second = self.tool('search_notes', query='свет', offset=5)
        self.assertEqual([r['id'] for r in second['items']], direct[5:])
        self.assertIsNone(second['next_offset'])
        beyond = self.tool('search_notes', query='свет', offset=7)
        self.assertEqual(beyond['items'], [])
        self.assertEqual(beyond['total'], 7)
        self.assertIsNone(beyond['next_offset'])

    def test_single_original_hit_suppresses_expansion_even_if_irrelevant(self):
        original = self.save('Свет в саду')
        self.save('Электричество — 800 рублей')
        result = self.tool('search_notes', query='Сколько стоит свет?')
        self.assertEqual([n['id'] for n in result['items']], [original])
        self.assertEqual(result['total'], 1)

    def test_stem_only_original_hit_suppresses_expansion(self):
        original = self.save('Ремонтом доволен')
        self.save('Сервис закрылся')
        self.assertEqual(self.ids('ремонт'), [original])

    def test_deleted_original_hit_does_not_suppress_expansion(self):
        deleted = self.save('Свет')
        target = self.save('Электроэнергия')
        # Even a stale FTS row for a soft-deleted note must not block fallback.
        with self.store.db:
            self.store.db.execute('UPDATE notes SET deleted=1 WHERE id=?', (deleted,))
        result = self.tool('search_notes', query='свет')
        self.assertEqual([n['id'] for n in result['items']], [target])
        self.assertEqual(result['total'], 1)

    def test_fallback_with_no_related_candidates_returns_empty_page(self):
        self.save('Купили хлеб')
        result = self.tool('search_notes', query='ремонт')
        self.assertEqual(result['items'], [])
        self.assertEqual(result['total'], 0)
        self.assertIsNone(result['next_offset'])

    def test_creating_and_updating_original_match_switches_search_mode(self):
        related = self.save('Электроэнергия')
        self.assertEqual(self.ids('свет'), [related])
        original = self.save('Свет')
        self.assertEqual(self.ids('свет'), [original])
        self.tool('update_note', id=original, content='Купили хлеб', event_date=None)
        self.assertEqual(self.ids('свет'), [related])

    def test_related_results_are_unique_and_stably_paginated(self):
        expected = {self.save(text) for text in (
            'Мастерская', 'Сервис', 'Починка', 'Чинить часы',
            'В сервисе заменили стекло', 'Мастерская чинит часы')}
        visited = []
        offset = 0
        while True:
            result = self.tool('search_notes', query='ремонт', offset=offset, limit=2)
            self.assertEqual(result['total'], len(expected))
            visited.extend(r['id'] for r in result['items'])
            if result['next_offset'] is None:
                break
            offset = result['next_offset']
        self.assertEqual(set(visited), expected)
        self.assertEqual(len(visited), len(expected))
        self.assertEqual(self.ids('ремонт', offset=100), [])

    def test_related_search_uses_current_index_after_update_and_delete(self):
        identity = self.save('Платье передано в фонд')
        self.assertEqual(self.ids('одежда'), [identity])
        self.tool('update_note', id=identity, content='Часы в мастерской', event_date=None)
        self.assertEqual(self.ids('одежда'), [])
        self.assertEqual(self.ids('ремонт'), [identity])
        self.tool('delete_note', id=identity)
        self.assertEqual(self.ids('ремонт'), [])

    def test_related_search_preserves_facts_and_original(self):
        text = 'Анна не отдавала ZX42 в мастерскую; без оплаты до 2026-11-03, цена 1700'
        identity = self.save(text)
        found = self.tool('search_notes', query='ремонт')['items'][0]
        self.assertEqual(found['id'], identity)
        self.assertEqual(found['content'], text)
        self.assertEqual(self.tool('get_note', id=identity)['original'], 'Сохрани памятку')
        self.assertEqual(self.ids('ZX43'), [])
        self.assertEqual(self.ids('Антон'), [])

    def test_query_expansion_is_bounded_and_does_not_expand_identifiers(self):
        for query in ('Светлана', 'Света', 'Свету', 'Свете', 'Соня', 'Соню', 'Соне', 'Соней',
                      'Плата', 'свет42', 'sleep', '1700',
                      '2026-11-03', 'не без после перед через'):
            with self.subTest(query=query):
                self.assertEqual(related_query(query), '')
        # Original query limit also bounds expansion; filtered words do not use it.
        self.assertEqual(related_query(' '.join(f'x{i}' for i in range(30)) + ' ремонт'), '')
        self.assertEqual(related_query('что ' * 40 + 'ремонт'), related_query('ремонт'))
        self.assertEqual(related_query('ремонт ремонтом ремонты'), related_query('ремонт'))
        self.assertEqual(match_query('что ' * 40 + 'фильтр'), match_query('фильтр'))

    def test_name_stem_collisions_do_not_add_related_hits(self):
        electricity = self.save('Счёт за электричество')
        sleep = self.save('Не мог заснуть')
        for query in ('Света', 'Свету', 'Свете', 'Соня', 'Соню', 'Соне', 'Соней'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [])
        self.assertIn(electricity, self.ids('свет'))
        svetlana = self.save('Света вернула книгу')
        sonya = self.save('Соня забрала ключ')
        self.assertEqual(self.ids('Света'), [svetlana])
        self.assertEqual(self.ids('Соня'), [sonya])
        # Existing Snowball name collisions count as original hits too.
        self.assertEqual(self.ids('свет'), [svetlana])
        self.assertIn(sleep, self.ids('сном'))

    def test_or_search_returns_each_note_once_and_pagination_makes_progress(self):
        expected = {self.save(text) for text in (
            'Фильтр', 'Заменили фильтры', 'Оля купила фильтр', 'Поговорить с Олей', 'Олю встретили')}
        visited, offset = [], 0
        while True:
            result = self.tool('search_notes', query='фильтра Оле фильтры', offset=offset, limit=2)
            self.assertEqual(result['total'], len(expected))
            visited.extend(r['id'] for r in result['items'])
            if result['next_offset'] is None:
                break
            self.assertGreater(result['next_offset'], offset)
            offset = result['next_offset']
        self.assertEqual(set(visited), expected)
        self.assertEqual(len(visited), len(expected))
        self.assertEqual(self.ids('фильтра Оле фильтры'), visited)

    def test_update_and_delete_remove_obsolete_search_forms(self):
        identity = self.save('Заменил фильтр у Оли')
        self.tool('update_note', id=identity, content='Маша забрала картридж', event_date=None)
        self.assertEqual(self.ids('фильтров Оле'), [])
        self.assertEqual(self.ids('картриджей Маше'), [identity])
        self.tool('delete_note', id=identity)
        self.assertEqual(self.ids('картриджей Маше'), [])
        self.assertIsNone(self.store.db.execute('SELECT rowid FROM notes_fts WHERE rowid=?', (identity,)).fetchone())

    def test_shared_prefix_does_not_match_unrelated_words(self):
        self.save('Фильтрация воздуха и кофейник')
        self.assertEqual(self.ids('фильтр'), [])
        self.assertEqual(self.ids('кофе'), [])

    def test_typos_and_synonyms_are_not_invented_matches(self):
        self.save('Я заменил фильтр. Оля отказалась от сладких напитков.')
        for query in ('филтр', 'газировка'):
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), [])

    def test_punctuation_and_fts_syntax_are_treated_as_text(self):
        identity = self.save('Заменил фильтр')
        self.assertEqual(self.ids('"фильтра" OR * --'), [identity])
        self.assertEqual(self.ids('NEAR(молоко, хлеб) NOT id:999'), [])
        self.assertEqual(self.ids('"*() : --'), [])

    def test_failed_index_write_rolls_back_note_and_both_search_columns(self):
        identity = self.save('Оля заменила фильтр')
        with patch('personal_assistent.tools.index_values', side_effect=RuntimeError('index failure')):
            with self.assertRaises(RuntimeError):
                self.runner.run(call('update_note', {
                    'id': identity, 'content': 'Маша купила картридж', 'event_date': None}, 'broken-update'), self.job, NOW)
            with self.assertRaises(RuntimeError):
                self.runner.run(call('create_note', {'content': 'Новая памятка'}, 'broken-create'), self.job, NOW)
        self.assertEqual(self.ids('фильтров Оле'), [identity])
        self.assertEqual(self.ids('картриджей Маше'), [])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM notes').fetchone()[0], 1)


class SearchFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_synonym_and_typo_can_fall_back_to_model_reading_actual_notes(self):
        for text, query, description in (
                ('Оля отказалась от сладких напитков', 'газировка', 'Обещание Оли'),
                ('Я заменил фильтр', 'филтр', 'Замена фильтра')):
            with self.subTest(query=query):
                store = Store(':memory:')
                try:
                    store.enqueue(update('Запомни факт'), NOW)
                    ToolRunner(store).run(call('create_note', {'content': text}, 'save'), store.next_job(), NOW)
                    with store.db:
                        store.db.execute("UPDATE jobs SET status='done'")
                    store.enqueue(update('Найди памятку: ' + query, 2), NOW)
                    model = Model(answer(call('search_notes', {'query': query}, 'search')),
                                  answer(call('list_notes', {}, 'list')),
                                  answer(text=description))
                    await Agent(store, model, clock=lambda: NOW).process(store.next_job())
                    self.assertIn('"items": []', model.inputs[1][-1]['content'])
                    self.assertIn(text, model.inputs[2][-1]['content'])
                    self.assertEqual(store.history()[-1]['content'], description)
                finally:
                    store.db.close()


class SearchMigrationTests(unittest.TestCase):
    def legacy_store(self, path):
        store = Store(path)
        with store.atomic():
            for identity, content, deleted in (
                    (7, 'Заменил фильтр у Оли', 0), (8, 'Удалённый фильтр', 1), (9, 'Маша забрала картридж', 0)):
                store.db.execute('INSERT INTO notes(id,content,original,event_date,created_at,updated_at,source_message_id,deleted) VALUES(?,?,?,?,?,?,?,?)',
                                 (identity, content, 'Исходная просьба', '2026-10-16', NOW, NOW, 1, deleted))
            store.db.execute('DROP TABLE notes_fts')
            store.db.execute("CREATE VIRTUAL TABLE notes_fts USING fts5(content, tokenize='unicode61')")
            store.db.executemany('INSERT INTO notes_fts(rowid,content) VALUES(?,?)',
                                 ((7, 'Старый текст'), (8, 'Удалённый фильтр'), (99, 'Осиротевший текст')))
            store.set_meta('notes_search_version', 'previous-version')
            store.set_meta('sentinel', {'keep': True})
        store.enqueue(update('Найти фильтр'), NOW)
        rows = [dict(r) for r in store.db.execute('SELECT * FROM notes ORDER BY id')]
        store.db.close()
        return rows

    def test_legacy_index_rebuilds_from_active_notes_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'notes.sqlite3'
            original = self.legacy_store(path)
            store = Store(path)
            try:
                result = ToolRunner(store).run(call('search_notes', {'query': 'фильтром'}, 'search'), store.next_job(), NOW)
                self.assertEqual([r['id'] for r in result['items']], [7])
                self.assertEqual([dict(r) for r in store.db.execute('SELECT * FROM notes ORDER BY id')], original)
                self.assertEqual([r[0] for r in store.db.execute('SELECT rowid FROM notes_fts ORDER BY rowid')], [7, 9])
                self.assertEqual(store.get_meta('sentinel'), {'keep': True})
                self.assertEqual(store.get_meta('notes_search_version'), INDEX_VERSION)
            finally:
                store.db.close()
            with patch('personal_assistent.storage.index_values', side_effect=AssertionError('must not rebuild')):
                store = Store(path)
            try:
                result = ToolRunner(store).run(call('search_notes', {'query': 'Оле'}, 'again'), store.next_job(), NOW)
                self.assertEqual([r['id'] for r in result['items']], [7])
            finally:
                store.db.close()

    def test_failed_rebuild_rolls_back_schema_index_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'notes.sqlite3'
            original = self.legacy_store(path)
            def normalize(content):
                if content.startswith('Маша'):
                    raise RuntimeError('failed after indexing the first note')
                return index_values(content)
            with patch('personal_assistent.storage.index_values', side_effect=normalize):
                with self.assertRaises(RuntimeError):
                    Store(path)
            db = sqlite3.connect(path)
            db.row_factory = sqlite3.Row
            try:
                self.assertEqual([r['name'] for r in db.execute('PRAGMA table_info(notes_fts)')], ['content'])
                self.assertEqual([dict(r) for r in db.execute('SELECT * FROM notes ORDER BY id')], original)
                self.assertEqual(db.execute("SELECT value FROM meta WHERE key='notes_search_version'").fetchone()[0], '"previous-version"')
                self.assertEqual(db.execute("SELECT content FROM notes_fts WHERE rowid=7").fetchone()[0], 'Старый текст')
            finally:
                db.close()

    def test_normalizer_version_change_rebuilds_even_when_schema_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'notes.sqlite3'
            store = Store(path)
            store.enqueue(update('Сохрани факт'), NOW)
            runner = ToolRunner(store)
            runner.run(call('create_note', {'content': 'Я заменил фильтр'}, 'save'), store.next_job(), NOW)
            with store.atomic():
                store.db.execute("UPDATE notes_fts SET normalized='устаревшая основа'")
                store.set_meta('notes_search_version', 'outdated-normalizer-version')
            store.db.close()
            store = Store(path)
            try:
                result = ToolRunner(store).run(call('search_notes', {'query': 'фильтра'}, 'search'), store.next_job(), NOW)
                self.assertEqual([r['content'] for r in result['items']], ['Я заменил фильтр'])
                self.assertEqual(store.get_meta('notes_search_version'), INDEX_VERSION)
            finally:
                store.db.close()

    def test_previous_normalizer_rebuilds_split_accented_words_from_original_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'notes.sqlite3'
            store = Store(path)
            store.enqueue(update('Сохрани факт'), NOW)
            content = 'Фи\u0301льтр куплен в İstanbul'
            ToolRunner(store).run(call('create_note', {'content': content}, 'save'), store.next_job(), NOW)
            with store.atomic():
                store.db.execute('UPDATE notes_fts SET content=?,normalized=?',
                                 ('фи льтр куплен в i stanbul', 'фи льтр купл в i stanbul'))
                store.set_meta('notes_search_version', 'russian-snowball-1:3.1.1')
            store.db.close()
            store = Store(path)
            try:
                runner = ToolRunner(store)
                for number, query in enumerate(('фильтра', 'istanbul')):
                    result = runner.run(call('search_notes', {'query': query}, f'search-{number}'), store.next_job(), NOW)
                    self.assertEqual([r['content'] for r in result['items']], [content])
                self.assertEqual(store.get_meta('notes_search_version'), INDEX_VERSION)
            finally:
                store.db.close()
