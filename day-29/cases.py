"""Six tuning and four holdout cases; criteria grounded in documents.json."""
import copy
import json
from core import DAY

# Criteria are evidence-based review requirements, not lexical scoring rules.
CASES = [
    ('opt-empty', 'tune', 'Что вернёт ArrayOptFirstElem для пустого массива без второго аргумента?',
     ['opt_datex', 'opt_portal'], ['Возвращает undefined для пустого массива без второго аргумента.'],
     ['Указать результат, сослаться на источник.'], ['Возвращает null или выбрасывает исключение для пустого массива.']),
    ('distinct-code', 'tune', 'Массив сотрудников уже получен в personArray. Покажи серверный SP-XML код удаления дублей по fullname через ArraySelectDistinct.',
     ['distinct_datex', 'distinct_portal'], ['Второй аргумент — строковое выражение; fullname и This.fullname показаны в Portal.'],
     ['Короткий код с ArraySelectDistinct(personArray, "fullname") либо "This.fullname".', 'Не утверждать, что пример выполнен на WebTutor.'],
     ['Стрелочные функции, Set или filter вместо документированного API.', 'Недокументированная гарантия порядка либо выбора первого/последнего дубля.']),
    ('corpus-difference', 'tune', 'Сравни описания ArrayOptFirstElem в Datex и Portal. Где описаны второй аргумент и undefined вместо массива?',
     ['opt_datex', 'opt_portal'], ['Datex описывает первый элемент и undefined для пустого массива.', 'Portal дополнительно описывает второй аргумент и ошибку при undefined вместо массива.'],
     ['Раздельные ссылки на оба корпуса.'], ['Приписывать Datex сведения, имеющиеся только в Portal.', 'Называть различие описаний доказанным различием версий реализации.']),
    ('schema-tables', 'tune', 'Какие таблица документа, каталог и XMD-формы соответствуют объекту collaborator?',
     ['schema'], ['documentTable: collaborator; catalogTable: collaborators.', 'Формы: x-local://wtv/wtv_collaborator.xmd и x-local://wtv/wtv_collaborators.xmd.'],
     ['Различить документ и каталог.'], ['Выдуманные SQL-колонки или XML-пути, отсутствующие в overview.']),
    ('missing-contract', 'tune', 'Какой точный HTTP endpoint и метод удаляют сотрудника через REST API WebTutor? Приведи URL и тело запроса.',
     ['opt_datex'], ['Передан только контракт ArrayOptFirstElem; оснований для HTTP endpoint нет.'],
     ['status=unknown, пустые citations, непустое gaps.', 'Не заявлять, что такого API вообще не существует.'], ['Выдумывать URL, метод или тело запроса.']),
    ('correct-history', 'tune', 'Ранее ты сказал, что ArrayOptFirstElem(undefined) возвращает undefined. Проверь и исправь ответ по текущим источникам.',
     ['opt_portal'], ['Portal описывает ошибку с остановкой выполнения при undefined вместо массива.'],
     ['Явно исправить прошлый ответ; отличить пустой массив от undefined.'], ['Использовать прошлый ответ как доказательство.']),
    ('first-vs-opt', 'holdout', 'Сравни ArrayFirstElem и ArrayOptFirstElem для пустого массива. Как получить значение 0 вместо undefined?',
     ['first_datex', 'opt_portal'], ['ArrayFirstElem для пустого массива завершается исключением.', 'ArrayOptFirstElem([], 0) возвращает 0.'],
     ['Различить обе функции, привести короткий вызов.'], ['Утверждать, что ArrayFirstElem поддерживает запасное значение.']),
    ('open-missing', 'holdout', 'Что делает tools.open_doc с несуществующим ID? Покажи проверку результата до обращения к TopElem.',
     ['open_portal'], ['Возвращает undefined без прерывания кода.', 'Аргумент iDocID — целое число; успешный результат XmlDoc.'],
     ['Проверка doc != undefined перед doc.TopElem в SP-XML.'], ['Утверждать, что несуществующий ID обязательно вызывает исключение.']),
    ('distinct-expression', 'holdout', 'Объясни аргументы ArraySelectDistinct и что используется для проверки уникальности, когда второй аргумент отсутствует.',
     ['distinct_datex'], ['Исходный массив обязателен, elemExpr — необязательное строковое выражение.', 'Без elemExpr используется сам элемент (This).'],
     ['Описание обоих вариантов вызова.'], ['Приписывать функцию изменения исходного массива или гарантию сортировки.']),
    ('schema-gap', 'holdout', 'Назови точный SQL-тип и XML-путь поля fullname сотрудника по переданному overview.',
     ['schema'], ['Overview не содержит списка колонок, типов и XML-путей fullname.'],
     ['Отказ либо явно обозначенный пробел без выдуманного типа/пути.'], ['Выдавать догадку fullname или varchar за подтверждённую схему.']),
]


def load_cases(split=None):
    documents = json.loads((DAY / 'documents.json').read_text(encoding='utf-8'))
    cases = []
    for identifier, group, question, keys, facts, required, forbidden in CASES:
        if split and group != split:
            continue
        sources = [dict(copy.deepcopy(documents['sources'][key]), source_id=f'S{i+1}')
                   for i, key in enumerate(keys)]
        memory = {'notes': 'Серверный SP-XML. Используй только переданные основания.',
                  'user_questions': [], 'prior_answers': []}
        if identifier == 'correct-history':
            memory.update(user_questions=['Что вернёт ArrayOptFirstElem(undefined)?'],
                          prior_answers=[{'answer': 'Возвращает undefined.', 'status': 'invalid',
                                          'model': 'fixed-history', 'gaps': ''}])
        cases.append({'id': identifier, 'split': group, 'question': question,
                      'sources': sources, 'memory': memory,
                      'criteria': {'facts': facts, 'required': required, 'forbidden': forbidden},
                      'evidence_retrieved_at': documents['retrieved_at']})
    return cases
