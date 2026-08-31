# День 1. Первый запрос к LLM через API

Минимальный CLI-клиент отправляет текстовый запрос в `deepseek-v4-flash`, получает ответ и выводит его в консоль.

Для запуска нужен Python 3.10 или новее. Ключ автоматически загружается из локального файла `.env` с помощью `python-dotenv`.

## Подготовка

1. Создайте и активируйте виртуальное окружение, затем установите зависимость:

   ```bash
   cd day-01
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

2. Создайте API-ключ в [DeepSeek Platform](https://platform.deepseek.com/api_keys).
3. Скопируйте шаблон локальных настроек:

   ```bash
   cp .env.example .env
   ```

4. Замените `your_api_key_here` в `.env` настоящим ключом.

Выполнять `source .env` не нужно: программа сама загружает файл при запуске. `.env` исключён из Git — не публикуйте и не показывайте API-ключ в видео.

## Запуск

Передайте запрос аргументом:

```bash
python3 main.py "Объясни простыми словами, что такое API"
```

Либо запустите программу без аргумента и введите запрос интерактивно:

```bash
python3 main.py
```

Пример результата:

```text
DeepSeek:
API — это набор правил, с помощью которых программы обмениваются данными.
```

## Что происходит в коде

1. `python-dotenv` загружает `day-01/.env`, независимо от каталога, из которого запущена программа.
2. Ключ читается из переменной окружения `DEEPSEEK_API_KEY`. Уже заданная системная переменная имеет приоритет над значением из `.env`.
3. Программа отправляет `POST` на `https://api.deepseek.com/chat/completions`.
4. В запросе используется модель `deepseek-v4-flash` с отключённым thinking mode — для первого короткого примера он не нужен.
5. Ответ извлекается из `choices[0].message.content` и печатается в консоль.

Документация: [первый API-запрос](https://api-docs.deepseek.com/), [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/) и [python-dotenv](https://pypi.org/project/python-dotenv/).
