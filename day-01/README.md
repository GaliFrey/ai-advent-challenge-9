# День 1. Первый запрос к LLM через API

Минимальный CLI-клиент отправляет текстовый запрос в `deepseek-v4-flash`, получает ответ и выводит его в консоль.

Для запуска достаточно Python 3.9 или новее. Внешние библиотеки не нужны.

## Подготовка

1. Создайте API-ключ в [DeepSeek Platform](https://platform.deepseek.com/api_keys).
2. Скопируйте шаблон локальных настроек:

   ```bash
   cd day-01
   cp .env.example .env
   ```

3. Замените `your_api_key_here` в `.env` настоящим ключом и загрузите его в окружение:

   ```bash
   set -a
   source .env
   set +a
   ```

Файл `.env` исключён из Git. Не публикуйте и не показывайте API-ключ в видео.

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

1. Ключ читается из переменной окружения `DEEPSEEK_API_KEY`.
2. Программа отправляет `POST` на `https://api.deepseek.com/chat/completions`.
3. В запросе используется модель `deepseek-v4-flash` с отключённым thinking mode — для первого короткого примера он не нужен.
4. Ответ извлекается из `choices[0].message.content` и печатается в консоль.

Документация: [первый API-запрос](https://api-docs.deepseek.com/) и [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/).
