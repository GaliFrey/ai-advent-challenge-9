// Панель «Помощь»: локальная переписка с сервером поддержки тюнера.
//
// Модуль не управляет звуком: он не импортирует capture.js/display.js, не
// обращается к getUserMedia, AudioContext или аудиобуферам и не останавливает
// захват. Диагностический снимок берётся только в момент отправки сообщения
// через существующую функцию collectSnapshot(). DeepSeek получает текстовый
// контекст; локальная E5 используется только для поиска материалов.

import { collectSnapshot } from "./diagnostics.js"

export const AI_NOTICE =
  "Ответы формирует DeepSeek. Сообщения, история текущего обращения и состояние тюнера передаются во внешний AI-сервис. Аудиозапись не отправляется. AI может ошибаться."

export function appendSources(doc, item, sources = [], legacy = false) {
  for (const source of sources) {
    const details = doc.createElement("details")
    details.className = "support-source"
    const summary = doc.createElement("summary")
    summary.textContent = `${legacy ? "Материал-кандидат (прежний формат)" : "Источник"}: ${source.title} / ${(source.headings || []).slice(1).join(" / ")}`
    const fragment = doc.createElement("pre")
    fragment.textContent = source.text
    details.append(summary, fragment)
    if (typeof source.url === "string" && /^\/api\/support\/conversations\/\d+\/messages\/\d+\/sources\/[A-Za-z0-9_-]+$/.test(source.url)) {
      const link = doc.createElement("a")
      link.href = source.url
      link.textContent = "Открыть сохранённый материал"
      details.append(link)
    }
    item.append(details)
  }
}

const CSRF_HEADERS = { "X-Support-Request": "1" }

// Тонкая обёртка над fetch: cookies same-origin, единый разбор ошибок API.
export function createApi({ fetchImpl = fetch, base = "" } = {}) {
  async function request(path, options = {}) {
    const response = await fetchImpl(base + path, {
      credentials: "same-origin",
      ...options,
      headers: { ...(options.headers || {}) },
    })
    let data = {}
    try {
      data = await response.json()
    } catch {
      data = {}
    }
    if (!response.ok) {
      const error = new Error(
        data && data.error ? data.error.message : "Ошибка запроса к серверу поддержки.",
      )
      error.code = data && data.error ? data.error.code : "request_failed"
      error.status = response.status
      throw error
    }
    return data
  }

  return {
    session: () => request("/api/support/session"),
    newConversation: () =>
      request("/api/support/conversations", {
        method: "POST",
        headers: { ...CSRF_HEADERS },
      }),
    history: (conversationId) =>
      request(`/api/support/conversations/${conversationId}/messages`),
    sendMessage: (conversationId, { text, snapshot, key }) =>
      request(`/api/support/conversations/${conversationId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...CSRF_HEADERS },
        body: JSON.stringify({ text, snapshot, idempotencyKey: key }),
      }),
  }
}

// Логика переписки без DOM: её можно проверять Node-тестами с подставным API.
export function createSupportController({ api, snapshotFactory, keyFactory }) {
  let conversationId = null
  let pending = null

  async function load() {
    const session = await api.session()
    conversationId = session.currentConversationId
    if (conversationId === null || conversationId === undefined) {
      const created = await api.newConversation()
      conversationId = created.conversationId
    }
    return api.history(conversationId)
  }

  async function refresh() {
    if (conversationId === null) return load()
    return api.history(conversationId)
  }

  async function newConversation() {
    const created = await api.newConversation()
    conversationId = created.conversationId
    pending = null
    return api.history(conversationId)
  }

  // Отправка: снимок собирается только здесь. При неудаче тот же текст, снимок и
  // ключ сохраняются для идемпотентного повтора.
  async function send(text) {
    if (pending === null) {
      pending = { text, snapshot: snapshotFactory(), key: keyFactory() }
    }
    const result = await api.sendMessage(conversationId, pending)
    pending = null
    return result
  }

  return {
    load,
    refresh,
    newConversation,
    send,
    retry: async (message) => {
      pending = { text: message.text, snapshot: message.snapshot, key: message.idempotencyKey }
      return send(message.text)
    },
    hasPending: () => pending !== null,
    pendingText: () => pending?.text,
    clearPending: () => {
      pending = null
    },
    get conversationId() {
      return conversationId
    },
  }
}

// Привязка к DOM. Пустые элементы или отсутствие browser-среды безопасны.
export function initSupport({
  doc = document,
  api = createApi(),
  snapshotFactory = collectSnapshot,
  keyFactory = () => globalThis.crypto.randomUUID(),
} = {}) {
  const openButton = doc.getElementById("support-open")
  const panel = doc.getElementById("support-panel")
  const closeButton = doc.getElementById("support-close")
  const newButton = doc.getElementById("support-new")
  const retryButton = doc.getElementById("support-retry")
  const form = doc.getElementById("support-form")
  const input = doc.getElementById("support-input")
  const sendButton = doc.getElementById("support-send")
  const messagesEl = doc.getElementById("support-messages")
  const statusEl = doc.getElementById("support-status")
  const noticeEl = doc.getElementById("support-notice")

  if (!openButton || !panel || !form || !messagesEl) return null

  if (noticeEl) noticeEl.textContent = AI_NOTICE

  const controller = createSupportController({ api, snapshotFactory, keyFactory })
  let busy = false
  let lastErrorCode = null

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text
  }

  function setRetryVisible(visible) {
    if (retryButton) retryButton.hidden = !visible
  }

  // Текст сообщения выводится через textContent: HTML остаётся текстом и не
  // исполняется.
  function renderMessages(messages) {
    messagesEl.replaceChildren()
    for (const message of messages) {
      const item = doc.createElement("li")
      item.className = `support-message support-message-${message.role}`

      const role = doc.createElement("span")
      role.className = "support-role"
      role.textContent = message.role === "user" ? "Вы" : "Поддержка · DeepSeek"

      const text = doc.createElement("p")
      text.className = "support-text"
      text.textContent = message.text

      item.append(role, text)
      appendSources(doc, item, message.sources, message.sourcesLegacy)

      if (message.snapshot) {
        const details = doc.createElement("details")
        const summary = doc.createElement("summary")
        summary.textContent = "Диагностический снимок (версия 1)"
        const pre = doc.createElement("pre")
        pre.textContent = JSON.stringify(message.snapshot, null, 2)
        details.append(summary, pre)
        item.append(details)
      }

      if (message.state === "error") {
        const error = doc.createElement("p")
        error.className = "support-error"
        error.textContent = ({
          corpus_missing: "Нет статей для поиска.",
          corpus_invalid: "Корпус статей недоступен или содержит недопустимые файлы.",
          model_missing: "Модель поиска не подготовлена или повреждена.",
          model_incompatible: "Окружение модели поиска несовместимо.",
          index_missing: "Индекс поиска ещё не построен.",
          index_stale: "Статьи изменились. Индекс поиска устарел.",
          index_corrupt: "Индекс поиска повреждён.",
          index_incompatible: "Версия индекса несовместима.",
          query_too_long: "Вопрос слишком длинный. Сократите его и отправьте как новый вопрос.",
          mcp_error: "Не удалось получить контекст обращения через MCP.",
          retrieval_failed: "Сбой локального поиска.",
          storage_failed: "Не удалось сохранить результат обработки. Повторите явно.",
          provider_failed: "Сбой внешнего вызова. Его исход может быть неизвестен; повторите явно.",
          provider_key_missing: "Ключ DeepSeek не настроен на сервере. После настройки повторите явно.",
          provider_auth: "Ключ DeepSeek не принят. Нужна настройка сервера.",
          provider_access: "Нет доступа к DeepSeek.",
          provider_balance: "Недостаточно средств DeepSeek.",
          provider_request: "DeepSeek отклонил параметры запроса.",
          provider_rate_limit: "DeepSeek ограничил частоту запросов. Повторите позже явно.",
          provider_unavailable: "DeepSeek временно недоступен.",
          provider_http: "Ошибка сервиса DeepSeek.",
          provider_network: "Сетевой сбой. Внешний исход может быть неизвестен; повторите явно.",
          provider_timeout: "Истекло время ожидания DeepSeek. Внешний исход может быть неизвестен.",
          provider_contract: "AI вернул некорректный ответ. Можно повторить явно.",
          provider_output_limit: "AI достиг предела длины ответа. Можно повторить явно.",
          context_budget: "Контекст слишком большой. Сократите вопрос или начните новый разговор.",
          context_invalid: "Контекст текущего вопроса недоступен.",
          session_limit: "Достигнут лимит внешних попыток сессии. Повторите после окончания окна.",
          generation_busy: "Сервис занят другим ответом. Повторите явно позже.",
          processing_timeout: "Истекло общее время обработки. Внешний исход может быть неизвестен.",
          interrupted: "Обработка прервана перезапуском. Повтор возможен только явно.",
          cancelled: "Обработка отменена. Внешний исход может быть неизвестен; повторите явно.",
        })[message.errorCode] || "Обработка прервана, сообщение можно отправить заново."
        item.append(error)
        if (message.role === "user" && message.idempotencyKey && message.snapshot) {
          const retry = doc.createElement("button")
          retry.type = "button"
          retry.textContent = "Повторить это сообщение"
          retry.addEventListener("click", () => retrySaved(message))
          item.append(retry)
        }
      }

      messagesEl.append(item)
    }
  }

  async function refresh() {
    try {
      const data = await controller.refresh()
      renderMessages(data.messages)
      setStatus("")
    } catch {
      setStatus("Не удалось загрузить переписку. Проверьте, что локальный сервер запущен.")
    }
  }

  async function openPanel() {
    panel.hidden = false
    await refresh()
  }

  function closePanel() {
    panel.hidden = true
  }

  async function retrySaved(message) {
    if (busy) return
    busy = true
    if (sendButton) sendButton.disabled = true
    setStatus("Ожидаем ответ DeepSeek…")
    try {
      await controller.retry(message)
      setRetryVisible(false)
      await refresh()
    } catch (error) {
      setStatus(error.message || "Не удалось повторить сообщение.")
      setRetryVisible(true)
    } finally {
      busy = false
      if (sendButton) sendButton.disabled = false
    }
  }

  async function submit() {
    if (busy) return
    const text = input.value
    if (!controller.hasPending() && !text.trim()) {
      setStatus("Введите сообщение.")
      return
    }
    busy = true
    if (sendButton) sendButton.disabled = true
    setRetryVisible(false)
    setStatus("Ожидаем ответ DeepSeek…")
    try {
      await controller.send(text)
      lastErrorCode = null
      input.value = ""
      await refresh()
    } catch (error) {
      lastErrorCode = error.code
      setStatus(
        `${error.message || "Не удалось отправить сообщение."} Нажмите «Повторить»: тот же текст и снимок уйдут повторно без дубля.`,
      )
      setRetryVisible(true)
      try {
        renderMessages((await controller.refresh()).messages)
      } catch {
        // Preserve the original safe error if history cannot be refreshed.
      }
    } finally {
      busy = false
      if (sendButton) sendButton.disabled = false
    }
  }

  async function startNewConversation() {
    if (busy) return
    busy = true
    try {
      controller.clearPending()
      const data = await controller.newConversation()
      renderMessages(data.messages)
      input.value = ""
      setRetryVisible(false)
      setStatus("Создан новый разговор.")
    } catch {
      setStatus("Не удалось создать новый разговор.")
    } finally {
      busy = false
    }
  }

  openButton.addEventListener("click", openPanel)
  if (closeButton) closeButton.addEventListener("click", closePanel)
  if (newButton) newButton.addEventListener("click", startNewConversation)
  if (retryButton) retryButton.addEventListener("click", submit)
  form.addEventListener("submit", (event) => {
    event.preventDefault()
    submit()
  })
  input.addEventListener("input", () => {
    if (lastErrorCode && input.value !== controller.pendingText()) {
      // Editing is a new question. An explicit unchanged retry still reuses
      // the original key and snapshot; the failed saved question remains.
      controller.clearPending()
      lastErrorCode = null
      setRetryVisible(false)
      setStatus("Изменённый текст будет новым сообщением с новым снимком.")
    }
  })
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  })
  doc.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) closePanel()
  })

  return { controller, openPanel, closePanel, refresh, submit }
}

// Автоинициализация только в браузере. В Node (тесты) модуль загружается без DOM.
if (typeof document !== "undefined" && typeof globalThis.window !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initSupport())
  } else {
    initSupport()
  }
}
