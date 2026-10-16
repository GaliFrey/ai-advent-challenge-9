// Панель «Ассистент разработки»: серверный агентный ассистент для frontend.
//
// Модуль не управляет звуком и не импортирует capture.js/display.js. Он только
// общается с API ассистента того же origin, выводит текст и diff исключительно
// через textContent (HTML не исполняется) и после успешного применения
// перезагружает страницу ровно один раз.

export const ASSISTANT_NOTICE =
  "Серверный агент DeepSeek читает и изменяет только разрешённые frontend-файлы этого проекта. " +
  "Содержимое прочитанных файлов передаётся во внешний AI-сервис. Ключ DeepSeek в браузер не попадает. " +
  "Commit, push и установка зависимостей не выполняются."

const CSRF_HEADERS = { "X-Support-Request": "1" }

// Тонкая обёртка над fetch: cookies same-origin, единый разбор ошибок API.
export function createAssistantApi({ fetchImpl = fetch, base = "" } = {}) {
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
        data && data.error ? data.error.message : "Ошибка запроса к ассистенту.",
      )
      error.code = data && data.error ? data.error.code : "request_failed"
      error.status = response.status
      throw error
    }
    return data
  }

  return {
    session: () => request("/api/assistant/session"),
    newConversation: () =>
      request("/api/assistant/conversations", { method: "POST", headers: { ...CSRF_HEADERS } }),
    history: (conversationId) =>
      request(`/api/assistant/conversations/${conversationId}/messages`),
    sendMessage: (conversationId, { text, key }) =>
      request(`/api/assistant/conversations/${conversationId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...CSRF_HEADERS },
        body: JSON.stringify({ text, idempotencyKey: key }),
      }),
  }
}

// Логика переписки без DOM: проверяется Node-тестами с подставным API.
export function createAssistantController({
  api,
  keyFactory = () => globalThis.crypto.randomUUID(),
}) {
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

  async function send(text) {
    if (pending === null) pending = { text, key: keyFactory() }
    const result = await api.sendMessage(conversationId, pending)
    pending = null
    return result
  }

  return {
    load,
    refresh,
    newConversation,
    send,
    applied: (result) =>
      Boolean(
        result &&
          result.assistantMessage &&
          result.assistantMessage.result &&
          result.assistantMessage.result.applied,
      ),
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
export function initAssistant({
  doc = document,
  api = createAssistantApi(),
  keyFactory = () => globalThis.crypto.randomUUID(),
  reload = () => globalThis.location.reload(),
} = {}) {
  const openButton = doc.getElementById("assistant-open")
  const panel = doc.getElementById("assistant-panel")
  const closeButton = doc.getElementById("assistant-close")
  const newButton = doc.getElementById("assistant-new")
  const retryButton = doc.getElementById("assistant-retry")
  const form = doc.getElementById("assistant-form")
  const input = doc.getElementById("assistant-input")
  const sendButton = doc.getElementById("assistant-send")
  const messagesEl = doc.getElementById("assistant-messages")
  const statusEl = doc.getElementById("assistant-status")
  const noticeEl = doc.getElementById("assistant-notice")

  if (!openButton || !panel || !form || !messagesEl) return null

  if (noticeEl) noticeEl.textContent = ASSISTANT_NOTICE

  const controller = createAssistantController({ api, keyFactory })
  let busy = false
  let reloaded = false
  let lastErrorCode = null

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text
  }

  function setRetryVisible(visible) {
    if (retryButton) retryButton.hidden = !visible
  }

  function appendResult(item, result) {
    if (!result) return
    if (result.summary) {
      const summary = doc.createElement("p")
      summary.className = "assistant-summary"
      summary.textContent = result.summary
      item.append(summary)
    }
    if (Array.isArray(result.changedPaths) && result.changedPaths.length) {
      const paths = doc.createElement("p")
      paths.className = "assistant-paths"
      paths.textContent = "Изменённые файлы: " + result.changedPaths.join(", ")
      item.append(paths)
    }
    if (result.diff) {
      const details = doc.createElement("details")
      const summary = doc.createElement("summary")
      summary.textContent = "Diff"
      const pre = doc.createElement("pre")
      pre.textContent = result.diff
      details.append(summary, pre)
      item.append(details)
    }
    if (result.checks && result.checks.command) {
      const details = doc.createElement("details")
      const summary = doc.createElement("summary")
      summary.textContent = `Проверка: ${result.checks.passed ? "успешно" : "ошибка"}`
      const pre = doc.createElement("pre")
      pre.textContent = result.checks.output || ""
      details.append(summary, pre)
      item.append(details)
    }
  }

  // Текст и diff выводятся через textContent: HTML остаётся текстом.
  function renderMessages(messages) {
    messagesEl.replaceChildren()
    for (const message of messages) {
      const item = doc.createElement("li")
      item.className = `assistant-message assistant-message-${message.role}`

      const role = doc.createElement("span")
      role.className = "assistant-role"
      role.textContent = message.role === "user" ? "Вы" : "Ассистент · DeepSeek"

      const text = doc.createElement("p")
      text.className = "assistant-text"
      text.textContent = message.text

      item.append(role, text)
      appendResult(item, message.result)

      if (message.state === "error") {
        const error = doc.createElement("p")
        error.className = "assistant-error"
        error.textContent = message.text || "Операция завершилась ошибкой."
        item.append(error)
        if (message.role === "user" && message.idempotencyKey) {
          const retry = doc.createElement("button")
          retry.type = "button"
          retry.textContent = "Повторить этот запрос"
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
      setStatus("Не удалось загрузить переписку ассистента. Проверьте, что сервер запущен.")
    }
  }

  async function openPanel() {
    panel.hidden = false
    await refresh()
  }

  function closePanel() {
    panel.hidden = true
  }

  function reloadOnce() {
    if (reloaded) return
    reloaded = true
    try {
      globalThis.sessionStorage?.setItem("gt-assistant-reloaded", "1")
    } catch {
      // sessionStorage может быть недоступен; перезагрузка всё равно одна.
    }
    reload()
  }

  function finishSuccess(result) {
    if (controller.applied(result)) {
      setStatus("Изменения применены. Перезагрузка страницы…")
      reloadOnce()
      return true
    }
    return false
  }

  async function retrySaved(message) {
    if (busy) return
    busy = true
    if (sendButton) sendButton.disabled = true
    setStatus("Ассистент работает…")
    try {
      const result = await controller.send(message.text)
      setRetryVisible(false)
      await refresh()
      finishSuccess(result)
    } catch (error) {
      setStatus(error.message || "Не удалось повторить запрос.")
      setRetryVisible(true)
      await refresh()
    } finally {
      busy = false
      if (sendButton) sendButton.disabled = false
    }
  }

  async function submit() {
    if (busy) return
    const text = input.value
    if (!controller.hasPending() && !text.trim()) {
      setStatus("Опишите цель изменения обычным языком.")
      return
    }
    busy = true
    if (sendButton) sendButton.disabled = true
    setRetryVisible(false)
    setStatus("Ассистент работает…")
    try {
      const result = await controller.send(text)
      lastErrorCode = null
      input.value = ""
      await refresh()
      finishSuccess(result)
    } catch (error) {
      lastErrorCode = error.code
      setStatus(error.message || "Не удалось выполнить запрос ассистента.")
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
      setStatus("Создана новая переписка ассистента.")
    } catch {
      setStatus("Не удалось создать новую переписку.")
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
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  })
  input.addEventListener("input", () => {
    if (lastErrorCode && input.value !== controller.pendingText()) {
      controller.clearPending()
      lastErrorCode = null
      setRetryVisible(false)
    }
  })
  doc.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) closePanel()
  })

  return { controller, openPanel, closePanel, refresh, submit, renderMessages }
}

// Автоинициализация только в браузере. В Node (тесты) модуль загружается без DOM.
if (typeof document !== "undefined" && typeof globalThis.window !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initAssistant())
  } else {
    initAssistant()
  }
}
