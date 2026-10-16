import test from "node:test"
import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"

import {
  ASSISTANT_NOTICE,
  createAssistantApi,
  createAssistantController,
  initAssistant,
} from "../src/assistant.js"

const read = (relative) => readFile(fileURLToPath(new URL(relative, import.meta.url)), "utf8")

const source = await read("../src/assistant.js")
const html = await read("../index.html")

function node(tag = "div") {
  return {
    tag,
    children: [],
    textContent: "",
    value: "",
    hidden: true,
    disabled: false,
    handlers: {},
    append(...nodes) {
      this.children.push(...nodes)
    },
    replaceChildren(...nodes) {
      this.children = nodes
    },
    addEventListener(name, handler) {
      this.handlers[name] = handler
    },
  }
}

function makeElements() {
  const names = ["open", "panel", "close", "new", "retry", "form", "input", "send", "messages", "status", "notice"]
  return Object.fromEntries(names.map((name) => [`assistant-${name}`, node()]))
}

test("C1: страница содержит отдельную панель ассистента разработки", () => {
  assert.match(html, /id="assistant-open"/)
  assert.match(html, /id="assistant-panel"/)
  assert.match(html, /id="assistant-form"/)
  assert.match(html, /id="assistant-notice"/)
  assert.match(html, /id="assistant-messages"/)
  assert.match(html, /src="\.\/src\/assistant\.js"/)
  // Панель «Помощь» не сломана.
  assert.match(html, /id="support-panel"/)
})

test("C7: ассистент выводит данные как текст, без innerHTML", () => {
  assert.match(source, /textContent/)
  assert.doesNotMatch(source, /innerHTML/)
  assert.doesNotMatch(source, /insertAdjacentHTML/)
})

test("C7: ключ DeepSeek не упоминается как браузерное значение", () => {
  assert.match(ASSISTANT_NOTICE, /Ключ DeepSeek в браузер не попадает/)
  assert.doesNotMatch(source, /api[_-]?key/i)
})

test("C8/контракт: API ассистента использует служебный заголовок и same-origin cookies", async () => {
  const calls = []
  const api = createAssistantApi({
    fetchImpl: async (url, options) => {
      calls.push([url, options])
      return { ok: true, json: async () => ({ ok: true }) }
    },
    base: "",
  })
  await api.newConversation()
  await api.sendMessage(3, { text: "цель", key: "k" })
  assert.equal(calls[0][0], "/api/assistant/conversations")
  assert.equal(calls[0][1].method, "POST")
  assert.equal(calls[0][1].headers["X-Support-Request"], "1")
  assert.equal(calls[0][1].credentials, "same-origin")
  const body = JSON.parse(calls[1][1].body)
  assert.deepEqual(body, { text: "цель", idempotencyKey: "k" })
})

test("C6: применение в ответе помечается и распознаётся", async () => {
  const api = {
    async session() { return { currentConversationId: 1 } },
    async history() { return { messages: [] } },
    async sendMessage() { return { assistantMessage: { result: { applied: true } } } },
  }
  const controller = createAssistantController({ api, keyFactory: () => "k" })
  await controller.load()
  const result = await controller.send("цель")
  assert.equal(controller.applied(result), true)
  assert.equal(controller.applied({ assistantMessage: { result: { applied: false } } }), false)
  assert.equal(controller.applied(null), false)
})

test("C6: успешное применение вызывает ровно одну перезагрузку", async () => {
  const elements = makeElements()
  const doc = { getElementById: (id) => elements[id], createElement: node, addEventListener() {} }
  let reloads = 0
  const api = {
    async session() { return { currentConversationId: 1 } },
    async history() { return { messages: [] } },
    async sendMessage() { return { assistantMessage: { result: { applied: true } } } },
  }
  const ui = initAssistant({ doc, api, keyFactory: () => "k", reload: () => { reloads += 1 } })
  await ui.refresh()
  elements["assistant-input"].value = "Поправь подписи"
  await ui.submit()
  assert.equal(reloads, 1)
  elements["assistant-input"].value = "Ещё правка"
  await ui.submit()
  assert.equal(reloads, 1, "повторная перезагрузка не запускается")
})

test("C6: без применения перезагрузки нет", async () => {
  const elements = makeElements()
  const doc = { getElementById: (id) => elements[id], createElement: node, addEventListener() {} }
  let reloads = 0
  const api = {
    async session() { return { currentConversationId: 1 } },
    async history() { return { messages: [] } },
    async sendMessage() { return { assistantMessage: { result: { applied: false } } } },
  }
  const ui = initAssistant({ doc, api, keyFactory: () => "k", reload: () => { reloads += 1 } })
  await ui.refresh()
  elements["assistant-input"].value = "Ничего не меняй"
  await ui.submit()
  assert.equal(reloads, 0)
})

test("C6: ошибка сохраняет запрос для явного повтора", async () => {
  const sent = []
  let rejectFirst
  const api = {
    async session() { return { currentConversationId: 1 } },
    async history() { return { messages: [] } },
    async sendMessage(id, payload) {
      sent.push(payload)
      if (sent.length === 1) return new Promise((resolve, reject) => { rejectFirst = reject })
      return { assistantMessage: { result: { applied: false } } }
    },
  }
  const controller = createAssistantController({ api, keyFactory: () => "key-1" })
  await controller.load()
  const pending = controller.send("цель")
  assert.equal(controller.hasPending(), true)
  rejectFirst(new Error("сеть"))
  await assert.rejects(() => pending)
  await controller.send("цель")
  assert.deepEqual(sent[0], sent[1])
  assert.equal(sent[1].key, "key-1")
  assert.equal(controller.hasPending(), false)
})

test("C1/C7: история и итог отображаются безопасно", async () => {
  const elements = makeElements()
  const doc = { getElementById: (id) => elements[id], createElement: node, addEventListener() {} }
  const api = {
    async session() { return { currentConversationId: 1 } },
    async history() {
      return {
        messages: [
          { id: 1, role: "user", text: "<script>alert(1)</script>", state: "completed", result: {} },
          { id: 2, role: "assistant", text: "Готово", state: "completed",
            result: { applied: true, changedPaths: ["web/src/app.js"], diff: "<img onerror=x>", checks: { passed: true, command: "node", output: "ok" } } },
        ],
      }
    },
  }
  const ui = initAssistant({ doc, api, keyFactory: () => "k", reload: () => {} })
  await ui.refresh()
  assert.equal(elements["assistant-notice"].textContent, ASSISTANT_NOTICE)
  const list = elements["assistant-messages"]
  assert.equal(list.children.length, 2)
  const userText = list.children[0].children[1]
  assert.equal(userText.textContent, "<script>alert(1)</script>")
  const assistantResult = list.children[1].children
  assert.ok(assistantResult.some((child) => child.textContent === "Готово"))
  assert.ok(assistantResult.some((child) => child.className === "assistant-paths"))
})
