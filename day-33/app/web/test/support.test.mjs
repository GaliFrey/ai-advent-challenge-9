import test from "node:test"
import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"

import { createSupportController, AI_NOTICE } from "../src/support.js"

const read = (relative) => readFile(fileURLToPath(new URL(relative, import.meta.url)), "utf8")

const source = await read("../src/support.js")
const html = await read("../index.html")

test("C10: модуль панели не управляет звуком и не трогает захват", () => {
  // Проверяем исполняемый код без комментариев: панель не импортирует и не
  // вызывает модули захвата/показаний и Web Audio.
  const code = source
    .replace(/\/\/[^\n]*/g, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
  assert.doesNotMatch(code, /from\s+"\.\/(capture|display)\.js"/)
  assert.doesNotMatch(
    code,
    /getUserMedia|AudioContext|createAnalyser|getFloatTimeDomainData|mediaDevices/,
  )
  // Снимок берётся существующей функцией диагностики.
  assert.match(source, /collectSnapshot/)
  assert.match(source, /from "\.\/diagnostics\.js"/)
})

test("C8: текст сообщений выводится как текст, без исполнения HTML", () => {
  assert.match(source, /textContent/)
  assert.doesNotMatch(source, /innerHTML/)
})

test("C10: панель сообщает о передаче контекста DeepSeek до отправки", () => {
  assert.match(AI_NOTICE, /Ответы формирует DeepSeek/)
  assert.match(AI_NOTICE, /AI может ошибаться/)
  assert.ok(html.includes(AI_NOTICE))
  assert.doesNotMatch(source, /Тестовый режим|тестовый ответ/)
  assert.match(source, /noticeEl\.textContent = AI_NOTICE/)
})

test("C1: страница содержит кнопку и панель помощи", () => {
  assert.match(html, /id="support-open"/)
  assert.match(html, /id="support-panel"/)
  assert.match(html, /id="support-form"/)
  assert.match(html, /id="support-notice"/)
  assert.match(html, /src="\.\/src\/support\.js"/)
})

test("C6: повтор отправки переиспользует тот же снимок и ключ", async () => {
  const sent = []
  const api = {
    async session() {
      return { currentConversationId: null }
    },
    async newConversation() {
      return { conversationId: 1 }
    },
    async history(conversationId) {
      return { conversationId, messages: [] }
    },
    async sendMessage(conversationId, payload) {
      sent.push(payload)
      if (sent.length === 1) throw new Error("network")
      return { ok: true }
    },
  }

  let snapshotCalls = 0
  const controller = createSupportController({
    api,
    snapshotFactory: () => {
      snapshotCalls += 1
      return { schemaVersion: 1 }
    },
    keyFactory: () => "key-42",
  })

  await controller.refresh()
  assert.equal(controller.conversationId, 1)

  await assert.rejects(() => controller.send("привет"))
  assert.equal(controller.hasPending(), true)
  assert.equal(snapshotCalls, 1)

  await controller.send("привет")
  assert.equal(snapshotCalls, 1, "повтор не собирает новый снимок")
  assert.equal(sent.length, 2)
  assert.deepEqual(sent[0], sent[1])
  assert.equal(sent[1].key, "key-42")
  assert.equal(controller.hasPending(), false)
})

test("C4: новый разговор создаётся отдельно от текущего", async () => {
  let nextId = 10
  const api = {
    async session() {
      return { currentConversationId: 5 }
    },
    async newConversation() {
      nextId += 1
      return { conversationId: nextId }
    },
    async history(conversationId) {
      return { conversationId, messages: [] }
    },
    async sendMessage() {
      return { ok: true }
    },
  }
  const controller = createSupportController({
    api,
    snapshotFactory: () => ({}),
    keyFactory: () => "key",
  })
  await controller.refresh()
  assert.equal(controller.conversationId, 5)
  await controller.newConversation()
  assert.equal(controller.conversationId, 11)
})

test("T010: сохранённые источники выводятся безопасно", async () => {
  const { appendSources } = await import("../src/support.js")
  assert.match(AI_NOTICE, /Аудиозапись не отправляется/)
  const doc = {
    createElement(tag) {
      return { tag, children: [], textContent: "", append(...nodes) { this.children.push(...nodes) } }
    },
  }
  const item = doc.createElement("li")
  const sources = [{ id: "hash", title: "Статья", headings: ["Статья", "Раздел"],
    text: "<img src=x onerror=alert(1)>\nСохранённые условия.", similarity: 0.9 }]
  appendSources(doc, item, sources)
  assert.equal(item.children[0].tag, "details")
  const [summary, fragment] = item.children[0].children
  assert.match(summary.textContent, /Источник: Статья \/ Раздел/)
  assert.equal(fragment.textContent, sources[0].text)
  assert.equal(fragment.children.length, 0)
  assert.doesNotMatch(summary.textContent + fragment.textContent, /similarity|0\.9|hash|passage:/)
  const restored = doc.createElement("li")
  appendSources(doc, restored, JSON.parse(JSON.stringify(sources)))
  assert.equal(restored.children[0].children[1].textContent, fragment.textContent)
})

test("T010: ссылки только на сохранённые серверные материалы, прежний формат помечен", async () => {
  const { appendSources } = await import("../src/support.js")
  const doc = { createElement(tag) {
    return { tag, children: [], append(...nodes) { this.children.push(...nodes) } }
  } }
  const item = doc.createElement("li")
  appendSources(doc, item, [
    { title: "<script>", text: "<img onerror=alert(1)>", url: "/api/support/conversations/1/messages/2/sources/abc123" },
    { title: "Bad", text: "text", url: "javascript:alert(1)" },
  ], true)
  assert.match(item.children[0].children[0].textContent, /прежний формат/)
  assert.equal(item.children[0].children[2].tag, "a")
  assert.equal(item.children[0].children[2].href, "/api/support/conversations/1/messages/2/sources/abc123")
  assert.equal(item.children[1].children.length, 2)
})

test("T010: ошибка из сохранённой истории повторяется с исходными данными", async () => {
  const original = { text: "Старый вопрос", snapshot: { schemaVersion: 1 }, idempotencyKey: "old-key" }
  const calls = []
  const api = {
    async session() { return { currentConversationId: 42 } },
    async history() { return { messages: [original] } },
    async sendMessage(id, payload) { calls.push([id, payload]); return {} },
  }
  const controller = createSupportController({ api,
    snapshotFactory: () => assert.fail("Повтор не собирает новый снимок"), keyFactory: () => assert.fail("Новый ключ") })
  await controller.load()
  await controller.retry(original)
  assert.deepEqual(calls, [[42, { text: original.text, snapshot: original.snapshot, key: "old-key" }]])
  assert.equal(controller.hasPending(), false)
})

test("T010: предупреждение, ожидание, безопасная ошибка и явный повтор в DOM", async () => {
  const { initSupport } = await import("../src/support.js")
  function node(tag = "div") {
    return { tag, children: [], textContent: "", value: "", hidden: true, disabled: false, handlers: {},
      append(...nodes) { this.children.push(...nodes) },
      replaceChildren(...nodes) { this.children = nodes },
      addEventListener(name, handler) { this.handlers[name] = handler },
    }
  }
  const names = ["open", "panel", "close", "new", "retry", "form", "input", "send", "messages", "status", "notice"]
  const elements = Object.fromEntries(names.map(n => [`support-${n}`, node()]))
  const doc = { getElementById(id) { return elements[id] }, createElement: node, addEventListener() {} }
  const calls = []
  let rejectFirst
  let snapshotCount = 0
  const api = {
    async session() { return { currentConversationId: 1 } },
    async history() { return { messages: [] } },
    async sendMessage(id, payload) {
      calls.push(payload)
      if (calls.length === 1) return new Promise((resolve, reject) => { rejectFirst = reject })
      return {}
    },
  }
  const ui = initSupport({ doc, api, snapshotFactory: () => { snapshotCount++; return { schemaVersion: 1 } }, keyFactory: () => "key" })
  assert.equal(elements["support-notice"].textContent, AI_NOTICE)
  await ui.refresh()
  elements["support-input"].value = "<script>alert(1)</script>"
  const pending = ui.submit()
  assert.equal(elements["support-send"].disabled, true)
  assert.match(elements["support-status"].textContent, /Ожидаем ответ DeepSeek/)
  await ui.submit() // no duplicate while busy
  assert.equal(calls.length, 1)
  const failure = new Error("DeepSeek временно недоступен.")
  failure.code = "provider_unavailable"
  rejectFirst(failure)
  await pending
  assert.equal(elements["support-send"].disabled, false)
  assert.equal(elements["support-retry"].hidden, false)
  assert.match(elements["support-status"].textContent, /Повторить/)
  await elements["support-retry"].handlers.click()
  assert.deepEqual(calls[1], calls[0])
  assert.equal(snapshotCount, 1)
})
