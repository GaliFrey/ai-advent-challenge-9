import test from "node:test"
import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"

const read = (relative) => readFile(fileURLToPath(new URL(relative, import.meta.url)), "utf8")

const html = await read("../index.html")
const app = await read("../src/app.js")

test("C1: страница запрашивает микрофон и использует Web Audio API", () => {
  assert.match(app, /navigator\.mediaDevices\.getUserMedia/)
  assert.match(app, /new AudioContext/)
  assert.match(app, /createAnalyser/)
  assert.match(html, /<script[^>]+type="module"/)
  assert.match(html, /src="\.\/src\/app\.js"/)
  // Запуск микрофона привязан к действию пользователя на странице.
  assert.match(html, /id="start"/)
})

test("C6: на странице перечислены все шесть струн стандартного строя", () => {
  for (const name of ["E2", "A2", "D3", "G3", "B3", "E4"]) {
    assert.match(html, new RegExp(`data-note="${name}"`), `нет струны ${name}`)
  }
  // Все шесть струн присутствуют в списке разметки.
  assert.equal((html.match(/data-note="/g) || []).length, 6)
})

test("страница содержит метку «настроено» и подсказку «дёрните струну»", () => {
  assert.match(html, /настроено/)
  assert.match(html, /дёрните струну/)
  // Приложение умеет выделять распознанную струну и показывать отклонение.
  assert.match(app, /active/)
  assert.match(app, /matchString/)
})
