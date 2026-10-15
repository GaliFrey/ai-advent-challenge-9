// Браузерная обвязка тюнера: микрофон, Web Audio API и обновление страницы.
// Вся обработка звука выполняется локально в браузере и не сохраняется и не передаётся.
import { detectPitch, matchString } from "./pitch.js"
import { createReadingState, updateReading, committedAtOf } from "./display.js"
import { createCapture } from "./capture.js"
import { diagnostics } from "./diagnostics.js"

const statusEl = document.getElementById("status")
const startEl = document.getElementById("start")
const directionEl = document.getElementById("direction")
const centsEl = document.getElementById("cents")
const tunedEl = document.getElementById("tuned")
const hintEl = document.getElementById("hint")
const stringEls = new Map(
  Array.from(document.querySelectorAll("[data-note]")).map((el) => [el.dataset.note, el]),
)

const readingState = createReadingState()

function render(reading) {
  for (const el of stringEls.values()) el.classList.remove("active")
  const el = reading.activeNote ? stringEls.get(reading.activeNote) : null
  if (el) el.classList.add("active")

  hintEl.hidden = !reading.hint
  tunedEl.hidden = !reading.tuned

  if (reading.cents === null) {
    directionEl.textContent = ""
    centsEl.textContent = ""
    centsEl.classList.remove("flat", "sharp")
    return
  }

  centsEl.textContent = `${reading.cents > 0 ? "+" : ""}${reading.cents} центов`
  centsEl.classList.toggle("sharp", reading.direction === "Ослабить")
  centsEl.classList.toggle("flat", reading.direction === "Подтянуть")
  directionEl.textContent = reading.direction
}

// Реальные браузерные вызовы остаются здесь; оркестрация стадий и цикл кадров
// вынесены в capture.js, а диагностическое состояние — в diagnostics.js.
const capture = createCapture({
  diagnostics,
  readingState,
  getUserMedia: () => navigator.mediaDevices.getUserMedia({ audio: true }),
  createContext: () => new AudioContext(),
  createSource: (context, stream) => context.createMediaStreamSource(stream),
  createAnalyser: (context) => context.createAnalyser(),
  createBuffer: (size) => new Float32Array(size),
  requestFrame: (callback) => requestAnimationFrame(callback),
  now: () => performance.now(),
  detectPitch,
  matchString,
  updateReading,
  committedAtOf,
  render,
  onRequesting: () => {
    startEl.disabled = true
    statusEl.textContent = "Запрашиваем доступ к микрофону…"
  },
  onListening: () => {
    statusEl.textContent = "Слушаем: дёрните струну."
  },
  onFailure: (error) => {
    startEl.disabled = false
    if (error && error.name === "NotAllowedError") {
      statusEl.textContent = "Доступ к микрофону запрещён. Разрешите его в браузере и попробуйте снова."
    } else {
      statusEl.textContent = "Не удалось получить доступ к микрофону. Откройте страницу через http://localhost и попробуйте снова."
    }
  },
})

startEl.addEventListener("click", () => capture.start())
render(updateReading(readingState, { frequency: null, match: null }, 0))
