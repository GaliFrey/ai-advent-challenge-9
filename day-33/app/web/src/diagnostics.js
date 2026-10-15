// Диагностический снимок состояния тюнера по контракту версии 1
// (docs/support-context.md, раздел 5.3). Модуль не зависит от DOM и Web Audio:
// он хранит уже вычисленные приложением значения и по запросу собирает из них
// новый независимый JSON-совместимый объект.
//
// Модуль только читает сохранённое состояние: получение снимка не открывает
// микрофон, не запускает повторный анализ аудио и не изменяет состояние
// тюнера. Наружу не отдаются MediaStream, AudioContext, readingState и
// аудиобуферы — только примитивы и null.

import { STRINGS } from "./pitch.js"

const SCHEMA_VERSION = 1

const STRING_NAMES = new Set(STRINGS.map((string) => string.name))
const ERROR_NAMES = new Set([
  "NotAllowedError",
  "NotFoundError",
  "NotReadableError",
  "AbortError",
  "InvalidStateError",
  "SecurityError",
  "TypeError",
  "OverconstrainedError",
])
const TRACK_STATES = new Set(["live", "ended"])
const CONTEXT_STATES = new Set(["suspended", "running", "closed", "interrupted"])
const DIRECTIONS = new Set(["Ослабить", "Подтянуть"])
const STATUSES = new Set(["idle", "requesting", "initializing", "capturing", "error"])

// Нормализует имя ошибки: сохраняется только известное имя из контракта,
// всё остальное (в том числе отсутствие имени) сводится к "other". message,
// stack и произвольные поля ошибки не сохраняются.
function normalizeErrorName(error) {
  if (error && typeof error.name === "string" && ERROR_NAMES.has(error.name)) {
    return error.name
  }
  return "other"
}

// Безопасное чтение свойства существующего объекта: сбой доступа не должен
// ломать сбор снимка, неизвестное значение даёт null.
function safeRead(read) {
  try {
    return read()
  } catch {
    return null
  }
}

// Конечное число в допустимых границах либо null.
function finiteNumber(value, min, max) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  if (min !== undefined && value < min) return null
  if (max !== undefined && value > max) return null
  return value
}

// Возраст по одной шкале монотонных часов: конечное неотрицательное число мс
// либо null, если событие ещё не наступило или время неизвестно.
function ageOrNull(collectedMono, eventMono) {
  if (!Number.isFinite(collectedMono) || !Number.isFinite(eventMono)) return null
  const age = collectedMono - eventMono
  return Number.isFinite(age) && age >= 0 ? age : null
}

// Фабрика хранилища диагностического состояния. Часы внедряются, чтобы сбор
// детерминированно проверялся тестами.
export function createDiagnostics({
  now = () => performance.now(),
  nowCalendar = () => new Date(),
} = {}) {
  const state = {
    status: "idle",
    lastError: null,
    track: null,
    context: null,
    measurement: { rms: null, frequencyHz: null, matchedString: null, atMono: null },
    display: { string: null, cents: null, direction: null, tuned: false },
    committedAt: null,
  }

  // Новая попытка запуска: стадия requesting, прошлая ошибка очищается.
  function beginRequest() {
    state.status = "requesting"
    state.lastError = null
  }

  // Доступ получен, идёт настройка аудиографа.
  function beginInitializing() {
    state.status = "initializing"
  }

  // Граф создан и цикл запланирован. Сохраняются ссылки на фактические объекты,
  // их свойства читаются при сборе; сами объекты наружу не отдаются.
  function beginCapturing({ track = null, context = null } = {}) {
    state.status = "capturing"
    state.track = track || null
    state.context = context || null
  }

  // Перехвачена ошибка запуска или кадра: сохраняется только имя.
  function failCapture(error) {
    state.status = "error"
    state.lastError = normalizeErrorName(error)
  }

  // Последнее завершённое измерение кадра.
  function recordMeasurement({ rms = null, frequencyHz = null, matchedString = null, atMono = null } = {}) {
    state.measurement = {
      rms,
      frequencyHz,
      matchedString,
      atMono: Number.isFinite(atMono) ? atMono : null,
    }
  }

  // Удерживаемое показание интерфейса и время его фактической фиксации.
  // Пустое показание очищает и display, и метку фиксации.
  function recordReading(reading, committedAt) {
    const name = reading && typeof reading.activeNote === "string" ? reading.activeNote : null
    if (name !== null && STRING_NAMES.has(name)) {
      state.display = {
        string: name,
        cents: reading.cents,
        direction: reading.direction,
        tuned: reading.tuned === true,
      }
      state.committedAt = Number.isFinite(committedAt) ? committedAt : null
    } else {
      state.display = { string: null, cents: null, direction: null, tuned: false }
      state.committedAt = null
    }
  }

  // Собирает новый независимый снимок версии 1. Время сбора читается один раз
  // для обоих возрастов; состояние копируется синхронно.
  function collectSnapshot() {
    const collectedMono = now()
    const collectedAt = nowCalendar().toISOString()

    const status = STATUSES.has(state.status) ? state.status : "idle"

    const track = state.track
    const context = state.context
    const rawTrackState = track ? safeRead(() => track.readyState) : null
    const rawTrackEnabled = track ? safeRead(() => track.enabled) : null
    const rawTrackMuted = track ? safeRead(() => track.muted) : null
    const rawContextState = context ? safeRead(() => context.state) : null

    const measurementAge = ageOrNull(collectedMono, state.measurement.atMono)
    let rms = null
    let frequencyHz = null
    let matchedString = null
    if (measurementAge !== null) {
      rms = finiteNumber(state.measurement.rms, 0)
      frequencyHz = finiteNumber(state.measurement.frequencyHz, 65, 450)
      if (frequencyHz !== null && STRING_NAMES.has(state.measurement.matchedString)) {
        matchedString = state.measurement.matchedString
      }
    }

    const displayString = STRING_NAMES.has(state.display.string) ? state.display.string : null
    let displayCents = null
    let displayDirection = null
    let displayTuned = false
    let commitAge = null
    if (displayString !== null) {
      displayTuned = state.display.tuned === true
      if (displayTuned) {
        // Скрытое число при метке «настроено» не превращается в 0.
        displayCents = null
        displayDirection = null
      } else {
        displayCents =
          Number.isInteger(state.display.cents) && Math.abs(state.display.cents) <= 100
            ? state.display.cents
            : null
        displayDirection =
          displayCents !== null && DIRECTIONS.has(state.display.direction)
            ? state.display.direction
            : null
      }
      commitAge = ageOrNull(collectedMono, state.committedAt)
    }

    return {
      schemaVersion: SCHEMA_VERSION,
      collectedAt,
      capture: { status, lastError: state.lastError },
      audio: {
        trackState: TRACK_STATES.has(rawTrackState) ? rawTrackState : null,
        trackEnabled: typeof rawTrackEnabled === "boolean" ? rawTrackEnabled : null,
        trackMuted: typeof rawTrackMuted === "boolean" ? rawTrackMuted : null,
        contextState: CONTEXT_STATES.has(rawContextState) ? rawContextState : null,
      },
      // Для версии 1 метка фактической дорожки не передаётся вовсе.
      device: { actualLabel: null },
      measurement: { rms, frequencyHz, matchedString, ageMs: measurementAge },
      display: {
        string: displayString,
        cents: displayCents,
        direction: displayDirection,
        tuned: displayTuned,
      },
      commit: { ageMs: commitAge },
    }
  }

  return {
    beginRequest,
    beginInitializing,
    beginCapturing,
    failCapture,
    recordMeasurement,
    recordReading,
    collectSnapshot,
  }
}

// Единственный экземпляр приложения. Будущий чат сможет получить снимок через
// `collectSnapshot()` или `diagnostics.collectSnapshot()`.
export const diagnostics = createDiagnostics()

export function collectSnapshot() {
  return diagnostics.collectSnapshot()
}
