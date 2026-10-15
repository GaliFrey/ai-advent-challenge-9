import test from "node:test"
import assert from "node:assert/strict"
import { createDiagnostics } from "../src/diagnostics.js"

// Управляемые часы: монотонное значение меняется вручную, календарное фиксировано.
function controllableClock(startMono = 0) {
  const value = { mono: startMono, calendarCalls: 0 }
  return {
    value,
    now: () => value.mono,
    nowCalendar: () => {
      value.calendarCalls += 1
      return new Date(0)
    },
  }
}

test("C1: свежий снимок соответствует idle", () => {
  const d = createDiagnostics({ now: () => 1000, nowCalendar: () => new Date(0) })
  const snap = d.collectSnapshot()

  assert.equal(snap.schemaVersion, 1)
  assert.equal(snap.capture.status, "idle")
  assert.equal(snap.capture.lastError, null)
  assert.deepEqual(snap.audio, {
    trackState: null,
    trackEnabled: null,
    trackMuted: null,
    contextState: null,
  })
  assert.deepEqual(snap.device, { actualLabel: null })
  assert.deepEqual(snap.measurement, {
    rms: null,
    frequencyHz: null,
    matchedString: null,
    ageMs: null,
  })
  assert.deepEqual(snap.display, { string: null, cents: null, direction: null, tuned: false })
  assert.deepEqual(snap.commit, { ageMs: null })
  assert.match(snap.collectedAt, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/)
})

test("C8: снимок сериализуется в JSON версии 1 и независим", () => {
  const clock = controllableClock(1000)
  const d = createDiagnostics({ now: clock.now, nowCalendar: clock.nowCalendar })
  d.beginRequest()
  d.beginInitializing()
  d.beginCapturing({
    track: { readyState: "live", enabled: true, muted: false },
    context: { state: "running" },
  })
  d.recordMeasurement({ rms: 0.083, frequencyHz: 109.49, matchedString: "A2", atMono: 990 })
  d.recordReading({ activeNote: "A2", cents: -8, direction: "Подтянуть", tuned: false }, 655)

  const snap = d.collectSnapshot()
  const text = JSON.stringify(snap)
  assert.ok(!text.includes("Infinity"), "в JSON не должно быть Infinity")
  assert.ok(!text.includes("NaN"), "в JSON не должно быть NaN")

  const parsed = JSON.parse(text)
  assert.equal(parsed.schemaVersion, 1)
  assert.equal(parsed.measurement.ageMs, 10)
  assert.equal(parsed.commit.ageMs, 345)
  assert.deepEqual(parsed.audio, {
    trackState: "live",
    trackEnabled: true,
    trackMuted: false,
    contextState: "running",
  })

  // Мутация возвращённого объекта не меняет внутреннее состояние.
  snap.measurement.ageMs = 999999
  snap.display.string = "E4"
  snap.capture.status = "capturing"
  const again = d.collectSnapshot()
  assert.equal(again.measurement.ageMs, 10)
  assert.equal(again.display.string, "A2")
  assert.match(again.capture.status, /capturing|initializing|requesting|idle|error/)
})

test("C8: каждый снимок — новый объект, время сбора читается один раз", () => {
  const clock = controllableClock(1000)
  const d = createDiagnostics({ now: clock.now, nowCalendar: clock.nowCalendar })
  d.recordMeasurement({ rms: 0.1, frequencyHz: 110, matchedString: "A2", atMono: 990 })
  d.recordReading({ activeNote: "A2", cents: -8, direction: "Подтянуть", tuned: false }, 900)

  const first = d.collectSnapshot()
  clock.value.mono = 1001
  const second = d.collectSnapshot()

  assert.notEqual(first, second)
  assert.notEqual(first.measurement, second.measurement)
  assert.equal(clock.value.calendarCalls, 2)
  assert.equal(first.measurement.ageMs, 10)
  assert.equal(second.measurement.ageMs, 11)
})

test("C3: ошибка нормализуется и очищается при новой попытке", () => {
  const d = createDiagnostics({ now: () => 0, nowCalendar: () => new Date(0) })
  d.beginRequest()
  d.failCapture({ name: "NotAllowedError", message: "secret", stack: "secret" })

  let snap = d.collectSnapshot()
  assert.equal(snap.capture.status, "error")
  assert.equal(snap.capture.lastError, "NotAllowedError")
  assert.ok(!JSON.stringify(snap).includes("secret"), "message/stack не сохраняются")
  assert.equal(snap.measurement.ageMs, null, "измерения не выдумываются")

  d.beginRequest()
  snap = d.collectSnapshot()
  assert.equal(snap.capture.status, "requesting")
  assert.equal(snap.capture.lastError, null)
})

test("C3: неизвестное имя ошибки и не-объект дают other", () => {
  const d = createDiagnostics({ now: () => 0, nowCalendar: () => new Date(0) })
  d.failCapture({ name: "WeirdName" })
  assert.equal(d.collectSnapshot().capture.lastError, "other")
  d.failCapture("boom")
  assert.equal(d.collectSnapshot().capture.lastError, "other")
  d.failCapture(null)
  assert.equal(d.collectSnapshot().capture.lastError, "other")
})

test("C4: измерение без частоты сохраняет RMS и возраст, неизвестное — null", () => {
  const clock = controllableClock(1000)
  const d = createDiagnostics({ now: clock.now, nowCalendar: clock.nowCalendar })
  d.recordMeasurement({ rms: 0, frequencyHz: null, matchedString: null, atMono: 990 })
  const snap = d.collectSnapshot()
  assert.equal(snap.measurement.rms, 0, "нулевой RMS — измеренное значение, а не null")
  assert.equal(snap.measurement.frequencyHz, null)
  assert.equal(snap.measurement.matchedString, null)
  assert.equal(snap.measurement.ageMs, 10)
})

test("C4/C8: NaN, Infinity и -Infinity не попадают в снимок", () => {
  const clock = controllableClock(1000)
  const d = createDiagnostics({ now: clock.now, nowCalendar: clock.nowCalendar })
  d.recordMeasurement({ rms: NaN, frequencyHz: Infinity, matchedString: "A2", atMono: 990 })
  d.recordReading(
    { activeNote: "A2", cents: -Infinity, direction: "Ослабить", tuned: false },
    NaN,
  )
  const snap = d.collectSnapshot()
  assert.equal(snap.measurement.rms, null)
  assert.equal(snap.measurement.frequencyHz, null)
  assert.equal(snap.measurement.matchedString, null)
  assert.equal(snap.display.cents, null)
  assert.equal(snap.display.direction, null)
  assert.equal(snap.commit.ageMs, null)
  const text = JSON.stringify(snap)
  assert.ok(!/Infinity|NaN/.test(text))
})

test("C8: скрытое число при tuned=true не превращается в 0", () => {
  const clock = controllableClock(1000)
  const d = createDiagnostics({ now: clock.now, nowCalendar: clock.nowCalendar })
  d.recordReading({ activeNote: "E2", cents: null, direction: "", tuned: true }, 500)
  const snap = d.collectSnapshot()
  assert.equal(snap.display.string, "E2")
  assert.equal(snap.display.cents, null)
  assert.equal(snap.display.direction, null)
  assert.equal(snap.display.tuned, true)
  assert.equal(snap.commit.ageMs, 500)
})

test("C7: состояние дорожки и контекста читается при сборе из существующих объектов", () => {
  const d = createDiagnostics({ now: () => 0, nowCalendar: () => new Date(0) })
  const track = { readyState: "live", enabled: true, muted: false }
  const context = { state: "running" }
  d.beginCapturing({ track, context })

  let snap = d.collectSnapshot()
  assert.equal(snap.audio.trackState, "live")
  assert.equal(snap.audio.trackEnabled, true)
  assert.equal(snap.audio.trackMuted, false)
  assert.equal(snap.audio.contextState, "running")

  track.readyState = "ended"
  track.enabled = false
  track.muted = true
  context.state = "suspended"
  snap = d.collectSnapshot()
  assert.equal(snap.audio.trackState, "ended")
  assert.equal(snap.audio.trackEnabled, false)
  assert.equal(snap.audio.trackMuted, true)
  assert.equal(snap.audio.contextState, "suspended")
})

test("C7: недоступные свойства не выдумываются", () => {
  const d = createDiagnostics({ now: () => 0, nowCalendar: () => new Date(0) })
  const track = {
    get readyState() {
      throw new Error("nope")
    },
  }
  d.beginCapturing({ track, context: null })
  const snap = d.collectSnapshot()
  assert.equal(snap.audio.trackState, null)
  assert.equal(snap.audio.trackEnabled, null)
  assert.equal(snap.audio.contextState, null)
})

test("C8: до измерения и до показания возрасты null", () => {
  const clock = controllableClock(5000)
  const d = createDiagnostics({ now: clock.now, nowCalendar: clock.nowCalendar })
  d.beginRequest()
  const snap = d.collectSnapshot()
  assert.equal(snap.measurement.ageMs, null)
  assert.equal(snap.commit.ageMs, null)
})
