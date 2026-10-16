import test from "node:test"
import assert from "node:assert/strict"
import { createCapture } from "../src/capture.js"
import { createDiagnostics } from "../src/diagnostics.js"
import { createReadingState, updateReading, committedAtOf } from "../src/display.js"
import { detectPitch, matchString } from "../src/pitch.js"

// Синтетический тон: чистый синус заданной частоты.
function tone(frequency, sampleRate, length) {
  const samples = new Float32Array(length)
  const step = (2 * Math.PI * frequency) / sampleRate
  for (let i = 0; i < length; i += 1) samples[i] = Math.sin(step * i)
  return samples
}

function shiftCents(frequency, cents) {
  return frequency * 2 ** (cents / 1200)
}

// Обвязка захвата с подставными браузерными объектами, управляемыми часами и
// реальными detectPitch/matchString/updateReading.
function createHarness(options = {}) {
  let mono = 0
  let samples = new Float32Array(4096)
  let scheduled = null
  let lastReading = null

  const counters = {
    getUserMedia: 0,
    float: 0,
    context: 0,
    analyser: 0,
    render: 0,
    requesting: 0,
    listening: 0,
    failure: 0,
  }
  const failures = []

  const now = () => mono
  const nowCalendar = () => new Date(0)
  const diagnostics = createDiagnostics({ now, nowCalendar })
  const readingState = createReadingState()

  const track = { readyState: "live", enabled: true, muted: false }
  const context = { state: "running", sampleRate: 48000 }
  const stream = { getAudioTracks: () => [track] }

  const getUserMedia = () => {
    counters.getUserMedia += 1
    return options.getUserMedia ? options.getUserMedia() : Promise.resolve(stream)
  }

  const analyser = {
    fftSize: 2048,
    getFloatTimeDomainData(target) {
      counters.float += 1
      target.set(samples.subarray(0, target.length))
    },
  }

  const capture = createCapture({
    diagnostics,
    readingState,
    getUserMedia,
    createContext: () => {
      counters.context += 1
      return context
    },
    createSource: () => ({ connect() {} }),
    createAnalyser: () => {
      counters.analyser += 1
      return analyser
    },
    createBuffer: (size) => new Float32Array(size),
    requestFrame: (callback) => {
      scheduled = callback
    },
    now,
    detectPitch,
    matchString,
    updateReading,
    committedAtOf,
    render: (reading) => {
      counters.render += 1
      lastReading = reading
    },
    onRequesting: () => {
      counters.requesting += 1
    },
    onListening: () => {
      counters.listening += 1
    },
    onFailure: (error) => {
      counters.failure += 1
      failures.push(error)
    },
  })

  return {
    capture,
    diagnostics,
    readingState,
    counters,
    failures,
    track,
    context,
    stream,
    analyser,
    setMono: (value) => {
      mono = value
    },
    setSamples: (value) => {
      samples = value
    },
    runFrame: () => {
      const callback = scheduled
      scheduled = null
      callback(mono)
    },
    hasScheduledFrame: () => scheduled !== null,
    get lastReading() {
      return lastReading
    },
    snapshot: () => diagnostics.collectSnapshot(),
  }
}

test("C2: requesting не выдаётся за захват, после настройки — capturing", async () => {
  let resolveMedia
  const media = new Promise((resolve) => {
    resolveMedia = resolve
  })
  const h = createHarness({ getUserMedia: () => media })

  const started = h.capture.start()
  assert.equal(h.snapshot().capture.status, "requesting")
  assert.notEqual(h.snapshot().capture.status, "capturing")
  assert.equal(h.counters.requesting, 1)

  resolveMedia(h.stream)
  await started

  const snap = h.snapshot()
  assert.equal(snap.capture.status, "capturing")
  assert.equal(snap.capture.lastError, null)
  assert.equal(snap.audio.trackState, "live")
  assert.equal(snap.audio.contextState, "running")
  assert.equal(h.counters.listening, 1)
})

test("C3: отказ доступа нормализуется и очищается при повторной попытке", async () => {
  let attempt = 0
  const h = createHarness({
    getUserMedia: () => {
      attempt += 1
      if (attempt === 1) {
        const error = new Error("internal detail")
        error.name = "NotAllowedError"
        return Promise.reject(error)
      }
      return Promise.resolve(h.stream)
    },
  })

  await h.capture.start()
  const failed = h.snapshot()
  assert.equal(failed.capture.status, "error")
  assert.equal(failed.capture.lastError, "NotAllowedError")
  assert.ok(!JSON.stringify(failed).includes("internal detail"))
  assert.equal(failed.measurement.rms, null)
  assert.equal(failed.measurement.ageMs, null)
  assert.equal(failed.display.string, null)
  assert.equal(failed.commit.ageMs, null)

  await h.capture.start()
  const retried = h.snapshot()
  assert.equal(retried.capture.lastError, null)
  assert.equal(retried.capture.status, "capturing")
})

test("C3: неизвестное имя ошибки нормализуется в other", async () => {
  const h = createHarness({
    getUserMedia: () => {
      const error = new Error("x")
      error.name = "SomethingElse"
      return Promise.reject(error)
    },
  })
  await h.capture.start()
  assert.equal(h.snapshot().capture.lastError, "other")
})

test("C3: ошибка кадра регистрируется, исключение не маскируется, цикл не продолжается", async () => {
  const h = createHarness()
  await h.capture.start()

  h.analyser.getFloatTimeDomainData = () => {
    const error = new Error("frame boom")
    error.name = "InvalidStateError"
    throw error
  }

  assert.throws(() => h.runFrame(), /frame boom/)
  const snap = h.snapshot()
  assert.equal(snap.capture.status, "error")
  assert.equal(snap.capture.lastError, "InvalidStateError")
  assert.equal(h.hasScheduledFrame(), false, "следующий кадр не перепланируется")
  assert.equal(h.counters.failure, 1)
})

test("C4: кадр сохраняет реальные результаты и возраст", async () => {
  const h = createHarness()
  await h.capture.start()

  h.setMono(500)
  h.setSamples(tone(110, 48000, 4096))
  await h.runFrame()

  const snap = h.snapshot()
  assert.ok(Number.isFinite(snap.measurement.rms) && snap.measurement.rms > 0)
  assert.ok(
    snap.measurement.frequencyHz >= 65 && snap.measurement.frequencyHz <= 450,
    "частота в допустимом диапазоне",
  )
  assert.equal(snap.measurement.matchedString, "A2")
  assert.equal(snap.measurement.ageMs, 0)
})

test("C4: без найденной частоты RMS и возраст всё равно известны", async () => {
  const h = createHarness()
  await h.capture.start()

  h.setMono(500)
  h.setSamples(new Float32Array(4096))
  await h.runFrame()

  const snap = h.snapshot()
  assert.equal(snap.measurement.rms, 0, "RMS измерен и равен нулю, а не null")
  assert.equal(snap.measurement.frequencyHz, null)
  assert.equal(snap.measurement.matchedString, null)
  assert.equal(snap.measurement.ageMs, 0)
})

test("C5: свежий кадр не обновляет возраст старого commit", async () => {
  const h = createHarness()
  await h.capture.start()

  const a2 = tone(110, 48000, 4096)
  const silence = new Float32Array(4096)
  for (let i = 0; i < 40; i += 1) {
    h.setMono(i * 16)
    h.setSamples(a2)
    h.runFrame()
  }

  const stable = h.snapshot()
  assert.equal(stable.display.string, "A2")
  assert.ok(Number.isFinite(stable.commit.ageMs))
  const commitBefore = stable.commit.ageMs

  for (let i = 40; i < 80; i += 1) {
    h.setMono(i * 16)
    h.setSamples(silence)
    h.runFrame()
  }

  const held = h.snapshot()
  assert.equal(held.display.string, "A2", "показание удерживается после затухания")
  assert.equal(held.measurement.rms, 0, "последнее измерение свежее")
  assert.ok(held.measurement.ageMs <= 1, "возраст измерения мал")
  assert.ok(held.commit.ageMs > commitBefore, "возраст старого commit растёт, но метка не меняется")
})

test("C6: сброс показания очищает display и commit.ageMs", async () => {
  const h = createHarness()
  await h.capture.start()

  const a2 = tone(110, 48000, 4096)
  for (let i = 0; i < 40; i += 1) {
    h.setMono(i * 16)
    h.setSamples(a2)
    h.runFrame()
  }
  assert.equal(h.snapshot().display.string, "A2")

  // Посторонний тон, не совпадающий с открытой струной: снимает показ.
  const foreign = tone(shiftCents(82.41, 200), 48000, 4096)
  for (let i = 40; i < 80; i += 1) {
    h.setMono(i * 16)
    h.setSamples(foreign)
    h.runFrame()
  }

  const cleared = h.snapshot()
  assert.equal(cleared.display.string, null)
  assert.equal(cleared.display.cents, null)
  assert.equal(cleared.display.tuned, false)
  assert.equal(cleared.commit.ageMs, null)
})

test("C7: изменение состояния дорожки/контекста отражается в снимке", async () => {
  const h = createHarness()
  await h.capture.start()

  assert.equal(h.snapshot().audio.trackState, "live")
  h.track.readyState = "ended"
  h.track.enabled = false
  h.track.muted = true
  h.context.state = "suspended"

  const snap = h.snapshot()
  assert.equal(snap.audio.trackState, "ended")
  assert.equal(snap.audio.trackEnabled, false)
  assert.equal(snap.audio.trackMuted, true)
  assert.equal(snap.audio.contextState, "suspended")
})

test("C9: сбор снимка не вызывает побочных действий и не меняет результат", async () => {
  const h = createHarness()
  await h.capture.start()

  const a2 = tone(110, 48000, 4096)
  for (let i = 0; i < 40; i += 1) {
    h.setMono(i * 16)
    h.setSamples(a2)
    h.runFrame()
  }

  const readingBefore = h.lastReading
  const countsBefore = { ...h.counters }
  const first = h.snapshot()
  const second = h.snapshot()

  assert.deepEqual(h.counters, countsBefore, "сбор не вызывает getUserMedia и анализ")
  assert.equal(h.lastReading, readingBefore, "результат тюнера не меняется")
  assert.deepEqual(first, second)
  assert.equal(first.schemaVersion, 1)
})

test("C10: успешный запуск сохраняет прежнее поведение и guard от повторного", async () => {
  const h = createHarness()
  await h.capture.start()

  assert.equal(h.counters.requesting, 1)
  assert.equal(h.counters.listening, 1)
  assert.equal(h.counters.failure, 0)
  assert.equal(h.snapshot().capture.status, "capturing")
  assert.equal(h.hasScheduledFrame(), true)

  await h.capture.start()
  assert.equal(h.counters.getUserMedia, 1, "повторное нажатие игнорируется")
})
