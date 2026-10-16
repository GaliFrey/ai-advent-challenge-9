import test from "node:test"
import assert from "node:assert/strict"
import { STRINGS, matchString } from "../src/pitch.js"
import { SETTLE_MS, createReadingState, updateReading } from "../src/display.js"

// Сдвиг частоты на заданное число центов.
function shiftCents(frequency, cents) {
  return frequency * 2 ** (cents / 1200)
}

// Кадр так, как его формирует приложение: частота → matchString.
function frameFor(frequency) {
  return { frequency, match: matchString(frequency) }
}

// Кадр с уровнем сигнала: используется для проверки гейта затухающего хвоста (C13).
function frameWithLevel(frequency, level) {
  return { frequency, match: matchString(frequency), level }
}

const A2 = STRINGS.find((string) => string.name === "A2")
const D3 = STRINGS.find((string) => string.name === "D3")

test("C9: устойчивый тон не дрожит, струна не меняется", () => {
  const state = createReadingState()
  const shown = []
  const notes = []
  let now = 0
  for (let i = 0; i < 200; i += 1) {
    // Намеренно изменчивый по кадрам, но симметричный сигнал. Центр вынесен за
    // диапазон настройки (+20 центов), иначе внутри ±5 числа скрыты (C5).
    const jitter = 20 + (i % 2 === 0 ? 40 : -40)
    const frameTime = now
    const reading = updateReading(state, frameFor(shiftCents(A2.frequency, jitter)), frameTime)
    now += 16
    if (frameTime >= SETTLE_MS + 96) {
      assert.notEqual(reading.cents, null, "вне диапазона настройки показание не должно пропадать")
      shown.push(reading.cents)
      notes.push(reading.activeNote)
    }
  }

  assert.equal(new Set(notes).size, 1, "распознанная струна не должна меняться")
  assert.equal(notes[0], "A2")
  const spread = Math.max(...shown) - Math.min(...shown)
  assert.ok(spread <= 5, `разброс показываемых центов ${spread} превышает 5`)
})

test("C10: показание сохраняется после затухания и снимается устойчивым посторонним тоном", () => {
  const state = createReadingState()
  let now = 0
  let reading = null
  for (let i = 0; i < 40; i += 1) {
    reading = updateReading(state, frameFor(shiftCents(A2.frequency, 10)), now)
    now += 16
  }
  assert.equal(reading.activeNote, "A2")

  // Сколь угодно длинная тишина не сбрасывает показание.
  for (let i = 0; i < 1000; i += 1) {
    reading = updateReading(state, { frequency: null, match: null }, now)
    now += 16
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.cents, 10)
  assert.equal(reading.hint, false)

  // Одиночный сбойный кадр постороннего тона показание не стирает (C10 rev5).
  const pressed = shiftCents(STRINGS[0].frequency, 200)
  assert.equal(matchString(pressed), null, "предпосылка: прижатая нота не сопоставляется")
  reading = updateReading(state, { frequency: pressed, match: matchString(pressed) }, now)
  assert.equal(reading.activeNote, "A2", "одиночный сбойный кадр не должен стирать показание")
  assert.equal(reading.hint, false)

  // Устойчивый посторонний тон (≥~0,25 с) снимает показание.
  for (let i = 0; i < 20; i += 1) {
    now += 16
    reading = updateReading(state, { frequency: pressed, match: matchString(pressed) }, now)
  }
  assert.equal(reading.activeNote, null)
  assert.equal(reading.hint, true)
})

test("C12: показ не мелькает и не перескакивает между далёкими значениями", () => {
  const state = createReadingState()
  const durationMs = 8000
  const step = 16
  const shown = []
  const notes = []
  for (let now = 0; now < durationMs; now += step) {
    // Плавное «дрожание» тона (вибрато) вокруг A2: изменчивый, но не скачущий сигнал.
    const cents = 20 * Math.sin((2 * Math.PI * now) / 2000)
    const reading = updateReading(state, frameFor(shiftCents(A2.frequency, cents)), now)
    if (reading.cents !== null) {
      shown.push(reading.cents)
      notes.push(reading.activeNote)
    }
  }

  assert.equal(new Set(notes).size, 1, "струна не должна меняться, пока звучит та же струна")
  assert.equal(notes[0], "A2")

  let changes = 0
  let maxJump = 0
  for (let i = 1; i < shown.length; i += 1) {
    if (shown[i] !== shown[i - 1]) {
      changes += 1
      maxJump = Math.max(maxJump, Math.abs(shown[i] - shown[i - 1]))
    }
  }
  const durationSec = durationMs / 1000
  assert.ok(
    changes <= Math.floor(3 * durationSec),
    `${changes} изменений за ${durationSec} с — чаще трёх раз в секунду`,
  )
  assert.ok(maxJump <= 100, `скачок показания ${maxJump} центов — слишком далеко`)
})

test("C13: неустойчивое начало щипка не выводится, после стабилизации — точное значение", () => {
  const state = createReadingState()
  const step = 16
  const onsetUntil = SETTLE_MS

  // Первые 0,3 с частота «плывёт» и затухает к целевой A2.
  for (let now = 0; now < onsetUntil; now += step) {
    const drift = 90 * Math.exp(-now / 60)
    const reading = updateReading(
      state,
      frameWithLevel(shiftCents(A2.frequency, drift), 0.2 + 0.8 * (now / onsetUntil)),
      now,
    )
    assert.equal(
      reading.cents,
      null,
      `во время нестабильного начала показание должно оставаться пустым (t=${now})`,
    )
    assert.equal(reading.activeNote, null)
  }

  // После стабилизации тон ровно на целевой частоте: показывается только метка
  // «настроено», числа и направления нет (C5).
  let reading = null
  for (let now = onsetUntil; now < onsetUntil + 1200; now += step) {
    reading = updateReading(state, frameWithLevel(A2.frequency, 1), now)
    assert.equal(reading.cents, null, "внутри допуска число не должно показываться")
    assert.equal(reading.direction, "", "внутри допуска направление не должно показываться")
    assert.equal(reading.tuned, true, "после стабилизации должна появиться метка «настроено»")
    assert.equal(reading.activeNote, "A2")
  }

  // Затухающий хвост: уровень ниже трети максимума, тон «плывёт» — оценка
  // остаётся по устойчивой середине и не меняется (C13).
  for (let now = onsetUntil + 1200; now < onsetUntil + 1800; now += step) {
    const drift = 90 * Math.exp(-(now - onsetUntil - 1200) / 60)
    const tail = updateReading(state, frameWithLevel(shiftCents(A2.frequency, drift), 0.1), now)
    assert.equal(tail.activeNote, "A2")
    assert.equal(tail.cents, null, "затухающий хвост не должен показывать число")
    assert.equal(tail.tuned, true)
  }
})

test("C14: метка «настроено» включается один раз и не мигает у границы ±5 центов", () => {
  const state = createReadingState()
  const shown = []
  let activations = 0
  let wasTuned = false
  for (let i = 0; i < 400; i += 1) {
    // Дрожание вокруг порога: чередование 4 и 6 центов с редкими выбросами 8.
    const jitter = i % 11 === 0 ? 8 : i % 2 === 0 ? 4 : 6
    const reading = updateReading(state, frameFor(shiftCents(A2.frequency, jitter)), i * 16)
    if (!reading.hint) {
      shown.push(reading)
      if (reading.tuned && !wasTuned) activations += 1
      wasTuned = reading.tuned
    }
  }
  assert.ok(shown.length > 0)
  assert.ok(shown.every((reading) => reading.activeNote === "A2"))
  assert.ok(
    shown.some((reading) => reading.tuned === true),
    "для настроенной струны метка должна появиться",
  )
  assert.ok(
    shown.every((reading) => (reading.tuned ? reading.cents === null && reading.direction === "" : true)),
    "при метке «настроено» число и направление не показываются (C5, C11)",
  )
  assert.equal(activations, 1, "метка включается не более одного раза за звучание струны")
})

test("C14: для расстроенной струны метка не показывается", () => {
  const state = createReadingState()
  let reading = null
  for (let i = 0; i < 100; i += 1) {
    reading = updateReading(state, frameFor(shiftCents(A2.frequency, 20)), i * 16)
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.tuned, false)
})

test("C14: противоречивый сигнал показывает «звук не распознан»", () => {
  const state = createReadingState()

  // Сначала устойчиво распознаём A2.
  let now = 0
  for (let i = 0; i < 150; i += 1) {
    updateReading(state, frameFor(A2.frequency), now)
    now += 16
  }

  // Затем звук начинает скакать между двумя струнами.
  const tail = []
  for (let i = 0; i < 30; i += 1) {
    const frequency = i % 2 === 0 ? D3.frequency : A2.frequency
    const reading = updateReading(state, frameFor(frequency), now)
    now += 50
    tail.push(reading)
  }

  for (const reading of tail.slice(-10)) {
    assert.equal(reading.activeNote, null, "противоречивый сигнал не должен показывать струну")
    assert.equal(reading.cents, null)
    assert.equal(reading.hint, true, "должна показываться подсказка")
  }
})

test("C9/C10/C14: реалистичный поток с пропусками определения высоты", () => {
  const state = createReadingState()
  const step = 16
  const frames = 500
  const readings = []
  let activations = 0
  let wasTuned = false
  let hints = 0
  for (let i = 0; i < frames; i += 1) {
    const now = i * step
    // Каждый четвёртый кадр высота не определяется (~25%), но сигнал громкий:
    // это пропуск детектора, а не новый щипок и не тишина (C10, C13).
    const frame =
      i % 4 === 0
        ? { frequency: null, match: null, level: 0.9 }
        : frameWithLevel(A2.frequency, 1)
    const reading = updateReading(state, frame, now)
    if (now >= SETTLE_MS + 96) {
      if (reading.hint) hints += 1
      readings.push(reading)
      if (reading.tuned && !wasTuned) activations += 1
      wasTuned = reading.tuned
    }
  }

  assert.ok(readings.length > 0)
  assert.equal(hints, 0, "пропуски не должны показывать состояние «звук не распознан»")
  assert.ok(
    readings.every((reading) => reading.activeNote === "A2"),
    "распознанная струна не должна меняться",
  )

  // Точная A2 — метка «настроено», числа и направления не видно (C5, C11).
  assert.ok(
    readings.some((reading) => reading.tuned === true),
    "для настроенной струны метка должна появиться",
  )
  assert.ok(
    readings.every(
      (reading) => reading.tuned && reading.cents === null && reading.direction === "",
    ),
    "внутри допуска показывается только метка «настроено»",
  )

  // Если числа появятся (вне допуска), они не должны мелькать или скакать (C9, C12).
  const shown = readings.map((reading) => reading.cents).filter((value) => value !== null)
  const spread = shown.length > 0 ? Math.max(...shown) - Math.min(...shown) : 0
  assert.ok(spread <= 5, `разброс показываемых центов ${spread} превышает 5`)

  let changes = 0
  let maxJump = 0
  for (let i = 1; i < shown.length; i += 1) {
    if (shown[i] !== shown[i - 1]) {
      changes += 1
      maxJump = Math.max(maxJump, Math.abs(shown[i] - shown[i - 1]))
    }
  }
  const durationSec = (frames * step) / 1000
  assert.ok(
    changes <= Math.floor(3 * durationSec),
    `${changes} изменений за ${durationSec} с — чаще трёх раз в секунду`,
  )
  assert.ok(maxJump <= 100, `скачок показания ${maxJump} центов — слишком далеко`)
  assert.equal(activations, 1, "метка включается не более одного раза и не мигает от пропусков")
})

test("C13/C14: низкоуровневый пропуск определения в хвосте не перезапускает оценку и метку", () => {
  const state = createReadingState()
  const step = 16
  let now = 0

  // Устойчивая середина: настроенная струна на +3 цента, метка «настроено»,
  // числа и направления не видно (C5).
  let reading = null
  for (; now < SETTLE_MS + 600; now += step) {
    reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 3), 1), now)
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.cents, null)
  assert.equal(reading.tuned, true)

  // Затухающий хвост с периодическими пропусками определения при низком уровне.
  // Пропуск — не тишина и не новый щипок: показ, струна и метка «настроено»
  // удерживаются до конца звучания (C13, C14).
  const shown = []
  for (let i = 0; i < 200; i += 1, now += step) {
    const frame =
      i % 4 === 0
        ? { frequency: null, match: null, level: 0.1 }
        : frameWithLevel(shiftCents(A2.frequency, 40), 0.1)
    reading = updateReading(state, frame, now)
    assert.equal(reading.activeNote, "A2", "пропуск в хвосте не должен снимать струну")
    assert.equal(reading.cents, null, "пропуск в хвосте не должен показывать число внутри допуска")
    assert.equal(reading.tuned, true, "пропуск в хвосте не должен снимать метку «настроено»")
    shown.push(reading.cents)
  }

  // Показ в хвосте вообще не мелькает — все значения одинаковы (C12, C13).
  const changes = shown.filter((value, index) => index > 0 && value !== shown[index - 1]).length
  assert.equal(changes, 0, "затухающий хвост с пропусками не должен менять показ")
})

test("C15: быстрый щипок любой струны обновляет показание в пределах ~0,5 с", () => {
  const step = 16
  // Первый кадр сразу после окна оценки (SETTLE_MS, шаг кадров 16 мс).
  const afterSettleMs = Math.ceil(SETTLE_MS / step) * step

  for (const target of STRINGS) {
    const state = createReadingState()
    // Прежнее показание — другая открытая струна (любая пара струн отличается
    // больше чем на полутон, поэтому гистерезис не удерживает прежнюю).
    const previous = STRINGS.find((string) => string.name !== target.name)

    // Устойчивое прежнее показание с высоким уровнем сигнала.
    let now = 0
    let reading = null
    for (; now < SETTLE_MS + 600; now += step) {
      reading = updateReading(state, frameWithLevel(shiftCents(previous.frequency, 10), 1), now)
    }
    assert.equal(reading.activeNote, previous.name, `предпосылка: показана ${previous.name}`)

    // Новый щипок целевой струны с быстрым затуханием: пик ~0,55, к концу окна
    // оценки уровень падает до ~0,15 — ниже трети максимума щипка.
    const start = now
    for (let t = start; t < start + SETTLE_MS; t += step) {
      const progress = (t - start) / SETTLE_MS
      const level = 0.55 - (0.55 - 0.15) * progress
      updateReading(state, frameWithLevel(shiftCents(target.frequency, 30), level), t)
    }

    // Первый кадр сразу после окна оценки: несмотря на упавший уровень, на
    // экране должна появиться именно новая струна с её отклонением (C15).
    const showAt = start + afterSettleMs
    assert.ok(
      showAt - start <= 500,
      `новый щипок должен показываться не позднее ~0,5 с (кадр через ${showAt - start} мс)`,
    )
    reading = updateReading(state, frameWithLevel(shiftCents(target.frequency, 30), 0.15), showAt)
    assert.equal(
      reading.activeNote,
      target.name,
      `щипок ${target.name} должен обновить показание, а не оставить ${previous.name}`,
    )
    assert.notEqual(reading.cents, null, "вместе со струной должно быть показано отклонение")
  }
})
