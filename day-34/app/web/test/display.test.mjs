import test from "node:test"
import assert from "node:assert/strict"
import { STRINGS, matchString } from "../src/pitch.js"
import { SETTLE_MS, committedAtOf, createReadingState, updateReading } from "../src/display.js"

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

// Непрерывный поток одинаковой частоты; возвращает последнее показание.
function feedStable(state, frequency, frames, step = 16, start = 0) {
  let reading = null
  let now = start
  for (let i = 0; i < frames; i += 1) {
    reading = updateReading(state, frameFor(frequency), now)
    now += step
  }
  return reading
}

test("C9: показываемое отклонение стабильно на непрерывном потоке кадров", () => {
  const state = createReadingState()
  const shown = []
  const notes = []
  const raw = []
  let now = 0
  for (let i = 0; i < 160; i += 1) {
    // Устойчивый тон с детерминированным дрожанием ±6 центов вокруг +20:
    // центр вынесен за диапазон настройки, иначе внутри ±5 числа скрыты (C5).
    const jitter = 20 + (i % 2 === 0 ? 6 : -6)
    const frameTime = now
    const reading = updateReading(state, frameFor(shiftCents(A2.frequency, jitter)), frameTime)
    now += 16
    raw.push(jitter)
    // Измеряем после окна стабилизации начала щипка (C13).
    if (frameTime >= SETTLE_MS + 96) {
      shown.push(reading.cents)
      notes.push(reading.activeNote)
    }
  }

  // Исходное дрожание заметно, а показываемое отклонение — нет.
  assert.ok(Math.max(...raw) - Math.min(...raw) > 5, "исходный поток должен дрожать")
  assert.ok(
    shown.every((value) => value !== null),
    "вне диапазона настройки показание не должно пропадать",
  )
  assert.equal(new Set(notes).size, 1, "распознанная струна не должна меняться")
  assert.equal(notes[0], "A2")
  const spread = Math.max(...shown) - Math.min(...shown)
  assert.ok(spread <= 5, `разброс показываемых центов ${spread} превышает 5`)
})

test("C10: последнее показание удерживается до следующего уверенного звука", () => {
  const state = createReadingState()
  let now = 0
  let reading = null
  for (let i = 0; i < 40; i += 1) {
    reading = updateReading(state, frameFor(shiftCents(A2.frequency, 10)), now)
    now += 16
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.hint, false)
  assert.equal(reading.direction, "Ослабить")
  assert.equal(reading.cents, 10)

  // Долгая тишина: показание остаётся на экране, таймер его не сбрасывает (C10).
  for (let i = 0; i < 400; i += 1) {
    reading = updateReading(state, { frequency: null, match: null }, now)
    now += 16
  }
  assert.equal(reading.activeNote, "A2", "тишина не должна сбрасывать показание")
  assert.equal(reading.hint, false)
  assert.equal(reading.direction, "Ослабить")
  assert.equal(reading.cents, 10)

  // Одиночный сбойный кадр постороннего тона показание не стирает (C10 rev5).
  const pressed = shiftCents(STRINGS[0].frequency, 200)
  assert.equal(matchString(pressed), null, "предпосылка: прижатая нота не сопоставляется")
  reading = updateReading(state, { frequency: pressed, match: matchString(pressed) }, now)
  assert.equal(reading.activeNote, "A2", "одиночный сбойный кадр не должен стирать показание")
  assert.equal(reading.cents, 10)
  assert.equal(reading.hint, false)

  // Устойчивый посторонний тон суммарно ≥~0,25 с снимает показание (C8/C10).
  for (let i = 0; i < 20; i += 1) {
    now += 16
    reading = updateReading(state, { frequency: pressed, match: matchString(pressed) }, now)
  }
  assert.equal(reading.activeNote, null)
  assert.equal(reading.cents, null)
  assert.equal(reading.hint, true)
})

test("C8: посторонний тон снимает показание, только когда держится", () => {
  const state = createReadingState()
  feedStable(state, shiftCents(A2.frequency, 10), 40)

  // Прижатая нота F#2 — на 200 центов выше E2, это не открытая струна.
  const pressed = shiftCents(STRINGS[0].frequency, 200)
  assert.equal(matchString(pressed), null, "предпосылка: прижатая нота не сопоставляется")

  // Одиночный кадр сохраняет прошлое показание (C10).
  let reading = updateReading(state, { frequency: pressed, match: matchString(pressed) }, 2000)
  assert.equal(reading.activeNote, "A2", "одиночный кадр не должен стирать показание")
  assert.equal(reading.hint, false)

  // Устойчивый посторонний тон ≥~0,25 с снимает показание и показывает подсказку.
  for (let i = 1; i <= 20; i += 1) {
    reading = updateReading(state, { frequency: pressed, match: matchString(pressed) }, 2000 + i * 16)
  }
  assert.equal(reading.activeNote, null, "устойчивый посторонний тон снимает показание")
  assert.equal(reading.cents, null)
  assert.equal(reading.hint, true)
})

test("C11: надписи согласованы при округлении отклонения", () => {
  // Отклонение, округляемое до нуля, показывается только меткой «настроено»:
  // ни числа, ни подсказки направления нет (C5).
  const zero = feedStable(createReadingState(), shiftCents(A2.frequency, 0.4), 40)
  assert.equal(zero.cents, null)
  assert.equal(zero.direction, "")
  assert.equal(zero.tuned, true)

  // Заметное отклонение: «Ослабить» (выше цели) и метка снята.
  const sharp = feedStable(createReadingState(), shiftCents(A2.frequency, 6), 40)
  assert.equal(sharp.cents, 6)
  assert.equal(sharp.direction, "Ослабить")
  assert.equal(sharp.tuned, false)

  // «Подтянуть» — звук ниже цели.
  const flat = feedStable(createReadingState(), shiftCents(A2.frequency, -6), 40)
  assert.equal(flat.cents, -6)
  assert.equal(flat.direction, "Подтянуть")
  assert.equal(flat.tuned, false)

  // Метка «настроено» согласована с отображаемым отклонением в пределах ±5.
  const edge = feedStable(createReadingState(), shiftCents(A2.frequency, 5), 40)
  assert.equal(edge.cents, null)
  assert.equal(edge.direction, "")
  assert.equal(edge.tuned, true)

  // Инвариант непротиворечивости (C11): пока показана метка «настроено», число
  // и направление скрыты; вне диапазона направление всегда вместе с величиной.
  for (const reading of [zero, edge, sharp, flat]) {
    if (reading.tuned) {
      assert.equal(reading.cents, null)
      assert.equal(reading.direction, "")
    } else {
      assert.ok(reading.cents !== null)
      assert.ok(reading.direction === "Ослабить" || reading.direction === "Подтянуть")
    }
  }
})

test("C4: показывает подсказку «Ослабить»/«Подтянуть» и величину отклонения", () => {
  const sharp = feedStable(createReadingState(), shiftCents(A2.frequency, 30), 40)
  assert.equal(sharp.activeNote, "A2")
  assert.equal(sharp.direction, "Ослабить")
  assert.equal(sharp.cents, 30)

  const flat = feedStable(createReadingState(), shiftCents(A2.frequency, -30), 40)
  assert.equal(flat.activeNote, "A2")
  assert.equal(flat.direction, "Подтянуть")
  assert.equal(flat.cents, -30)
})

test("C5: метка «настроено» только при отклонении не более 5 центов", () => {
  for (const cents of [0, 4.9, 5]) {
    const reading = feedStable(createReadingState(), shiftCents(A2.frequency, cents), 40)
    assert.equal(reading.tuned, true)
    // Внутри допуска видно только «настроено»: ни числа, ни направления (C11).
    assert.equal(reading.cents, null, `при ${cents} центах число не должно показываться`)
    assert.equal(reading.direction, "", `при ${cents} центах направление не должно показываться`)
  }
  const off = feedStable(createReadingState(), shiftCents(A2.frequency, 6), 40)
  assert.equal(off.tuned, false)
  assert.equal(off.cents, 6)
})

test("C13: затухающий хвост не меняет показ и метку «настроено»", () => {
  const state = createReadingState()
  const step = 16
  let now = 0

  // Начало щипка: тихо и неустойчиво — показ ещё пуст.
  for (; now < SETTLE_MS; now += step) {
    const reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 3), 0.05), now)
    assert.equal(reading.cents, null)
  }

  // Устойчивая середина: громко и ровно у цели — появляется метка «настроено»,
  // число и направление при этом скрыты (C5).
  let reading = null
  for (; now < SETTLE_MS + 600; now += step) {
    reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 3), 1), now)
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.cents, null)
  assert.equal(reading.tuned, true)

  // Затухающий хвост: уровень упал ниже трети максимума, а тон «уполз».
  // Показ и метка должны остаться прежними (оценка по устойчивой середине).
  for (let i = 0; i < 60; i += 1, now += step) {
    reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 40), 0.1), now)
    assert.equal(reading.activeNote, "A2")
    assert.equal(reading.cents, null, "затухающий хвост не должен показывать число внутри допуска")
    assert.equal(reading.tuned, true, "затухающий хвост не должен снимать метку «настроено»")
  }
})

test("C13: новый щипок после затухания снова обновляет показ", () => {
  const state = createReadingState()
  const step = 16
  let now = 0

  // Первый щипок: громкая устойчивая середина на +10 центов, затем затухание.
  for (; now < SETTLE_MS + 600; now += step) {
    updateReading(state, frameWithLevel(shiftCents(A2.frequency, 10), 1), now)
  }
  for (; now < SETTLE_MS + 1200; now += step) {
    updateReading(state, frameWithLevel(shiftCents(A2.frequency, 10), 0.1), now)
  }

  // Второй громкий щипок той же струны, но заметно ниже цели (-30 центов):
  // это новый щипок, а не продолжение хвоста — показ должен обновиться.
  let reading = null
  for (; now < SETTLE_MS + 2400; now += step) {
    reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, -30), 1), now)
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.cents, -30)
  assert.equal(reading.direction, "Подтянуть")
})

test("C13: пропуски определения при высоком уровне не сбрасывают оценку", () => {
  const state = createReadingState()
  const step = 16
  let now = 0

  // Устойчивая середина: громкий ровный тон на +3 цента, метка «настроено».
  let reading = null
  for (; now < SETTLE_MS + 600; now += step) {
    reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 3), 1), now)
  }
  assert.equal(reading.activeNote, "A2")
  assert.equal(reading.cents, null)
  assert.equal(reading.tuned, true)

  // Каждый четвёртый кадр высота не определяется, но уровень остаётся высоким.
  // Это пропуск детектора, а не новый щипок и не тишина: показ, окно оценки и
  // метка «настроено» должны сохраняться (C10, C13, C14).
  for (let i = 0; i < 120; i += 1, now += step) {
    const frame =
      i % 4 === 0
        ? { frequency: null, match: null, level: 0.9 }
        : frameWithLevel(shiftCents(A2.frequency, 3), 1)
    reading = updateReading(state, frame, now)
    assert.equal(reading.activeNote, "A2", "пропуск не должен снимать распознанную струну")
    assert.equal(reading.cents, null, "пропуск не должен показывать число внутри допуска")
    assert.equal(reading.tuned, true, "пропуск не должен снимать метку «настроено»")
    assert.equal(reading.hint, false)
  }

  // Максимум уровня не сброшен пропусками, поэтому гейт затухающего хвоста
  // по-прежнему работает и не даёт показанию «уползти» (C13).
  for (let i = 0; i < 60; i += 1, now += step) {
    reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 40), 0.1), now)
    assert.equal(reading.cents, null, "затухающий хвост после пропусков не должен менять показ")
    assert.equal(reading.tuned, true, "затухающий хвост после пропусков не должен снимать метку")
  }
})

test("C13: одиночный пропуск в хвосте не перезапускает оценку, тишина и рост — перезапускают", () => {
  const step = 16

  // Подготовка: громкая устойчивая середина на +3 цента (метка «настроено»).
  function pluckedState() {
    const state = createReadingState()
    let now = 0
    for (; now < SETTLE_MS + 600; now += step) {
      updateReading(state, frameWithLevel(shiftCents(A2.frequency, 3), 1), now)
    }
    return { state, now }
  }

  // Одиночный кадр без высоты при любом уровне не считается ни тишиной, ни новым
  // щипком (C10, C13): даже слабый «уплывший» хвост сохраняет оценку середины.
  for (const skipLevel of [0.9, 0.1]) {
    const { state, now } = pluckedState()
    updateReading(state, { frequency: null, match: null, level: skipLevel }, now)

    let reading = null
    for (let i = 1; i <= 40; i += 1) {
      reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 40), 0.1), now + step * i)
    }
    assert.equal(
      reading.cents,
      null,
      `пропуск (level=${skipLevel}) не должен перезапускать оценку хвоста`,
    )
    assert.equal(reading.tuned, true, "пропуск в хвосте не должен снимать метку «настроено»")
  }

  // Заметный возврат громкости в хвосте (~в полтора раза) — новый щипок: показ
  // пересчитывается по новому звуку (C13).
  {
    const { state, now } = pluckedState()
    let reading = null
    for (let i = 0; i < 30; i += 1) {
      reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, 3), 0.1), now + step * i)
    }
    assert.equal(reading.cents, null, "предпосылка: затухающий хвост держит оценку середины")

    let t = now + step * 30
    for (let i = 0; i < SETTLE_MS / step + 40; i += 1, t += step) {
      reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, -30), 0.2), t)
    }
    assert.equal(reading.cents, -30, "возврат громкости ~в 1,5 раза — новый щипок")
    assert.equal(reading.direction, "Подтянуть")
  }

  // Длительная тишина, затем новый звук — новый щипок: показ пересчитывается
  // по новому звуку (C13).
  {
    const { state, now } = pluckedState()
    let t = now
    for (let i = 0; i < 40; i += 1, t += step) {
      updateReading(state, { frequency: null, match: null, level: 0 }, t)
    }

    let reading = null
    for (let i = 0; i < SETTLE_MS / step + 40; i += 1, t += step) {
      reading = updateReading(state, frameWithLevel(shiftCents(A2.frequency, -30), 1), t)
    }
    assert.equal(reading.cents, -30, "звук после тишины — новый щипок")
    assert.equal(reading.direction, "Подтянуть")
  }
})

test("C6: метка фиксации обновляется повторным commit и очищается при сбросе", () => {
  const state = createReadingState()
  assert.equal(committedAtOf(state), null, "до фиксации метки нет")

  let now = 0
  // Первый уверенный щипок A2 с отклонением +10 центов.
  for (let i = 0; i < 60; i += 1, now += 16) {
    updateReading(state, frameWithLevel(shiftCents(A2.frequency, 10), 1), now)
  }
  const first = committedAtOf(state)
  assert.ok(Number.isFinite(first), "после commit должна быть метка фиксации")

  // Подтверждённая тишина не сбрасывает показ и не обновляет метку.
  for (let i = 0; i < 30; i += 1, now += 16) {
    updateReading(state, { frequency: null, match: null, level: 0 }, now)
  }
  assert.equal(committedAtOf(state), first, "удержание показания не обновляет метку")

  // Новый щипок той же струны с тем же показанием — повторный commit.
  for (let i = 0; i < 60; i += 1, now += 16) {
    updateReading(state, frameWithLevel(shiftCents(A2.frequency, 10), 1), now)
  }
  const second = committedAtOf(state)
  assert.ok(second > first, "повторный commit с тем же показанием обновляет метку")

  // Устойчивый посторонний тон снимает показ и очищает метку фиксации.
  const pressed = shiftCents(STRINGS[0].frequency, 200)
  assert.equal(matchString(pressed), null, "предпосылка: прижатая нота не сопоставляется")
  for (let i = 0; i < 30; i += 1, now += 16) {
    updateReading(state, { frequency: pressed, match: matchString(pressed) }, now)
  }
  assert.equal(committedAtOf(state), null, "снятие показания очищает метку фиксации")
})
