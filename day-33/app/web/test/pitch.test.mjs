import test from "node:test"
import assert from "node:assert/strict"
import {
  STRINGS,
  TUNED_CENTS,
  RECOGNIZE_CENTS,
  centsBetween,
  detectPitch,
  matchString,
} from "../src/pitch.js"

// Синтетический тон: чистый синус заданной частоты.
function tone(frequency, sampleRate, length) {
  const samples = new Float32Array(length)
  const step = (2 * Math.PI * frequency) / sampleRate
  for (let i = 0; i < length; i += 1) samples[i] = Math.sin(step * i)
  return samples
}

// Тон с гармониками (основной тон + обертоны), ближе к реальной струне.
function harmonicTone(frequency, sampleRate, length) {
  const samples = new Float32Array(length)
  const step = (2 * Math.PI * frequency) / sampleRate
  for (let i = 0; i < length; i += 1) {
    samples[i] =
      Math.sin(step * i) +
      0.5 * Math.sin(2 * step * i) +
      0.33 * Math.sin(3 * step * i) +
      0.25 * Math.sin(4 * step * i)
  }
  return samples
}

// Шаг в центах от целевой частоты.
function shiftCents(frequency, cents) {
  return frequency * 2 ** (cents / 1200)
}

// Детерминированный псевдослучайный шум (xorshift32).
function noise(length, seed = 2463534242) {
  const samples = new Float32Array(length)
  let state = seed >>> 0
  for (let i = 0; i < length; i += 1) {
    state ^= state << 13
    state >>>= 0
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    samples[i] = (state / 4294967295) * 2 - 1
  }
  return samples
}

test("C2: определяет частоту основного тона открытых струн", () => {
  for (const sampleRate of [44100, 48000]) {
    for (const string of STRINGS) {
      for (const signal of [
        tone(string.frequency, sampleRate, 4096),
        harmonicTone(string.frequency, sampleRate, 4096),
      ]) {
        const detected = detectPitch(signal, sampleRate)
        assert.ok(Number.isFinite(detected), `${string.name}@${sampleRate}: нет частоты`)
        const cents = centsBetween(detected, string.frequency)
        assert.ok(
          Math.abs(cents) < 25,
          `${string.name}@${sampleRate}: ${detected.toFixed(2)} Гц, ${cents.toFixed(2)} центов`,
        )
      }
    }
  }
})

test("C3: автоматически сопоставляет звук с одной из шести струн", () => {
  for (const string of STRINGS) {
    const result = matchString(string.frequency)
    assert.ok(result, `нет сопоставления для ${string.name}`)
    assert.equal(result.string.name, string.name)
    assert.equal(result.index, string.index)
  }

  // Полный конвейер: распознавание тона и выбор струны без ручного ввода.
  const detected = detectPitch(tone(110.0, 48000, 4096), 48000)
  const match = matchString(detected)
  assert.ok(match)
  assert.equal(match.string.name, "A2")
})

test("C4: показывает направление и величину отклонения в центах", () => {
  const sharp = matchString(shiftCents(110.0, 50))
  assert.equal(sharp.string.name, "A2")
  assert.equal(sharp.direction, "high")
  assert.ok(sharp.cents > 49 && sharp.cents < 51, `выше: ${sharp.cents} центов`)

  const flat = matchString(shiftCents(110.0, -50))
  assert.equal(flat.string.name, "A2")
  assert.equal(flat.direction, "low")
  assert.ok(flat.cents < -49 && flat.cents > -51, `ниже: ${flat.cents} центов`)
})

test("C5: метка «настроено» только при отклонении не более 5 центов", () => {
  assert.equal(matchString(shiftCents(110.0, 0)).tuned, true)
  assert.equal(matchString(shiftCents(110.0, 4.9)).tuned, true)
  assert.equal(matchString(shiftCents(110.0, 5.1)).tuned, false)
  assert.equal(matchString(shiftCents(110.0, 6)).tuned, false)
  assert.equal(TUNED_CENTS, 5)
})

test("C7: сильно расстроенная открытая струна остаётся распознанной", () => {
  for (const cents of [60, 90, 95, 99.5, -60, -90, -95, -99.5]) {
    const match = matchString(shiftCents(110.0, cents))
    assert.ok(match, `не распознано при ${cents} центах`)
    assert.equal(match.string.name, "A2")
    assert.equal(match.tuned, false)
  }
  // Строгая граница полутона: за пределами 100 центов открытая струна не распознаётся.
  assert.equal(matchString(shiftCents(110.0, 100.5)), null)
  assert.equal(RECOGNIZE_CENTS, 100)
})

test("C8: тишина, шум и прижатая нота не выделяют струну", () => {
  // Тишина.
  assert.equal(detectPitch(new Float32Array(4096), 48000), null)
  assert.equal(matchString(detectPitch(new Float32Array(4096), 48000)), null)

  // Посторонний шум.
  assert.equal(detectPitch(noise(4096), 48000), null)

  // Прижатая нота, не совпадающая с открытыми струнами (F2 — на 200 центов выше E2).
  const f2 = shiftCents(STRINGS[0].frequency, 200)
  const detected = detectPitch(tone(f2, 48000, 4096), 48000)
  assert.ok(Number.isFinite(detected), "прижатая нота должна распознаваться как тон")
  assert.equal(matchString(detected), null)

  // Некорректный вход не даёт ложного сопоставления.
  assert.equal(matchString(null), null)
  assert.equal(matchString(NaN), null)
  assert.equal(matchString(0), null)
})
