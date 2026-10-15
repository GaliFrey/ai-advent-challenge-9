// Чистый модуль определения основного тона и сопоставления открытой струны.
// Не зависит от DOM и Node-глобалов: один и тот же файл импортируется
// браузером как ES-модуль и Node-тестами без сборки и зависимостей.

// Опорная нота настройки.
export const A4 = 440

// Стандартный строй от толстой струны к тонкой.
export const STRINGS = [
  { index: 0, name: "E2", note: "E", label: "6-я струна", frequency: 82.41 },
  { index: 1, name: "A2", note: "A", label: "5-я струна", frequency: 110.0 },
  { index: 2, name: "D3", note: "D", label: "4-я струна", frequency: 146.83 },
  { index: 3, name: "G3", note: "G", label: "3-я струна", frequency: 196.0 },
  { index: 4, name: "B3", note: "B", label: "2-я струна", frequency: 246.94 },
  { index: 5, name: "E4", note: "E", label: "1-я струна", frequency: 329.63 },
]

// Порог метки «настроено» (C5) и порог распознавания открытой струны (C7).
export const TUNED_CENTS = 5
export const RECOGNIZE_CENTS = 100

// Диапазон поиска основного тона с запасом покрывает E2 (82.41 Гц) и E4 (329.63 Гц).
const MIN_FREQUENCY = 65
const MAX_FREQUENCY = 450
// Минимальный нормированный пик автокорреляции (отсекает тишину и шум, C8).
const MIN_CONFIDENCE = 0.6
// Доля от максимального пика, при которой выбирается первый кандидат (защита от октавных ошибок).
const PEAK_RATIO = 0.9
// Порог среднеквадратичной амплитуды (отсекает тишину, C8).
const MIN_RMS = 0.005

// Отклонение частоты от целевой в центах: > 0 — звук выше цели, < 0 — ниже.
export function centsBetween(frequency, target) {
  return 1200 * Math.log2(frequency / target)
}

// Оценка частоты основного тона по временному буферу.
// Принимает Float32Array (а также обычный массив) и sampleRate.
// Возвращает частоту в Гц либо null при тишине или неуверенном сигнале.
export function detectPitch(samples, sampleRate) {
  if (!samples || !Number.isFinite(sampleRate) || sampleRate <= 0) return null
  const length = samples.length
  const minLag = Math.max(2, Math.floor(sampleRate / MAX_FREQUENCY))
  const maxLag = Math.min(length - 2, Math.ceil(sampleRate / MIN_FREQUENCY))
  if (maxLag <= minLag) return null

  let energy = 0
  for (let i = 0; i < length; i += 1) {
    const value = samples[i]
    energy += value * value
  }
  const rms = Math.sqrt(energy / length)
  if (!(rms > MIN_RMS)) return null

  // Нормированная автокорреляционная функция (NSDF).
  const nsdf = new Float64Array(maxLag + 2)
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let correlation = 0
    let norm = 0
    const limit = length - lag
    for (let i = 0; i < limit; i += 1) {
      const a = samples[i]
      const b = samples[i + lag]
      correlation += a * b
      norm += a * a + b * b
    }
    nsdf[lag] = norm > 0 ? (2 * correlation) / norm : 0
  }

  // Локальные максимумы и наибольший из них.
  const peaks = []
  let maxPeak = 0
  for (let lag = minLag + 1; lag < maxLag; lag += 1) {
    if (nsdf[lag] > nsdf[lag - 1] && nsdf[lag] >= nsdf[lag + 1]) {
      peaks.push(lag)
      if (nsdf[lag] > maxPeak) maxPeak = nsdf[lag]
    }
  }
  if (peaks.length === 0 || maxPeak < MIN_CONFIDENCE) return null

  // Первый (наименьший лаг) достаточно сильный пик — основной тон, а не субгармоника.
  const threshold = PEAK_RATIO * maxPeak
  let chosen = -1
  for (const lag of peaks) {
    if (nsdf[lag] > 0 && nsdf[lag] >= threshold) {
      chosen = lag
      break
    }
  }
  if (chosen < 0) return null

  // Параболическая интерполяция вершины для субдискретной точности.
  let refined = chosen
  const y0 = nsdf[chosen - 1]
  const y1 = nsdf[chosen]
  const y2 = nsdf[chosen + 1]
  const denom = y0 - 2 * y1 + y2
  if (denom !== 0) {
    const delta = (0.5 * (y0 - y2)) / denom
    if (Math.abs(delta) <= 1) refined = chosen + delta
  }
  if (!(refined > 0)) return null

  const frequency = sampleRate / refined
  if (frequency < MIN_FREQUENCY || frequency > MAX_FREQUENCY) return null
  return frequency
}

// Автоматическое сопоставление частоты с одной из шести открытых струн.
// Возвращает null, если минимальное по модулю отклонение не меньше 100 центов
// (тишина, шум, прижатая нота) — C8. Иначе — распознанную струну, знак и
// величину отклонения в центах и признак «настроено» (C3, C4, C5, C7).
export function matchString(frequency) {
  if (!Number.isFinite(frequency) || frequency <= 0) return null
  let best = null
  for (let i = 0; i < STRINGS.length; i += 1) {
    const string = STRINGS[i]
    const cents = centsBetween(frequency, string.frequency)
    if (best === null || Math.abs(cents) < Math.abs(best.cents)) {
      best = { index: i, string, cents }
    }
  }
  if (Math.abs(best.cents) >= RECOGNIZE_CENTS) return null
  return {
    index: best.index,
    string: best.string,
    cents: best.cents,
    direction: best.cents > 0 ? "high" : best.cents < 0 ? "low" : "in-tune",
    tuned: Math.abs(best.cents) <= TUNED_CENTS,
  }
}
