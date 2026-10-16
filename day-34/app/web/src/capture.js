// Оркестрация стадий захвата микрофона и цикла кадров. Браузерные вызовы
// внедряются, поэтому переходы стадий, сохранение результатов кадра и
// обработка ошибок проверяются Node-тестами с подставными объектами. Реальные
// вызовы Web Audio сосредоточены в тонком адаптере web/src/app.js.
//
// Стадии запуска отслеживаются явно и не выводятся из флага «идёт попытка»:
// requesting → (getUserMedia) → initializing → (аудиограф) → capturing.
// Ошибка регистрируется нормализованным именем (см. diagnostics.js); цикл
// ошибочных кадров не продолжается, исключение кадра не маскируется.

// Среднеквадратичный уровень сигнала. Считается один раз за кадр и передаётся
// и в автомат показаний, и в диагностический снимок.
function rmsOf(samples) {
  let energy = 0
  for (let i = 0; i < samples.length; i += 1) energy += samples[i] * samples[i]
  return Math.sqrt(energy / samples.length)
}

export function createCapture(deps) {
  const {
    diagnostics,
    readingState,
    getUserMedia,
    createContext,
    createSource,
    createAnalyser,
    createBuffer,
    requestFrame,
    now,
    detectPitch,
    matchString,
    updateReading,
    committedAtOf,
    render,
    onRequesting,
    onListening,
    onFailure,
  } = deps

  let active = false
  let analyser = null
  let context = null
  let buffer = null

  // Один кадр обработки. Время читается один раз и используется и как `now` для
  // автомата показаний, и как монотонная метка измерения. При сбое цикл не
  // перепланируется, ошибка регистрируется, исключение перебрасывается дальше —
  // оно не маскируется.
  function frame() {
    try {
      const time = now()
      analyser.getFloatTimeDomainData(buffer)
      const frequency = detectPitch(buffer, context.sampleRate)
      const match = matchString(frequency)
      const level = rmsOf(buffer)
      const reading = updateReading(readingState, { frequency, match, level }, time)
      diagnostics.recordMeasurement({
        rms: level,
        frequencyHz: frequency,
        matchedString: match ? match.string.name : null,
        atMono: time,
      })
      diagnostics.recordReading(reading, committedAtOf(readingState))
      render(reading)
      requestFrame(frame)
    } catch (error) {
      diagnostics.failCapture(error)
      if (onFailure) onFailure(error)
      throw error
    }
  }

  async function start() {
    if (active) return
    active = true
    diagnostics.beginRequest()
    if (onRequesting) onRequesting()
    try {
      const stream = await getUserMedia()
      diagnostics.beginInitializing()
      context = createContext()
      const source = createSource(context, stream)
      analyser = createAnalyser(context)
      analyser.fftSize = 4096
      source.connect(analyser)
      buffer = createBuffer(analyser.fftSize)
      const track =
        stream && typeof stream.getAudioTracks === "function"
          ? stream.getAudioTracks()[0] || null
          : null
      diagnostics.beginCapturing({ track, context })
      if (onListening) onListening()
      requestFrame(frame)
    } catch (error) {
      active = false
      diagnostics.failCapture(error)
      if (onFailure) onFailure(error)
    }
  }

  return { start }
}
