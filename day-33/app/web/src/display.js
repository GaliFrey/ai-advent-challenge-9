// Чистый модуль представления показаний тюнера. Не зависит от DOM и аудио;
// время передаётся параметром `now`, поэтому поведение детерминированно
// проверяется Node-тестами.
//
// Реализует:
// - C8/C10: тишина удерживает последнее уверенное показание бессрочно; показ
//   снимает только посторонний тон (не открытая струна), который держится
//   суммарно не менее ~0,25 с, — одиночный сбойный кадр показание не стирает;
//   тишина и пропуск определения высоты различаются по уровню сигнала, а не по
//   отсутствию распознанной ноты (C10);
// - C4/C5/C9/C11: сглаживание отклонения и согласованность надписей: внутри
//   диапазона настройки (±5 центов) видна только метка «настроено» без числа и
//   направления, а вне диапазона направление показывается вместе с величиной;
// - C12: показ меняется не чаще трёх раз в секунду и не перескакивает между
//   далёкими значениями;
// - C13: неустойчивое начало щипка (~первые 0,3 с) и затухающий хвост (уровень
//   сигнала упал ниже ~трети максимума щипка) не выводятся на экран — оценка
//   берётся по устойчивой середине звука; единичный пропуск определения высоты
//   (в том числе низкоуровневый в затухающем хвосте) не считается новым щипком
//   или тишиной и не сбрасывает окно оценки, максимум уровня и метку
//   «настроено»; затухающий звук не переоценивается до нового щипка, а новый
//   щипок — это заметный возврат громкости (~в 1,5 раза) или звук после
//   подтверждённой тишины;
// - C14: метка «настроено» включается однократно за звучание и не мигает у
//   границы ±5 центов, а противоречивый сигнал показывает состояние
//   «звук не распознан»;
// - C15: первый показ нового щипка/струны берётся по устойчивой середине сразу
//   после окна оценки и не отменяется гейтом затухающего хвоста, поэтому щипок
//   любой струны (в том числе быстро затухающей 3-й) появляется на экране не
//   позднее ~0,5 с, а не остаётся прежняя струна.

import { STRINGS, centsBetween, RECOGNIZE_CENTS } from "./pitch.js"

// Сколько последних кадров усредняется в показываемом отклонении (C9).
export const SMOOTH_FRAMES = 8
// Окно стабилизации после начала щипка: пока оно идёт, показ не меняется (C13).
export const SETTLE_MS = 300
// Минимальный интервал между изменениями показа (C12: не чаще трёх раз в секунду).
export const UPDATE_INTERVAL_MS = 340
// Окно и число смен распознанной струны, при которых сигнал противоречив (C14).
export const CONTRADICTION_WINDOW_MS = 500
export const CONTRADICTION_CHANGES = 3
// Посторонний тон (не открытая струна) снимает показ, только если держится
// суммарно не менее этого времени; одиночный сбойный кадр показание не стирает
// (C10).
export const FOREIGN_TONE_MS = 250
// Разрыв между кадрами постороннего тона больше этого значения не суммируется
// как длительность тона (пропущенный кадр детектора).
const FOREIGN_FRAME_GAP_MS = 100
// Расширенная зона удержания метки «настроено»: пока сглаженное отклонение не
// вышло за неё, зафиксированное показание не пересчитывается, поэтому метка не
// мигает у границы ±5 центов (C14). Пока метка показана, число и направление
// скрыты — показание внутри допуска несёт только метку (C5, C11).
export const TUNED_RELEASE_CENTS = 8
// Доля от максимума уровня щипка, ниже которой звук считается затухающим
// хвостом: пока уровень ниже неё, показ и метка не пересчитываются — оценка
// берётся по устойчивой середине звука (C13).
export const TAIL_LEVEL_RATIO = 1 / 3
// Затухающий звук не переоценивается до нового щипка (C13). Новый щипок
// распознаётся по заметному возврату громкости: уровень вырос примерно в
// полтора раза относительно текущего уровня хвоста, а не относительно максимума
// щипка — иначе настоящий новый щипок не отличался бы от продолжения хвоста.
export const PLUCK_GROWTH_RATIO = 1.5
// Тишина подтверждается, только если уровень сигнала остаётся ниже
// абсолютного пола SILENCE_LEVEL_FLOOR не менее SILENCE_MS: одиночный кадр без
// определённой высоты (в том числе низкоуровневый в затухающем хвосте) — не
// тишина и не новый щипок (C10, C13).
export const SILENCE_MS = 200
export const SILENCE_LEVEL_FLOOR = 0.02

function stringByName(name) {
  for (const string of STRINGS) {
    if (string.name === name) return string
  }
  return null
}

function average(window) {
  let sum = 0
  for (const value of window) sum += value
  return window.length > 0 ? sum / window.length : 0
}

// Есть ли в кадре сигнал, достаточный для того, чтобы считать отсутствие
// определённой высоты пропуском детектора, а не тишиной (C10, C13). Пока есть
// активный щипок, уровень сравнивается с его максимумом: пока он не упал ниже
// доли затухающего хвоста, звук продолжается. Пока щипка нет, сигналом
// считается любой ненулевой уровень. Без `level` сигнала нет (обратная
// совместимость: пропуск и тишина не различаются, как раньше).
function hasSignal(state, level) {
  if (level === null) return false
  if (state.levelMax > 0) return level >= state.levelMax * TAIL_LEVEL_RATIO
  return level > 0
}

// Тишина — это отсутствие сигнала: уровень не задан или лежит ниже
// абсолютного пола. Пропуск определения высоты при значимом уровне тишиной не
// является, поэтому одиночный низкоуровневый кадр в хвосте не считается
// тишиной и не начинает новый щипок (C10, C13).
function isBelowFloor(level) {
  return level === null || level < SILENCE_LEVEL_FLOOR
}

// Единое правило отображения (C4, C5, C11): округлённое значение задаёт и
// направление, и метку «настроено». Внутри диапазона настройки (|displayed| <= 5)
// показывается только метка «настроено»: ни числового отклонения, ни подсказки
// направления нет, поэтому небольшие колебания внутри допуска не видны (C5).
// Вне диапазона направление показывается всегда вместе с величиной отклонения
// (C4, C11).
function labelsForCents(smoothed) {
  let displayed = Math.sign(smoothed) * Math.round(Math.abs(smoothed))
  if (displayed === 0) displayed = 0
  if (Math.abs(displayed) <= 5) {
    return { direction: "", cents: null, tuned: true }
  }
  const direction = displayed > 0 ? "Ослабить" : "Подтянуть"
  return { direction, cents: displayed, tuned: false }
}

function emptyReading() {
  return { activeNote: null, direction: null, cents: null, tuned: false, hint: true }
}

function readingFromDisplay(activeNote, display) {
  if (!display) return emptyReading()
  return {
    activeNote,
    direction: display.direction,
    cents: display.cents,
    tuned: display.tuned,
    hint: false,
  }
}

function resetState(state) {
  state.window = []
  state.activeNote = null
  state.display = null
  state.settleUntil = 0
  state.awaitingCommit = false
  state.toneActive = false
  state.lastTargetName = null
  state.rawNote = null
  state.rawChanges = []
  state.foreignMs = 0
  state.foreignLastAt = null
  state.tunedLatched = false
  state.levelMax = 0
  state.tailActive = false
  state.tailLevel = null
  state.silentSince = null
  // Снятие показания очищает метку фактической фиксации: возраста фиксации
  // без показания быть не должно.
  state.committedAt = null
}

function commit(state, note, labels, time) {
  state.activeNote = note
  state.display = labels
  state.lastUpdateAt = time
  // Отдельная nullable-метка фактической фиксации показания. В отличие от
  // lastUpdateAt она не инициализируется -Infinity и очищается при снятии
  // показания, поэтому пригодна для контракта снимка (commit.ageMs).
  state.committedAt = time
  state.awaitingCommit = false
}

export function createReadingState() {
  return {
    window: [],
    activeNote: null,
    display: null,
    settleUntil: 0,
    awaitingCommit: false,
    toneActive: false,
    lastTargetName: null,
    lastUpdateAt: -Infinity,
    committedAt: null,
    rawNote: null,
    rawChanges: [],
    contradictionUntil: 0,
    foreignMs: 0,
    foreignLastAt: null,
    tunedLatched: false,
    levelMax: 0,
    tailActive: false,
    tailLevel: null,
    silentSince: null,
  }
}

// Монотонное время фактической фиксации текущего показания либо null, если
// показание не зафиксировано. Позволяет диагностике получить метку commit, не
// отдавая наружу сам readingState.
export function committedAtOf(state) {
  return state && Number.isFinite(state.committedAt) ? state.committedAt : null
}

// Обновляет состояние показаний по одному кадру.
// frame: { frequency, match, level } — результат detectPitch/matchString (как в
// app.js) и необязательный уровень сигнала (RMS) для гейта затухающего хвоста;
// без level гейт хвоста не применяется (обратная совместимость).
// now: время в мс (в браузере — timestamp из requestAnimationFrame).
export function updateReading(state, frame, now) {
  const frequency = frame ? frame.frequency : null
  const match = frame ? frame.match : null
  const level = frame && Number.isFinite(frame.level) && frame.level >= 0 ? frame.level : null
  const time = Number.isFinite(now) ? now : 0
  const hasTone = Number.isFinite(frequency) && frequency > 0

  // (б) Тон есть, но это не открытая струна (прижатая нота, шум). Единичный
  // сбойный кадр показание не снимает: показываем прежнее, пока посторонний тон
  // не накопится суммарно не менее ~0,25 с (C8, C10).
  if (hasTone && !match) {
    if (state.foreignLastAt !== null) {
      const delta = time - state.foreignLastAt
      if (delta > 0 && delta <= FOREIGN_FRAME_GAP_MS) state.foreignMs += delta
    }
    state.foreignLastAt = time
    state.silentSince = null
    if (state.foreignMs >= FOREIGN_TONE_MS) {
      resetState(state)
      state.contradictionUntil = 0
      return emptyReading()
    }
    return readingFromDisplay(state.activeNote, state.display)
  }

  // (а) Высота не определилась. Это может быть тишина или пропуск детектора;
  // различаем их по уровню сигнала, а не по отсутствию распознанной ноты (C10).
  // Пропуск при высоком уровне — продолжение того же звука, а не новый щипок и
  // не тишина: окно оценки, максимум уровня и метка «настроено» не сбрасываются
  // (C13, C14).
  if (!match) {
    if (hasSignal(state, level)) {
      // Максимум уровня может только вырасти — его не сбрасываем (C13).
      if (level > state.levelMax) state.levelMax = level
      state.silentSince = null
      return readingFromDisplay(state.activeNote, state.display)
    }

    // Низкий уровень. Пока идёт активный щипок (в том числе его затухающий
    // хвост), одиночный кадр без высоты — не тишина: показ, окно оценки,
    // максимум уровня и метка «настроено» сохраняются (C10, C13, C14).
    // Затухающий звук не переоценивается до нового щипка.
    if (state.toneActive) {
      // Реальный затухающий хвост отслеживается по уровню; без уровня его
      // отличить нельзя, но одиночный кадр всё равно не тишина (C13).
      if (level !== null) {
        state.tailActive = true
        state.tailLevel = level
      }
      // Тишина подтверждается только по длительному отсутствию сигнала; лишь
      // после неё следующий уверенный звук считается новым щипком (C13).
      if (state.silentSince === null) state.silentSince = time
      if (time - state.silentSince >= SILENCE_MS && isBelowFloor(level)) {
        state.toneActive = false
        state.tailActive = false
        state.tailLevel = null
        state.silentSince = null
        state.foreignMs = 0
        state.foreignLastAt = null
      }
      return readingFromDisplay(state.activeNote, state.display)
    }

    // Уже тишина: показание сохраняется бессрочно до следующего уверенного
    // звука (C8, C10).
    state.silentSince = null
    return readingFromDisplay(state.activeNote, state.display)
  }

  // Распознанная струна — уверенный звук, накопитель постороннего тона сброшен.
  state.foreignMs = 0
  state.foreignLastAt = null
  state.silentSince = null

  // Детекция противоречивого сигнала: частая смена распознанной струны (C14).
  const rawName = match.string.name
  if (state.rawNote !== rawName) {
    state.rawChanges.push(time)
    state.rawNote = rawName
  }
  state.rawChanges = state.rawChanges.filter((t) => time - t <= CONTRADICTION_WINDOW_MS)
  if (state.rawChanges.length >= CONTRADICTION_CHANGES) {
    resetState(state)
    state.contradictionUntil = time + SETTLE_MS
    return emptyReading()
  }
  if (time < state.contradictionUntil) {
    // После противоречия держим «звук не распознан», пока сигнал не станет устойчивым.
    return readingFromDisplay(state.activeNote, state.display)
  }

  // Гистерезис текущей струны: пока выбранная струна ещё в пределах
  // распознавания, не переключаемся на соседнюю (C7, C9).
  let target = match.string
  if (state.activeNote) {
    const current = stringByName(state.activeNote)
    if (current && Math.abs(centsBetween(frequency, current.frequency)) < RECOGNIZE_CENTS) {
      target = current
    }
  }

  const cents = centsBetween(frequency, target.frequency)

  // Начало щипка: возобновление после подтверждённой тишины или смена
  // распознанной струны. Пока окно стабилизации не прошло, показ не меняется
  // (C13). Новый щипок той же струны во время затухающего хвоста распознаётся по
  // заметному возврату громкости (C13).
  const resumed = !state.toneActive
  const stringChanged = state.lastTargetName !== target.name
  // Новый щипок во время затухающего хвоста распознаётся по заметному возврату
  // громкости относительно текущего уровня хвоста, а не относительно максимума
  // щипка (C13).
  const newPluck =
    state.tailActive &&
    level !== null &&
    state.tailLevel !== null &&
    level > state.tailLevel * PLUCK_GROWTH_RATIO
  if (resumed || stringChanged || newPluck) {
    state.window = []
    state.settleUntil = time + SETTLE_MS
    state.awaitingCommit = true
    // Новый щипок/струна — прежняя фиксация метки «настроено» снимается (C14).
    state.tunedLatched = false
    // Максимум уровня считается заново для нового щипка (C13).
    state.levelMax = level !== null ? level : 0
    state.tailActive = false
    state.tailLevel = null
  }
  state.toneActive = true
  state.lastTargetName = target.name

  // Накопление максимума уровня текущего щипка, включая пик атаки (C13).
  if (level !== null && level > state.levelMax) state.levelMax = level

  state.window.push(cents)
  while (state.window.length > SMOOTH_FRAMES) state.window.shift()

  if (time < state.settleUntil) {
    // Неустойчивое начало щипка не выводится: остаётся предыдущее показание
    // или подсказка (C13).
    return readingFromDisplay(state.activeNote, state.display)
  }

  const smoothed = average(state.window)
  const labels = labelsForCents(smoothed)

  // Первый показ нового щипка/струны берётся по устойчивой середине сразу после
  // окна оценки и не отменяется гейтом затухающего хвоста: у быстро затухающей
  // струны (например 3-й) уровень успевает упасть ниже трети максимума ещё до
  // первого показа, и без этого на экране оставалась бы прежняя струна (C15).
  // Заморозка хвоста ниже относится только к последующим кадрам (C13).
  if (state.awaitingCommit) {
    commit(state, target.name, labels, time)
    state.tunedLatched = labels.tuned
    return readingFromDisplay(state.activeNote, state.display)
  }

  // Затухающий хвост: уровень упал ниже ~трети максимума щипка. Пока хвост
  // активен (и новый щипок не распознан), оценка берётся по устойчивой середине,
  // поэтому показ и метка «настроено» не пересчитываются (C13).
  const levelTail =
    level !== null && state.levelMax > 0 && level < state.levelMax * TAIL_LEVEL_RATIO
  if (levelTail || state.tailActive) {
    if (levelTail) state.tailLevel = level
    state.tailActive = true
    return readingFromDisplay(state.activeNote, state.display)
  }

  // Пока метка «настроено» зафиксирована, показание не пересчитывается, пока та
  // же струна и сглаженное отклонение не вышло из расширенной зоны удержания
  // (C14). Зафиксированное значение всегда в пределах ±5 центов (C11).
  if (state.tunedLatched) {
    if (state.activeNote === target.name && Math.abs(smoothed) <= TUNED_RELEASE_CENTS) {
      return readingFromDisplay(state.activeNote, state.display)
    }
    state.tunedLatched = false
  }

  // Первое показание (C9). Настроенное значение фиксируется лэтчем (C14).
  if (state.display === null) {
    commit(state, target.name, labels, time)
    state.tunedLatched = labels.tuned
    return readingFromDisplay(state.activeNote, state.display)
  }

  // Показ не меняется, если значение и струна те же (C9, C12, C14).
  if (state.activeNote === target.name && state.display.cents === labels.cents) {
    return readingFromDisplay(state.activeNote, state.display)
  }

  // Иначе значение меняется не чаще одного раза в UPDATE_INTERVAL_MS (C12).
  if (time - state.lastUpdateAt >= UPDATE_INTERVAL_MS) {
    commit(state, target.name, labels, time)
    state.tunedLatched = labels.tuned
  }
  return readingFromDisplay(state.activeNote, state.display)
}
