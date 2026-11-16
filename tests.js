import { execSync, spawnSync } from "child_process"
import Module from "module"
import { fileURLToPath } from "url"
import path from "path"
import fs from "fs"
import v8 from "v8"
import vm from "vm"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const ENGINE = "llama3pure-js-engine.js"
const ENGINE_DEV = "llama3pure-js-engine-dev.js"
const PERF_ROUNDS = 5

// the engine is loaded on demand, so a process only holds the engine it measures. it is
// always compiled under the name of the engine, because the heap in use depends on the
// length of the file name: with its own name, an identical dev engine has more heap
let llama3pure = null
const loadEngine = (engineFile) => {
  const enginePath = path.resolve(__dirname, ENGINE)
  const engineModule = new Module(enginePath)
  engineModule.filename = enginePath
  engineModule._compile(
    fs.readFileSync(path.resolve(__dirname, engineFile), "utf8"),
    enginePath
  )
  llama3pure = engineModule.exports
}

if (!process.execArgv.includes("--max-old-space-size=16384")) {
  execSync(
    "node --max-old-space-size=16384 " +
      __filename +
      " " +
      process.argv.slice(2).join(" "),
    { stdio: "inherit" }
  )
  process.exit()
}

const models = [
  "gemma-3-270m-it-Q2_K_L.gguf",
  "gemma-3-270m-it-Q3_K_M.gguf",
  "gemma-3-270m-it-Q4_K_M.gguf",
  "gemma-3-270m-it-Q5_K_M.gguf",
  "gemma-3-270m-it-Q6_K.gguf",
  "gemma-3-270m-it-Q8_0.gguf",
  "gemma-3-270m-it-F16.gguf",
  "gemma-3-1b-it-Q2_K_L.gguf",
  "gemma-3-1b-it-Q3_K_M.gguf",
  "gemma-3-1b-it-Q4_K_M.gguf",
  "gemma-3-1b-it-Q5_K_M.gguf",
  "gemma-3-1b-it-Q6_K.gguf",
  "gemma-3-1b-it-Q8_0.gguf",
  "gemma-3-1b-it-BF16.gguf",
  "Llama-3.2-1B-Instruct-Q3_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q4_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q5_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q6_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q8_0.gguf",
  "Llama-3.2-1B-Instruct-f16.gguf",
  "Llama-3.2-3B-Instruct-Q3_K_L.gguf",
  "Llama-3.2-3B-Instruct-Q4_K_L.gguf",
  "Llama-3.2-3B-Instruct-Q5_K_L.gguf",
  "Llama-3.2-3B-Instruct-Q6_K_L.gguf",
  "Llama-3.2-3B-Instruct-Q8_0.gguf",
  "Llama-3.2-3B-Instruct-f16.gguf",
  "gemma-3-4b-it-Q2_K_L.gguf",
  "gemma-3-4b-it-Q3_K_M.gguf",
  "gemma-3-4b-it-Q4_K_M.gguf",
  "gemma-3-4b-it-Q5_K_M.gguf",
  "gemma-3-4b-it-Q6_K.gguf",
  "gemma-3-4b-it-Q8_0.gguf",
  "gemma-3-4b-it-BF16.gguf",
  "Meta-Llama-3-8B-Instruct-Q2_K.gguf",
  "Meta-Llama-3-8B-Instruct-Q3_K_M.gguf",
  "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
  "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
  "Meta-Llama-3-8B-Instruct-Q6_K.gguf",
  "Meta-Llama-3-8B-Instruct-Q8_0.gguf",
  "Meta-Llama-3-8B-Instruct-fp16.gguf",
]

const readFileAsArrayBuffer = (filePath) => {
  const fd = fs.openSync(filePath, "r")
  const fileSize = fs.fstatSync(fd).size
  const arrayBuffer = new ArrayBuffer(fileSize)
  const fileUint8 = new Uint8Array(arrayBuffer)
  const chunkSize = 256 * 1024 * 1024
  let pos = 0
  while (pos < fileSize) {
    const toRead = Math.min(chunkSize, fileSize - pos)
    fs.readSync(fd, fileUint8, pos, toRead, pos)
    pos = pos + toRead
  }
  fs.closeSync(fd)
  return arrayBuffer
}

const testModelUsingC = (model) => {
  try {
    execSync(
      //`./llama3pure -model "${model}" -prompt "Tell me in one line what Microsoft is." -max_tokens -1 -context_size 2048`,
      `./llama3pure -model "${model}" -chathistory tests.txt -max_tokens -1 -context_size 2048`,
      {
        stdio: "inherit",
      }
    )
  } catch (error) {
    console.log(error.message)
  }
}

const testModelUsingNode = (model) => {
  const modelPath = path.resolve(__dirname, model)

  llama3pure({
    type: "load",
    model: readFileAsArrayBuffer(modelPath),
    maxTokens: -1,
    contextSize: 2048,
    cbRender: function (token) {
      process.stdout.write(token)
    },
  })

  llama3pure({
    type: "generate",
    chatHistory: [
      { type: "system", text: "You are a helpful assistant." },
      { type: "user", text: "Tell me in one line what Microsoft is." },
      {
        type: "model",
        response: [
          "Microsoft is a global technology leader known for its innovative products and services.",
        ],
      },
      { type: "user", text: "Tell me in one line who the founders are." },
    ],
  })

  process.stdout.write("\n")
}

const perfColumns = [
  "Model",
  "RAM start",
  "RAM after read",
  "RAM after load",
  "RAM after generate",
  "RAM peak",
  "Speed load",
  "Speed prefill",
  "Speed first token",
  "Speed time per token",
  "Speed total",
  "RAM after generate turn 2",
  "Speed prefill turn 2",
  "Speed first token turn 2",
  "Speed time per token turn 2",
  "RAM heap start",
  "RAM heap after load",
  "RAM heap after generate",
  "RAM heap after generate turn 2",
  "RAM buffers after load",
  "RAM buffers after generate",
  "RAM buffers after generate turn 2",
  "Reply",
  "Reply turn 2",
]

// one model, in a process of its own; its results go to the parent through fd 3.
// it is a chat of two turns with the two questions of the normal tests: the reply to the
// first question is sent back in the history of the second one
const testPerformanceOfModel = (engineFile, modelName) => {
  loadEngine(engineFile)

  const MB = 1024 * 1024
  const KB = 1024

  // the exact counters: the heap in use and the bytes of the buffers, read after a
  // forced collection so that only what is still referenced is counted. the memory of
  // the process is always read before the collection, and the time of the collections
  // is taken out of the total
  v8.setFlagsFromString("--expose-gc")
  const collect = vm.runInNewContext("gc")
  let tCollect = 0
  let modelBytes = 0
  const exactCounters = () => {
    const tStart = performance.now()
    collect()
    const usage = process.memoryUsage()
    tCollect = tCollect + (performance.now() - tStart)
    return {
      heap: usage.heapUsed / KB + " KB",
      buffers: (usage.arrayBuffers - modelBytes) / KB + " KB",
    }
  }

  const m0 = process.memoryUsage().rss
  const exactStart = exactCounters()
  tCollect = 0
  const t0 = performance.now()
  const modelPathPerf = path.resolve(__dirname, modelName)

  const buf = readFileAsArrayBuffer(modelPathPerf)

  const t1 = performance.now()
  const m1 = process.memoryUsage().rss
  modelBytes = buf.byteLength

  let tokenCount = 0
  let mPeak = Math.max(m0, m1)
  let tFirstToken = 0
  let tLastToken = 0
  let reply = ""

  llama3pure({
    type: "load",
    model: buf,
    cbRender: (token) => {
      tLastToken = performance.now()
      if (tokenCount === 0) {
        tFirstToken = tLastToken
      }
      tokenCount++
      reply = reply + token
      const r = process.memoryUsage().rss
      if (r > mPeak) {
        mPeak = r
      }
    },
    maxTokens: 256,
    contextSize: 2048,
    temperature: 0,
  })

  const t2 = performance.now()
  const m2 = process.memoryUsage().rss
  if (m2 > mPeak) {
    mPeak = m2
  }
  const exactLoad = exactCounters()

  // one turn: the memory after it and the times of its prefill, first token and tokens.
  // first token = prefill + one single-token pass, so the prefill is the time to the
  // first token minus the average time of a single-token pass
  const runTurn = (chatHistory) => {
    tokenCount = 0
    reply = ""

    const tStart = performance.now()

    llama3pure({ type: "generate", chatHistory: chatHistory })

    const ram = process.memoryUsage().rss
    if (ram > mPeak) {
      mPeak = ram
    }

    const firstToken = tokenCount > 0 ? tFirstToken - tStart : 0
    const timePerToken =
      tokenCount > 1 ? (tLastToken - tFirstToken) / (tokenCount - 1) : 0
    const prefill = tokenCount > 1 ? Math.max(0, firstToken - timePerToken) : 0
    const exact = exactCounters()

    return {
      ram: ram / MB + " MB",
      heap: exact.heap,
      buffers: exact.buffers,
      prefill: prefill + " ms",
      firstToken: firstToken + " ms",
      timePerToken: timePerToken + " ms",
      reply: reply,
    }
  }

  const chatHistory = [
    { type: "system", text: "You are a helpful assistant." },
    { type: "user", text: "Tell me in one line what Microsoft is." },
  ]

  const turn1 = runTurn(chatHistory)

  chatHistory.push({ type: "model", response: [reply] })
  chatHistory.push({
    type: "user",
    text: "Tell me in one line who the founders are.",
  })

  const turn2 = runTurn(chatHistory)

  const t3 = performance.now()

  const row = [
    modelName,
    m0 / MB + " MB",
    m1 / MB + " MB",
    m2 / MB + " MB",
    turn1.ram,
    mPeak / MB + " MB",
    t2 - t1 + " ms",
    turn1.prefill,
    turn1.firstToken,
    turn1.timePerToken,
    t3 - t0 - tCollect + " ms",
    turn2.ram,
    turn2.prefill,
    turn2.firstToken,
    turn2.timePerToken,
    exactStart.heap,
    exactLoad.heap,
    turn1.heap,
    turn2.heap,
    exactLoad.buffers,
    turn1.buffers,
    turn2.buffers,
    turn1.reply,
    turn2.reply,
  ]

  const result = {}
  perfColumns.forEach((column, index) => {
    result[column] = row[index]
  })

  fs.writeSync(3, JSON.stringify(result))
}

// the average of a column in the rounds of an engine, or null when a round of the model
// failed or the column is missing
const perfAverage = (rounds, model, column) => {
  let sum = 0
  for (let i = 0; i < rounds.length; i++) {
    const row = rounds[i].find((item) => item.Model === model)
    if (!row || row.Error || row[column] === undefined) {
      return null
    }
    sum = sum + parseFloat(row[column])
  }
  return rounds.length > 0 ? sum / rounds.length : null
}

// the lowest and the highest value of a column in the rounds of an engine, or null
// when a round of the model failed or the column is missing
const perfRange = (rounds, model, column) => {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < rounds.length; i++) {
    const row = rounds[i].find((item) => item.Model === model)
    if (!row || row.Error || row[column] === undefined) {
      return null
    }
    const value = parseFloat(row[column])
    min = Math.min(min, value)
    max = Math.max(max, value)
  }
  return rounds.length > 0 ? { min: min, max: max } : null
}

// true when a reply column is the same text in every round of both engines
const perfSameReplies = (baseRounds, devRounds, model, column) => {
  const replies = baseRounds.concat(devRounds).map((round) => {
    const row = round.find((item) => item.Model === model)
    return row ? row[column] : undefined
  })
  return replies.every((reply) => reply !== undefined && reply === replies[0])
}

// the report of the dev engine with respect to the engine: per model, the columns where
// it is faster, slower or equal and where it uses less, more or the same RAM, with the
// percentage of difference between the averages of the rounds. a difference has no
// overlap when every round of one engine is on the same side of every round of the
// other one; with overlap, the rounds of both engines are mixed
const perfReport = (baseRounds, devRounds, perfModels) => {
  const lines = []
  const totals = {
    better: 0,
    worse: 0,
    equal: 0,
    missing: 0,
    betterNoOverlap: 0,
    worseNoOverlap: 0,
  }
  const columns = perfColumns.filter(
    (column) => column.startsWith("RAM ") || column.startsWith("Speed ")
  )
  const percent = (base, dev) => {
    const value = base !== 0 ? (dev / base - 1) * 100 : 0
    return (value > 0 ? "+" : "") + value.toFixed(2) + " %"
  }

  lines.push(
    "Results of " +
      ENGINE_DEV +
      " with respect to " +
      ENGINE +
      " (average of " +
      PERF_ROUNDS +
      " rounds per engine)"
  )

  perfModels.forEach((model) => {
    const groups = {
      Faster: [],
      Slower: [],
      "Same speed": [],
      "Less RAM": [],
      "More RAM": [],
      "Same RAM": [],
      "Not measured": [],
    }

    columns.forEach((column) => {
      const isRam = column.startsWith("RAM ")
      const base = perfAverage(baseRounds, model, column)
      const dev = perfAverage(devRounds, model, column)
      const baseRange = perfRange(baseRounds, model, column)
      const devRange = perfRange(devRounds, model, column)
      if (base === null || dev === null) {
        groups["Not measured"].push(column)
        totals.missing++
      } else if (dev < base) {
        const noOverlap = devRange.max < baseRange.min
        groups[isRam ? "Less RAM" : "Faster"].push(
          column + " " + percent(base, dev) + (noOverlap ? " (no overlap)" : " (overlap)")
        )
        totals.better++
        totals.betterNoOverlap += noOverlap ? 1 : 0
      } else if (dev > base) {
        const noOverlap = devRange.min > baseRange.max
        groups[isRam ? "More RAM" : "Slower"].push(
          column + " " + percent(base, dev) + (noOverlap ? " (no overlap)" : " (overlap)")
        )
        totals.worse++
        totals.worseNoOverlap += noOverlap ? 1 : 0
      } else {
        groups[isRam ? "Same RAM" : "Same speed"].push(column + " 0.00 %")
        totals.equal++
      }
    })

    lines.push("")
    lines.push(model)
    Object.keys(groups).forEach((group) => {
      if (groups[group].length > 0) {
        lines.push("  " + group + ": " + groups[group].join(", "))
      }
    })
    lines.push(
      "  Replies: " +
        (perfSameReplies(baseRounds, devRounds, model, "Reply") &&
        perfSameReplies(baseRounds, devRounds, model, "Reply turn 2")
          ? "identical"
          : "different")
    )
  })

  lines.push("")
  lines.push(
    "Total: " +
      totals.better +
      " faster or with less RAM, " +
      totals.worse +
      " slower or with more RAM, " +
      totals.equal +
      " equal, " +
      totals.missing +
      " not measured"
  )
  lines.push(
    "Without overlap between the rounds of both engines: " +
      totals.betterNoOverlap +
      " faster or with less RAM, " +
      totals.worseNoOverlap +
      " slower or with more RAM"
  )

  return lines.join("\n")
}

// every model found in the folder (or the ones given as arguments), one process at a
// time so the memory of a model is not counted in the next one. the rounds alternate
// (engine, dev engine, engine, dev engine...) so the order does not favor either of
// them. nothing is saved: the report of the dev engine with respect to the engine is
// printed at the end
const testPerformance = (requestedModels) => {
  if (!fs.existsSync(path.resolve(__dirname, ENGINE_DEV))) {
    console.log(ENGINE_DEV + " was not found: there is nothing to compare.")
    return
  }

  const perfModels = (requestedModels.length > 0 ? requestedModels : models).filter(
    (model) => fs.existsSync(path.resolve(__dirname, model))
  )
  const results = {}

  results[ENGINE] = []
  results[ENGINE_DEV] = []

  for (let round = 0; round < PERF_ROUNDS; round++) {
    ;[ENGINE, ENGINE_DEV].forEach((engineFile) => {
      const rows = []
      results[engineFile].push(rows)

      perfModels.forEach((model) => {
        const result = spawnSync(
          process.execPath,
          [
            "--max-old-space-size=16384",
            __filename,
            "perf-model",
            engineFile === ENGINE ? "0" : "1",
            model,
          ],
          { stdio: ["ignore", "ignore", "inherit", "pipe"] }
        )

        const output =
          result.output && result.output[3] ? result.output[3].toString() : ""
        if (result.status === 0 && output.length > 0) {
          rows.push(JSON.parse(output))
        } else {
          rows.push({ Model: model, Error: "The performance test failed." })
        }
      })
    })
  }

  console.log(perfReport(results[ENGINE], results[ENGINE_DEV], perfModels))
}

const testAllModels = () => {
  const availableModels = models.filter((model) => fs.existsSync(model))

  const jsBuildExists = fs.existsSync(ENGINE)
  const nativeBuildExists =
    fs.existsSync("llama3pure") || fs.existsSync("llama3pure.exe")

  if (nativeBuildExists && availableModels.length > 0) {
    console.log("\x1b[1mRunning C tests...\x1b[0m")

    availableModels.forEach(function (model) {
      console.log("\x1b[1m" + model + "\x1b[0m")
      testModelUsingC(model)
    })
  }

  if (jsBuildExists && availableModels.length > 0) {
    console.log("\x1b[1mRunning Node.js tests...\x1b[0m")
    loadEngine(ENGINE)

    models.forEach(function (model) {
      if (fs.existsSync(model)) {
        console.log("\x1b[1m" + model + "\x1b[0m")
        testModelUsingNode(model)
      }
    })
  }
}

if (process.argv[2] === "perf") {
  testPerformance(process.argv.slice(3))
} else if (process.argv[2] === "perf-model") {
  // the engine is given as 0 or 1 and not by its name: the arguments are in the heap, and
  // a longer name would count as more heap in use
  testPerformanceOfModel(
    [ENGINE, ENGINE_DEV][process.argv[3]] || process.argv[3],
    process.argv[4]
  )
} else {
  testAllModels()
}
