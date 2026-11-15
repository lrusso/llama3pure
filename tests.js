import llama3pure from "./llama3pure-js-engine.js"
import { execSync, spawnSync } from "child_process"
import { fileURLToPath } from "url"
import path from "path"
import fs from "fs"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

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
      { type: "user", text: "Tell me in 1 line the names of the founders." },
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
  "Reply",
  "Reply turn 2",
]

// one model, in a process of its own; its results go to the parent through fd 3.
// it is a chat of two turns with the two questions of the normal tests: the reply to the
// first question is sent back in the history of the second one
const testPerformanceOfModel = (modelName) => {
  const MB = 1024 * 1024
  const m0 = process.memoryUsage().rss
  const t0 = performance.now()
  const modelPathPerf = path.resolve(__dirname, modelName)

  const buf = readFileAsArrayBuffer(modelPathPerf)

  const t1 = performance.now()
  const m1 = process.memoryUsage().rss

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

    return {
      ram: ram / MB + " MB",
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
  chatHistory.push({ type: "user", text: "Tell me in 1 line the names of the founders." })

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
    t3 - t0 + " ms",
    turn2.ram,
    turn2.prefill,
    turn2.firstToken,
    turn2.timePerToken,
    turn1.reply,
    turn2.reply,
  ]

  const result = {}
  perfColumns.forEach((column, index) => {
    result[column] = row[index]
  })

  fs.writeSync(3, JSON.stringify(result))
}

// every model found in the folder (or the ones given as arguments), one process at a
// time so the memory of a model is not counted in the next one. nothing is printed:
// the results are saved in results.json, which is written again after every model
const testPerformance = (requestedModels) => {
  const perfModels = (requestedModels.length > 0 ? requestedModels : models).filter(
    (model) => fs.existsSync(path.resolve(__dirname, model))
  )
  const resultsPath = path.resolve(__dirname, "results.json")
  const results = []

  perfModels.forEach((model) => {
    const result = spawnSync(
      process.execPath,
      ["--max-old-space-size=16384", __filename, "perf-model", model],
      { stdio: ["ignore", "ignore", "inherit", "pipe"] }
    )

    const output = result.output && result.output[3] ? result.output[3].toString() : ""
    if (result.status === 0 && output.length > 0) {
      results.push(JSON.parse(output))
    } else {
      results.push({ Model: model, Error: "The performance test failed." })
    }

    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2) + "\n")
  })
}

const testAllModels = () => {
  const availableModels = models.filter((model) => fs.existsSync(model))

  const jsBuildExists = fs.existsSync("llama3pure-js-engine.js")
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
  testPerformanceOfModel(process.argv[3])
} else {
  testAllModels()
}
