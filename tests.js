import llama3pure from "./llama3pure-js-engine.js"
import { execSync } from "child_process"
import { fileURLToPath } from "url"
import { existsSync } from "fs"
import path from "path"
import fs from "fs"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

if (!process.execArgv.includes("--max-old-space-size=16384")) {
  execSync("node --max-old-space-size=16384 " + __filename, { stdio: "inherit" })
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
      //`./llama3pure -model "${model}" -prompt "Tell me in 1 line what is Microsoft." -max_tokens -1 -context_size 2048`,
      `./llama3pure -model "${model}" -chathistory tests.txt -max_tokens -1 -context_size 2048`,
      {
        stdio: "inherit",
      }
    )
  } catch (error) {
    console.error(error.message)
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
      { role: "user", content: "Tell me in 1 line what is Microsoft." },
      {
        role: "assistant",
        content:
          "Microsoft is a global technology leader known for its innovative products and services.",
      },
      { role: "user", content: "Tell me in 1 line the names of the founders." },
    ],
  })

  process.stdout.write("\n")
}

console.log("\x1b[1mRunning C tests...\x1b[0m")

models.forEach(function (model) {
  if (existsSync(model)) {
    console.log("\x1b[1m" + model + "\x1b[0m")
    testModelUsingC(model)
  }
})

console.log("\x1b[1mRunning Node.js tests...\x1b[0m")

models.forEach(function (model) {
  if (existsSync(model)) {
    console.log("\x1b[1m" + model + "\x1b[0m")
    testModelUsingNode(model)
  }
})
