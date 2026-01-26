import llama3pure from "./llama3pure-nodejs-engine.js"
import { execSync } from "child_process"
import fs from "fs"

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

const prompt = "Tell me in 1 line what is Microsoft."

const testModelUsingC = (model) => {
  try {
    execSync(`./llama3pure-c-engine -model "${model}" -prompt "${prompt}"`, {
      stdio: "inherit",
    })
  } catch (error) {
    console.error(error.message)
  }
}

const testModelUsingNode = (model) => {
  const buffer = fs.readFileSync(model)
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  )

  llama3pure({
    type: "load",
    arrayBuffer: arrayBuffer,
    filename: model,
    cbRender: function (token) {
      process.stdout.write(token)
    },
  })
  llama3pure({ type: "generate", prompt: "Tell me in 1 line what is Microsoft." })
  process.stdout.write("\n")
}

console.log("\x1b[1mRunning C tests...\x1b[0m")

models.forEach(function (model) {
  console.log("\x1b[1m" + model + "\x1b[0m")
  testModelUsingC(model)
})

console.log("\x1b[1mRunning Node.js tests...\x1b[0m")

models.forEach(function (model) {
  console.log("\x1b[1m" + model + "\x1b[0m")
  testModelUsingNode(model)
})
