import llama3pure from "./llama3pure-nodejs-engine.js"
import { fileURLToPath } from "url"
import path from "path"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const model = "gemma-3-270m-it-Q8_0.gguf"
const modelPath = path.resolve(__dirname, model)

console.log("Testing: " + model)

try {
  console.log("Loading model...")
  llama3pure({
    type: "load",
    filePath: modelPath,
    filename: model,
    maxTokens: 32,
    contextSize: 512,
    cbRender: function (token) {
      process.stdout.write(token)
    },
  })
  console.log("Model loaded. Generating...")
  llama3pure({
    type: "generate",
    chatHistory: [
      { role: "user", content: "Hello" },
    ],
  })
  console.log("\nDone.")
} catch (e) {
  console.error("ERROR:", e.message)
  console.error(e.stack)
}
