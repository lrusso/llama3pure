import llama3pure from "./llama3pure-nodejs-engine.js"
import path from "path"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function myRenderFunction(token) {
  process.stdout.write(token)
}

function main() {
  const modelPath = path.resolve(__dirname, "gemma-3-270m-it-Q8_0.gguf")

  llama3pure({
    systemPrompt: "You are a helpful assistant.",
    temperature: 0.9,
    maxTokens: 256,
    contextSize: 2048,
    type: "load",
    filePath: modelPath,
    filename: "gemma-3-270m-it-Q8_0.gguf",
    cbRender: myRenderFunction,
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

main()
