import llama3pure from "./llama3pure-nodejs-engine.js"
import { fileURLToPath } from "url"
import path from "path"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const modelName = "gemma-3-270m-it-Q8_0.gguf"
const modelPath = path.resolve(__dirname, modelName)

const main = () => {
  llama3pure({
    systemPrompt: "You are a helpful assistant.",
    maxTokens: 256,
    contextSize: 2048,
    temperature: 0.9,
    topP: 0.9,
    topK: 40,
    type: "load",
    filePath: modelPath,
    filename: modelName,
    cbRender: (token) => {
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

main()
