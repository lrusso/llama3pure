import llama3pure from "./llama3pure-nodejs-engine.js"
import fs from "fs"

function myRenderFunction(token) {
  process.stdout.write(token)
}

function main() {
  var buffer = fs.readFileSync("gemma-3-270m-it-Q8_0.gguf")
  var arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  )

  llama3pure({
    systemPrompt: "You are a helpful assistant.",
    temperature: 0.9,
    maxTokens: 256,
    contextSize: 2048,
    type: "load",
    arrayBuffer: arrayBuffer,
    filename: "gemma-3-270m-it-Q8_0.gguf",
    cbRender: myRenderFunction,
  })
  llama3pure({ type: "generate", prompt: "Tell me in 1 line what is Microsoft." })
  process.stdout.write("\n")
}

main()
