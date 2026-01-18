import { execSync } from "child_process"

const models = [
  "Gemma-3-270M-it-Q2_K_L.gguf",
  "Gemma-3-270M-it-Q3_K_M.gguf",
  "Gemma-3-270M-it-Q4_K_M.gguf",
  "Gemma-3-270M-it-Q5_K_M.gguf",
  "Gemma-3-270M-it-Q6_K.gguf",
  "Gemma-3-270M-it-Q8_0.gguf",
  "Gemma-3-270M-it-F16.gguf",
  "Gemma-3-1B-it-Q2_K_L.gguf",
  "Gemma-3-1B-it-Q3_K_M.gguf",
  "Gemma-3-1B-it-Q4_K_M.gguf",
  "Gemma-3-1B-it-Q5_K_M.gguf",
  "Gemma-3-1B-it-Q6_K.gguf",
  "Gemma-3-1B-it-Q8_0.gguf",
  "Gemma-3-1B-it-BF16.gguf",
  "Llama-3.2-1B-Instruct-Q3_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q4_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q5_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q6_K_L.gguf",
  "Llama-3.2-1B-Instruct-Q8_0.gguf",
  "Llama-3.2-1B-Instruct-F16.gguf",
]

const prompt = "Tell me in 1 line what is Microsoft."

const startTime = Date.now()

models.forEach(function (model) {
  console.log(model)

  try {
    execSync(`./llama3pure -model "${model}" -prompt "${prompt}"`, {
      stdio: "inherit",
    })
  } catch (error) {
    console.error(error.message)
  }
})

const endTime = Date.now()
const elapsedSeconds = (endTime - startTime) / 1000
console.log("Total elapsed time: " + elapsedSeconds.toFixed(2) + " seconds")
