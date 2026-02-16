# llama3pure

Three inference engines for Llama 3: pure C for desktop systems, pure JavaScript for Node.js, and pure JavaScript for Web environments. Supports both Llama and Gemma architectures.

![demo](https://github.com/lrusso/llama3pure/blob/main/README.gif?raw=true)

## Table of Contents

- [Build and Run (macOS / Linux)](#build-and-run-macos--linux)
- [Build and Run (Windows)](#build-and-run-windows)
- [Running in Node.js](#running-in-nodejs)
- [Running in Web Environments](#running-in-web-environments)
- [Suggested Models and Engines](#suggested-models-and-engines)
- [Tested Models](#tested-models)
- [Author's Notes](#authors-notes)
- [Credits](#based-on-the-work-of)

## Build and Run (macOS / Linux)

- Build the engine
```
make llama3pure
```

- Run with basic prompt
```
./llama3pure -model Llama3.gguf -prompt "Tell me in 1 line what is Microsoft."`
```

- Run with chat history (sample chat history in [tests.txt](https://github.com/lrusso/llama3pure/blob/main/tests.txt))
```
./llama3pure -model Llama3.gguf -chathistory chat.txt
```

- Run with custom parameters
```
./llama3pure -model Llama3.gguf -temperature 0.9 -top_p 0.9 -top_k 40 -max_tokens 256 -context_size 2048 -system_prompt "You are a helpful assistant." -prompt "Tell me in 1 line what is Microsoft."
```

## Build and Run (Windows)

- Build the engine (use the x64 Native Tools Command Prompt for VS)
```
cl /O2 llama3pure-c-engine.c /Fe:llama3pure.exe
```

- Run with basic prompt
```
llama3pure.exe -model Llama3.gguf -prompt "Tell me in 1 line what is Microsoft."
```

- Run with chat history (sample chat history in [tests.txt](https://github.com/lrusso/llama3pure/blob/main/tests.txt))
```
llama3pure.exe -model Llama3.gguf -chathistory chat.txt
```

- Run with custom parameters
```
llama3pure.exe -model Llama3.gguf -temperature 0.9 -top_p 0.9 -top_k 40 -max_tokens 256 -context_size 2048 -system_prompt "You are a helpful assistant." -prompt "Tell me in 1 line what is Microsoft."
```

## Running in Node.js

```javascript
import llama3pure from "./llama3pure-nodejs-engine.js"
import { fileURLToPath } from "url"
import path from "path"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const modelPath = path.resolve(__dirname, "gemma-3-270m-it-Q8_0.gguf")

const main = () => {
  llama3pure({
    type: "load",
    model: modelPath,
    cbRender: (token) => {
      process.stdout.write(token)
    },
    systemPrompt: "You are a helpful assistant.",
    maxTokens: 256,
    contextSize: 2048,
    temperature: 0.9,
    topP: 0.9,
    topK: 40,
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
}

main()
```

Demo script in [llama3pure-nodejs-demo.js](https://github.com/lrusso/llama3pure/blob/main/llama3pure-nodejs-demo.js).

## Running in Web Environments

Try the Web engine [here](https://lrusso.github.io/llama3pure/llama3pure-web-demo.htm) or with custom `maxTokens`, `contextSize`, `topP` and `topK` [here](https://lrusso.github.io/llama3pure/llama3pure-web-demo.htm?maxTokens=2048&contextSize=4096&topP=0.9&topK=40).

Due to universal browser memory constraints regarding ArrayBuffer size limits, the Web engine can only read GGUF files up to 2 GB.

## Suggested Models and Engines

| MODEL                                                                                                                                                     |  C  | NODE.JS | WEB |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :-: | :-----: | :-: |
| [Gemma-3-1B-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf?download=true)                           | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true) | ✅  |   ✅    | ✅  |
| [Llama-3.2-3B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true) | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q8_0.gguf?download=true)                           | ✅  |   ✅    | ❌  |

## Tested Models

| MODEL                                                                                                                                                             |  C  | NODE.JS | WEB |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-: | :-----: | :-: |
| [Gemma-3-270M-it-Q2_K_L.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q2_K_L.gguf?download=true)                         | ✅  |   ✅    | ✅  |
| [Gemma-3-270M-it-Q3_K_M.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q3_K_M.gguf?download=true)                         | ✅  |   ✅    | ✅  |
| [Gemma-3-270M-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf?download=true)                         | ✅  |   ✅    | ✅  |
| [Gemma-3-270M-it-Q5_K_M.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q5_K_M.gguf?download=true)                         | ✅  |   ✅    | ✅  |
| [Gemma-3-270M-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q6_K.gguf?download=true)                             | ✅  |   ✅    | ✅  |
| [Gemma-3-270M-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf?download=true)                             | ✅  |   ✅    | ✅  |
| [Gemma-3-270M-it-F16.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf?download=true)                               | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-Q2_K_L.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q2_K_L.gguf?download=true)                               | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-Q3_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q3_K_M.gguf?download=true)                               | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf?download=true)                               | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-Q5_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q5_K_M.gguf?download=true)                               | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf?download=true)                                   | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf?download=true)                                   | ✅  |   ✅    | ✅  |
| [Gemma-3-1B-it-BF16.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-BF16.gguf?download=true)                                   | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-Q3_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_L.gguf?download=true)     | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-Q4_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_L.gguf?download=true)     | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-Q5_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_L.gguf?download=true)     | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-Q6_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K_L.gguf?download=true)     | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true)         | ✅  |   ✅    | ✅  |
| [Llama-3.2-1B-Instruct-f16.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf?download=true)           | ✅  |   ✅    | ❌  |
| [Llama-3.2-3B-Instruct-Q3_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_L.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3.2-3B-Instruct-Q4_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_L.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3.2-3B-Instruct-Q5_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_L.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3.2-3B-Instruct-Q6_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3.2-3B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true)         | ✅  |   ✅    | ❌  |
| [Llama-3.2-3B-Instruct-f16.gguf](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf?download=true)           | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q2_K_L.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q2_K_L.gguf?download=true)                               | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q3_K_M.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q3_K_M.gguf?download=true)                               | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf?download=true)                               | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q5_K_M.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q5_K_M.gguf?download=true)                               | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q6_K.gguf?download=true)                                   | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q8_0.gguf?download=true)                                   | ✅  |   ✅    | ❌  |
| [Gemma-3-4b-it-BF16.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-BF16.gguf?download=true)                                   | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-Q2_K.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q2_K.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-Q3_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q3_K_M.gguf?download=true) | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf?download=true) | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-Q5_K_M.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf?download=true) | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-Q6_K.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf?download=true)     | ✅  |   ✅    | ❌  |
| [Llama-3-8B-Instruct-fp16.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-fp16.gguf?download=true)     | ✅  |   ✅    | ❌  |

## Author's Notes

- Using quantizations below Q4 is generally discouraged because the loss in logic and coherence makes them nearly unusable for most tasks.

- There isn't a Python engine because a ported and pure version would be very slow. Using NumPy wouldn't make sense because it uses C under the hood, and for that, there is already a C engine.

## Based on the work of

https://github.com/karpathy/llama2.c
