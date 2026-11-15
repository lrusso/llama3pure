# AGENTS.md

Guide for anyone (human or AI agent) changing this repository: what each part does, how the parts fit together, and the rules every change must meet.

## What this repository is

llama3pure runs Llama 3 and Gemma 3 language models from GGUF files with no external code. It has three ways to run the same inference:

| Target | File | How it runs |
| :-- | :-- | :-- |
| Desktop (C) | `llama3pure-c-engine.c` | Command-line program built with `make llama3pure` |
| Node.js | `llama3pure-js-engine.js` | Imported as a module and called as a function |
| Web | `llama3pure-js-engine.js` | The same file, started as a Web Worker by a page |

The JavaScript engine is one file used both by Node.js and by the browser. The C engine is a separate implementation of the same pipeline.

## Files

| File | Purpose |
| :-- | :-- |
| `llama3pure-js-engine.js` | JavaScript inference engine (Node.js and Web Worker) |
| `llama3pure-c-engine.c` | C inference engine, single source file |
| `Makefile` | Builds the C engine. `make llama3pure` is the build that matters |
| `llama3pure-web-demo.htm` | Web chat client. Loads the engine from `llama3pure-js-engine.js` |
| `llama3pure-web-standalone.htm` | The same client with the engine embedded inside the page |
| `llama3pure-nodejs-demo.js` | Minimal Node.js example |
| `tests.js` | Test runner (all models, or one performance run) |
| `tests.txt` | Sample chat history in JSON, used by the C tests |
| `index.html` | Landing page of the project site |
| `README.md` | User documentation: arguments, parameters, messages, tested models |

GGUF model files are not part of the repository. `tests.js` lists the file names it looks for in the repository folder and skips the ones that are missing.

## The inference pipeline (both engines)

Both engines do the same steps in the same order:

1. **Parse the GGUF file**: header, metadata (architecture, dimensions, tokenizer vocabulary) and the tensor table.
2. **Load the weights**: the large matrices stay quantized, exactly as they are in the file, and are read in place. Only small tensors (normalization weights) are turned into floats. Nothing is dequantized ahead of time, because that would multiply the memory used.
3. **Encode the chat**: the chat history is turned into tokens with the model's chat template. Llama uses header tokens and a tiktoken-style vocabulary; Gemma uses turn tokens and a SentencePiece-style vocabulary, with the system prompt merged into the first user message.
4. **Prefill**: the prompt tokens go through the transformer in batches of up to 32 tokens (`PREFILL_BATCH_SIZE`). Each weight row is dequantized once into a small reusable buffer and applied to several tokens.
5. **Generate**: one token at a time. Each step runs the transformer for a single token with kernels that read the quantized weights directly, computes the logits over the vocabulary and samples the next token (temperature, top-k, top-p).
6. **Stop** on the end-of-sequence or end-of-turn token, when the token limit is reached, or when the output is stuck repeating itself.

Common details:

- **Supported weight formats**: F32, F16, BF16, Q8_0, the block-32 formats (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL) and the K-quants (Q2_K to Q6_K). Each format has its own dequantizer and its own matrix kernels.
- **KV cache**: the keys and values of attention are stored quantized to Q8_0, and the cache grows as the conversation gets longer instead of being reserved for the whole context.
- **Gemma specifics**: sliding-window attention on the local layers, extra normalization steps, GELU activation and soft-capping of the final logits. Llama uses SiLU.
- **Context limit**: when the prompt does not fit in the context (or in the token limit), nothing is computed. The JavaScript engine reports it with a `complete` message whose `output` is `null`; the C engine prints a "Context length exceeded" line.

## JavaScript engine (`llama3pure-js-engine.js`)

One file, no imports, written in ES5. It is organized in sections, in this order: constants, UTF-8 decoding, state, DataView helpers, FP16/BF16 conversion, Q8_0 KV cache, dequantizers, matrix kernels (prefill and single-token, per format), math functions, GGUF parsing, model loading, transformer forward pass, sampling, tokenizer, generation and the message handler.

**Entry point.** The function `llama3pure(data)` receives a message object:

- `{ type: "load", model: ArrayBuffer, ... }` parses the file and prepares the model. Optional fields: `cbRender`, `systemPrompt`, `maxTokens`, `contextSize`, `temperature`, `topP`, `topK`.
- `{ type: "generate", chatHistory: [...] }` encodes the chat and produces a reply.

The same function serves both environments:

- In **Node.js** the file is imported (`module.exports`) and called directly. Tokens arrive through the `cbRender` callback.
- In a **Web Worker** the file sets `self.onmessage` and answers with `postMessage`: `progress`, `loaded`, `token` and `complete`.

The engine has no user-facing text. Error situations are reported as data (for example `output: null` with `promptTokens` and `tokenLimit`), and the client decides what to show.

**Memory.** The model `ArrayBuffer` given by the caller is the only large block. The engine reads the weights through typed-array views over it. Everything else is small and allocated on first use: the vocabulary tables, the prefill buffers (released when the prefill ends), the logits and the generation scratch (released when the reply ends), and the KV cache (grown on demand).

**Kernels.** Almost all the time of a run is spent in the matrix kernels, so they are written for the JIT compilers of the browsers:

- Prefill: a row dequantizer per format fills a small Float64 buffer, and a tile kernel multiplies 4 rows by 3 tokens per pass.
- Single token: fused kernels per format read the quantized weights as 32-bit words and process several rows per pass.

**Between calls.** The KV cache is kept after a reply. On the next `generate`, the positions whose tokens match the start of the new prompt are reused, so a chat does not recompute its whole history every turn. With Q8_0 layer matrices only the positions written by the prefill are reused.

## C engine (`llama3pure-c-engine.c`)

One source file that only needs the C standard library and the math library. Build it with:

```bash
make llama3pure
```

That runs `gcc -O3 -o llama3pure llama3pure-c-engine.c -lm`. On Windows the README shows the `cl` command. The other Makefile targets are optional variants.

**How it runs.** Each execution is one reply: the program reads its arguments, maps the model file into memory, encodes the prompt (`-prompt`) or the JSON chat history (`-chathistory`), runs the prefill, generates the reply to standard output and exits. Nothing is kept between executions, so there is no KV cache reuse between turns. The arguments are listed in `README.md`.

**Structure.** The file follows the same order as the pipeline: GGUF definitions, model structures and memory management, FP16 table, dequantizers, matrix kernels, Q8_0 KV cache, GGUF parsing, weight loading, the transformer blocks, the tokenizer, the chat template and chat-history parsing, and `main`.

**Kernels.** The matrix kernels are written in portable C so the compiler can vectorize them: rows are dequantized in interleaved groups and multiplied by several tokens at once in the prefill, and the last prompt token is computed inside the last prefill batch. No compiler-specific intrinsics headers are used.

After any change to this file, rebuild with `make llama3pure` and test with that binary.

## Web client (`llama3pure-web-demo.htm`)

A single-page chat application in one file, with its styles and script inside, written in ES5.

- It creates a Web Worker from `llama3pure-js-engine.js`, so it must be served by a web server next to that file.
- The model is chosen from disk (read with `FileReader`) or downloaded from a URL in chunks with `XMLHttpRequest` range requests. The `ArrayBuffer` is transferred to the worker, not copied.
- It keeps the chat history, sends it whole with every `generate` message and renders the reply while the `token` messages arrive (Markdown to HTML, code blocks with a copy button).
- A reply can be regenerated, alternative replies can be browsed, and the last prompt can be edited.
- The settings (gear icon) hold the theme and the context size, which can be changed until a model is loaded.
- The interface texts are in English and Spanish, chosen from the browser language. Messages for situations reported by the engine (such as a prompt that exceeds the context) are written here, not in the engine.

## Web standalone client (`llama3pure-web-standalone.htm`)

The same page as the web client with one difference: the Worker is created from a `data:text/javascript;base64,...` URL that contains the whole JavaScript engine. It can be opened as a local file, without a web server.

Two things must always hold:

- Apart from the line that creates the Worker, both `.htm` files are identical. A change to the client goes into both.
- The embedded engine is byte for byte the content of `llama3pure-js-engine.js`. After every change to the engine, encode the file again in base64 and replace the embedded text, then check that decoding it gives the same bytes as the file.

## Testing

```bash
node tests.js        # every model found in the folder, with the C engine and with Node.js
node tests.js perf   # one timed run with gemma-3-1b-it-Q8_0.gguf
```

`node tests.js perf` prints the time to read the file, load the model and generate, the number of tokens, and the process memory at the start, after reading, after loading, after generating and at its peak. Its sampling is random, so the number of generated tokens changes from run to run; compare runs with that in mind.

## Rules for every improvement or fix

These rules have no exceptions. A change that breaks one of them is not acceptable, whatever else it improves.

1. **It must not use more RAM at any point.** Memory is compared at every stage of the process (start, after reading the model, after loading, during and after each reply, and the peak), on every model and in every engine the change touches. A higher number at any single point breaks the rule, even when other points go down.
2. **The inference process must not be slower at any point.** Every stage counts: loading, tokenizing, prefill, first token, time per token, each turn of a chat and the total. A stage that gets slower breaks the rule, even when another stage gets faster.
3. **No libraries or dependencies.** No packages, no imports of external code, no WebAssembly, no scripts from a CDN, no build tools, no third-party C libraries or intrinsics headers. Everything the engines and the clients need is written by hand in this repository.
4. **The JavaScript engine must be pure ES5.** Use `var`, function declarations and expressions, and typed arrays. Do not use `let`, `const`, arrow functions, classes, template literals, destructuring, spread, `Promise`, `async`/`await`, `for...of`, `Map`, `Set` or any other feature added in ES2015 or later. The same applies to the script inside the two `.htm` files.

How to work within the rules:

- Measure before and after with real models, on the unchanged code and on the changed code, and compare stage by stage. Do not estimate.
- Run measurements one process at a time and repeat them, alternating the old and the new code, before concluding that a number went up or down.
- Check that the output did not change unless changing it is the purpose of the fix.
- When a change cannot meet a rule, do not present it as finished: report the numbers and where the rule is broken.
- Keep the two engines consistent when a change applies to both, rebuild the C binary after changing the C engine, and update the engine embedded in the standalone page after changing the JavaScript engine.

## Limit on testing rounds

A testing round is the set of runs an AI agent makes to check one possible improvement.

- **Exactly 5 test runs per model in every round. Always 5: never fewer, never more.** Every model that is tested is run 5 times for the improvement being tried, and then the round is over for that model.
- If the result is not good enough after those 5 runs, the agent does not keep running the same thing. It tries something else (a different change or a different approach), and that new attempt gets its own round of exactly 5 runs per model, and so on.
- The number applies to every model separately and to every round. It is not a total for the whole session.

## Report required at the end of every improvement

When an AI agent finishes a possible improvement, it must end its work with a results table. The table is mandatory, whether the improvement worked or not.

- **One row per GGUF model that was tested.** Every model used in the tests appears, with its file name. Models that were not tested are not listed as if they had been.
- **One column per instance of the process, for RAM and for speed.** The instances are the stages that were measured: start, after reading the model, after loading, prefill, first token, time per token, each chat turn, after generating, peak, and total.
- **No numbers in the cells.** Each cell holds only one of two emojis:
  - ✅ the expected result was reached at that instance.
  - ❌ it was not.
- **What ✅ means in a RAM cell:** at that instance the new code uses the same RAM as before, or less.
- **What ✅ means in a speed cell:** at that instance the new code is as fast as before, or faster.
- Any other outcome is ❌: more RAM at that instance, or slower at that instance. An instance that was not measured is left out of the table; it is never marked ✅.

Example of the layout (the cells are illustrative):

RAM (same or less than before):

| Model | Start | After read | After load | After generate | Peak |
| :-- | :-: | :-: | :-: | :-: | :-: |
| gemma-3-1b-it-Q8_0.gguf | ✅ | ✅ | ✅ | ✅ | ✅ |
| Llama-3.2-1B-Instruct-Q8_0.gguf | ✅ | ✅ | ✅ | ❌ | ❌ |

Speed (same or faster than before):

| Model | Load | Prefill | First token | Time per token | Total |
| :-- | :-: | :-: | :-: | :-: | :-: |
| gemma-3-1b-it-Q8_0.gguf | ✅ | ✅ | ✅ | ✅ | ✅ |
| Llama-3.2-1B-Instruct-Q8_0.gguf | ✅ | ✅ | ✅ | ✅ | ✅ |

A single ❌ anywhere in the table means the improvement breaks the rules above and must not be presented as valid.
