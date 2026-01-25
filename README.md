# llama3pure

Inference Llama 3 in pure C, in pure JavaScript for Node.js, and in pure JavaScript for Web. Supports both Llama and Gemma architectures.

![algo](https://github.com/lrusso/llama3pure/blob/main/README.gif?raw=true)

## How to build and run the C project:

```bash
# build the project
make llama3pure

# pass a prompt

./llama3pure -model Llama3.gguf -prompt "Tell me in 1 line what is Microsoft."

# pass a prompt with additional parameters

./llama3pure -model Llama3.gguf -temperature 0.9 -max_tokens 256 -context_size 2048 -system_prompt "You are a helpful assistant." -prompt "Tell me in 1 line what is Microsoft."
```

## Web build:

https://lrusso.github.io/llama3pure/llama3pure-web-demo.htm

## Passing custom maxTokens and contextSize values:

https://lrusso.github.io/llama3pure/llama3pure-web-demo.htm?maxTokens=2048&contextSize=4096

## Tested with the following models:

| MODEL  | C ENGINE  | NODE.JS ENGINE | WEB ENGINE
| :------------ |:---------------:| :-----:| :-----:|
| [Gemma-3-270M-it-Q2_K_L.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q2_K_L.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-270M-it-Q3_K_M.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q3_K_M.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-270M-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-270M-it-Q5_K_M.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q5_K_M.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-270M-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q6_K.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-270M-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-270M-it-F16.gguf](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-Q2_K_L.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q2_K_L.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-Q3_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q3_K_M.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-Q5_K_M.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q5_K_M.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-Q8_0.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf?download=true) | ✅ | ✅ | ✅
| [Gemma-3-1B-it-BF16.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-BF16.gguf?download=true) | ✅ | ✅ | ✅
| [Llama-3.2-1B-Instruct-Q3_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_L.gguf?download=true) | ✅ | ✅ | ✅
| [Llama-3.2-1B-Instruct-Q4_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_L.gguf?download=true) | ✅ | ✅ | ✅
| [Llama-3.2-1B-Instruct-Q5_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_L.gguf?download=true) | ✅ | ✅ | ✅
| [Llama-3.2-1B-Instruct-Q6_K_L.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K_L.gguf?download=true) | ✅ | ✅ | ✅
| [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true) | ✅ | ✅ | ✅
| [Llama-3.2-1B-Instruct-F16.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf?download=true) | ✅ | ✅ | ✅

## Disclaimer

You are legally responsible for any damage that you could cause with this software.

## Based on the work of:

https://github.com/karpathy/llama2.c
