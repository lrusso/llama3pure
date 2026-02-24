/*
----------------------------------------------------------------------------

Designed by Leonardo Javier Russo
https://www.lrusso.com

JavaScript Engine for LLM Inference - Llama-3 and Gemma-3 Transformer models.
Supports GGUF file format with various quantization types.

----------------------------------------------------------------------------
*/

"use strict"

// ----------------------------------------------------------------------------
// Constants

var GGUF_MAGIC = 0x46554747 // "GGUF" in little-endian

// GGUF value types
var GGUF_TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
}

// GGML tensor types (quantization formats)
var GGML_TYPE = {
  F32: 0,
  F16: 1,
  Q4_0: 2,
  Q4_1: 3,
  Q5_0: 6,
  Q5_1: 7,
  Q8_0: 8,
  Q8_1: 9,
  Q2_K: 10,
  Q3_K: 11,
  Q4_K: 12,
  Q5_K: 13,
  Q6_K: 14,
  Q8_K: 15,
  IQ4_NL: 20,
  BF16: 29,
}

// Block sizes
var QK4_0 = 32
var QK4_1 = 32
var QK5_0 = 32
var QK5_1 = 32
var QK8_0 = 32
var QK_K = 256
var QK4_NL = 32

// IQ4_NL lookup table
var kvalues_iq4nl = [
  -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
]

// ----------------------------------------------------------------------------
// State

var config = null
var weights = null
var state = null
var tokenizer = null
var ggufData = null
var dataView = null
var ggufTextDecoder = new TextDecoder("utf-8")
var offset = 0

// Q8_0 buffers for quantizing x vector in matmulQuantized
var xQ8Buf = null
var xQ8Int8Buf = null
var matmulDeqBuf = null

var temperature = 0.9
var topP = 0.9
var topK = 40
var systemPrompt = "You are a helpful assistant."
var maxTokens = -1
var contextSize = 0

// QuantizedTensor structure: { dataOffset, type, rows, cols }
// Stores metadata to read quantized weights on-the-fly during matmul

// ----------------------------------------------------------------------------
// DataView helpers

function readUint8() {
  var val = dataView.getUint8(offset)
  offset = offset + 1
  return val
}

function readUint16() {
  var val = dataView.getUint16(offset, true)
  offset = offset + 2
  return val
}

function readUint32() {
  var val = dataView.getUint32(offset, true)
  offset = offset + 4
  return val
}

function readUint64() {
  var low = dataView.getUint32(offset, true)
  var high = dataView.getUint32(offset + 4, true)
  offset = offset + 8
  return low + high * 0x100000000
}

function readInt8() {
  var val = dataView.getInt8(offset)
  offset = offset + 1
  return val
}

function readInt32() {
  var val = dataView.getInt32(offset, true)
  offset = offset + 4
  return val
}

function readInt64() {
  var low = dataView.getUint32(offset, true)
  var high = dataView.getInt32(offset + 4, true)
  offset = offset + 8
  return low + high * 0x100000000
}

function readFloat32() {
  var val = dataView.getFloat32(offset, true)
  offset = offset + 4
  return val
}

function readFloat64() {
  var val = dataView.getFloat64(offset, true)
  offset = offset + 8
  return val
}

function readString() {
  var len = readUint64()
  var bytes = ggufUint8
    ? ggufUint8.subarray(offset, offset + len)
    : new Uint8Array(ggufData, offset, len)
  offset = offset + len
  return ggufTextDecoder.decode(bytes)
}

// Cached full-buffer typed array views (initialized on model load)
var ggufUint8 = null
var ggufInt8 = null

// Get a Uint8Array view from the buffer
function getUint8ArrayAt(srcOffset, length) {
  return new Uint8Array(ggufData, srcOffset, length)
}

// Get an Int8Array view from the buffer
function getInt8ArrayAt(srcOffset, length) {
  return new Int8Array(ggufData, srcOffset, length)
}

// Get a Uint16Array view (for F16/BF16)
function getUint16ArrayAt(srcOffset, count) {
  return new Uint16Array(ggufData, srcOffset, count)
}

// Get a Float32Array view
function getFloat32ArrayAt(srcOffset, count) {
  return new Float32Array(ggufData, srcOffset, count)
}

// ----------------------------------------------------------------------------
// FP16/BF16 conversion - optimized with lookup table

// Pre-allocated buffer for float conversion (shared)
var convBuffer = new ArrayBuffer(4)
var convInt = new Uint32Array(convBuffer)
var convFloat = new Float32Array(convBuffer)

// Pre-computed FP16 to FP32 lookup table (256KB)
var fp16Table = new Float32Array(65536)
;(function () {
  for (var h = 0; h < 65536; h = h + 1) {
    var sign = (h & 0x8000) >> 15
    var exp = (h >> 10) & 0x1f
    var mant = h & 0x3ff

    if (exp === 0) {
      if (mant === 0) {
        fp16Table[h] = sign ? -0 : 0
        continue
      }
      // Denormalized
      while (!(mant & 0x400)) {
        mant <<= 1
        exp = exp - 1
      }
      exp = exp + 1
      mant &= ~0x400
    } else if (exp === 31) {
      fp16Table[h] = mant === 0 ? (sign ? -Infinity : Infinity) : NaN
      continue
    }

    exp = exp + (127 - 15)
    mant = mant << 13
    convInt[0] = (sign << 31) | (exp << 23) | mant
    fp16Table[h] = convFloat[0]
  }
})()

// Pre-computed BF16 to FP32 lookup table (256KB)
var bf16Table = new Float32Array(65536)
;(function () {
  for (var h = 0; h < 65536; h = h + 1) {
    convInt[0] = h << 16
    bf16Table[h] = convFloat[0]
  }
})()

function fp16ToFp32(h) {
  return fp16Table[h]
}

function bf16ToFp32(h) {
  return bf16Table[h]
}

// Convert FP32 to FP16 (for Q8_0 scale storage)
function fp32ToFp16(f) {
  convFloat[0] = f
  var bits = convInt[0]
  var sign = (bits >> 16) & 0x8000
  var exp = ((bits >> 23) & 0xff) - 127 + 15
  var mant = (bits >> 13) & 0x3ff

  if (exp <= 0) {
    // Denormalized or zero
    if (exp < -10) {
      // Too small, return signed zero
      return sign
    }
    mant = (mant | 0x400) >> (1 - exp)
    return sign | mant
  } else if (exp >= 31) {
    // Overflow to infinity
    return sign | 0x7c00
  }
  return sign | (exp << 10) | mant
}

// ----------------------------------------------------------------------------
// Q8_0 KV Cache functions
// Q8_0 format: 2 bytes (FP16 scale) + 32 bytes (int8 quants) = 34 bytes per 32 floats

var Q8_0_BLOCK_SIZE = 34 // 2 + 32

// Quantize a float vector to Q8_0 format in cache
// src: Float32Array source, srcOffset: start index in src
// dst: Uint8Array destination cache, dstOffset: byte offset in dst
// count: number of floats (must be multiple of 32)
function quantizeToQ8_0Cache(src, srcOffset, dst, dstInt8, dstOffset, count) {
  var nb = count >> 5 // count / 32
  var bo = dstOffset // byte offset in destination

  for (var i = 0; i < nb; i = i + 1) {
    var bs = srcOffset + (i << 5) // i * 32

    // Find max absolute value in block - unrolled by 8
    var amax = 0.0
    var av0
    var av1
    var av2
    var av3
    var av4
    var av5
    var av6
    var av7

    av0 = src[bs]
    if (av0 < 0) {
      av0 = -av0
    }
    av1 = src[bs + 1]
    if (av1 < 0) {
      av1 = -av1
    }
    av2 = src[bs + 2]
    if (av2 < 0) {
      av2 = -av2
    }
    av3 = src[bs + 3]
    if (av3 < 0) {
      av3 = -av3
    }
    av4 = src[bs + 4]
    if (av4 < 0) {
      av4 = -av4
    }
    av5 = src[bs + 5]
    if (av5 < 0) {
      av5 = -av5
    }
    av6 = src[bs + 6]
    if (av6 < 0) {
      av6 = -av6
    }
    av7 = src[bs + 7]
    if (av7 < 0) {
      av7 = -av7
    }
    if (av0 > amax) {
      amax = av0
    }
    if (av1 > amax) {
      amax = av1
    }
    if (av2 > amax) {
      amax = av2
    }
    if (av3 > amax) {
      amax = av3
    }
    if (av4 > amax) {
      amax = av4
    }
    if (av5 > amax) {
      amax = av5
    }
    if (av6 > amax) {
      amax = av6
    }
    if (av7 > amax) {
      amax = av7
    }

    av0 = src[bs + 8]
    if (av0 < 0) {
      av0 = -av0
    }
    av1 = src[bs + 9]
    if (av1 < 0) {
      av1 = -av1
    }
    av2 = src[bs + 10]
    if (av2 < 0) {
      av2 = -av2
    }
    av3 = src[bs + 11]
    if (av3 < 0) {
      av3 = -av3
    }
    av4 = src[bs + 12]
    if (av4 < 0) {
      av4 = -av4
    }
    av5 = src[bs + 13]
    if (av5 < 0) {
      av5 = -av5
    }
    av6 = src[bs + 14]
    if (av6 < 0) {
      av6 = -av6
    }
    av7 = src[bs + 15]
    if (av7 < 0) {
      av7 = -av7
    }
    if (av0 > amax) {
      amax = av0
    }
    if (av1 > amax) {
      amax = av1
    }
    if (av2 > amax) {
      amax = av2
    }
    if (av3 > amax) {
      amax = av3
    }
    if (av4 > amax) {
      amax = av4
    }
    if (av5 > amax) {
      amax = av5
    }
    if (av6 > amax) {
      amax = av6
    }
    if (av7 > amax) {
      amax = av7
    }

    av0 = src[bs + 16]
    if (av0 < 0) {
      av0 = -av0
    }
    av1 = src[bs + 17]
    if (av1 < 0) {
      av1 = -av1
    }
    av2 = src[bs + 18]
    if (av2 < 0) {
      av2 = -av2
    }
    av3 = src[bs + 19]
    if (av3 < 0) {
      av3 = -av3
    }
    av4 = src[bs + 20]
    if (av4 < 0) {
      av4 = -av4
    }
    av5 = src[bs + 21]
    if (av5 < 0) {
      av5 = -av5
    }
    av6 = src[bs + 22]
    if (av6 < 0) {
      av6 = -av6
    }
    av7 = src[bs + 23]
    if (av7 < 0) {
      av7 = -av7
    }
    if (av0 > amax) {
      amax = av0
    }
    if (av1 > amax) {
      amax = av1
    }
    if (av2 > amax) {
      amax = av2
    }
    if (av3 > amax) {
      amax = av3
    }
    if (av4 > amax) {
      amax = av4
    }
    if (av5 > amax) {
      amax = av5
    }
    if (av6 > amax) {
      amax = av6
    }
    if (av7 > amax) {
      amax = av7
    }

    av0 = src[bs + 24]
    if (av0 < 0) {
      av0 = -av0
    }
    av1 = src[bs + 25]
    if (av1 < 0) {
      av1 = -av1
    }
    av2 = src[bs + 26]
    if (av2 < 0) {
      av2 = -av2
    }
    av3 = src[bs + 27]
    if (av3 < 0) {
      av3 = -av3
    }
    av4 = src[bs + 28]
    if (av4 < 0) {
      av4 = -av4
    }
    av5 = src[bs + 29]
    if (av5 < 0) {
      av5 = -av5
    }
    av6 = src[bs + 30]
    if (av6 < 0) {
      av6 = -av6
    }
    av7 = src[bs + 31]
    if (av7 < 0) {
      av7 = -av7
    }
    if (av0 > amax) {
      amax = av0
    }
    if (av1 > amax) {
      amax = av1
    }
    if (av2 > amax) {
      amax = av2
    }
    if (av3 > amax) {
      amax = av3
    }
    if (av4 > amax) {
      amax = av4
    }
    if (av5 > amax) {
      amax = av5
    }
    if (av6 > amax) {
      amax = av6
    }
    if (av7 > amax) {
      amax = av7
    }

    // Compute scale
    var d = amax / 127.0
    var id = d > 0 ? 127.0 / amax : 0.0

    // Store scale as FP16
    var dFp16 = fp32ToFp16(d)
    dst[bo] = dFp16 & 0xff
    dst[bo + 1] = (dFp16 >> 8) & 0xff

    // Quantize and store values - unrolled by 8
    var qo = bo + 2
    var v0
    var v1
    var v2
    var v3
    var v4
    var v5
    var v6
    var v7

    v0 = src[bs] * id
    v1 = src[bs + 1] * id
    v2 = src[bs + 2] * id
    v3 = src[bs + 3] * id
    v4 = src[bs + 4] * id
    v5 = src[bs + 5] * id
    v6 = src[bs + 6] * id
    v7 = src[bs + 7] * id
    dstInt8[qo] = v0 > 0 ? (v0 + 0.5) | 0 : (v0 - 0.5) | 0
    dstInt8[qo + 1] = v1 > 0 ? (v1 + 0.5) | 0 : (v1 - 0.5) | 0
    dstInt8[qo + 2] = v2 > 0 ? (v2 + 0.5) | 0 : (v2 - 0.5) | 0
    dstInt8[qo + 3] = v3 > 0 ? (v3 + 0.5) | 0 : (v3 - 0.5) | 0
    dstInt8[qo + 4] = v4 > 0 ? (v4 + 0.5) | 0 : (v4 - 0.5) | 0
    dstInt8[qo + 5] = v5 > 0 ? (v5 + 0.5) | 0 : (v5 - 0.5) | 0
    dstInt8[qo + 6] = v6 > 0 ? (v6 + 0.5) | 0 : (v6 - 0.5) | 0
    dstInt8[qo + 7] = v7 > 0 ? (v7 + 0.5) | 0 : (v7 - 0.5) | 0

    v0 = src[bs + 8] * id
    v1 = src[bs + 9] * id
    v2 = src[bs + 10] * id
    v3 = src[bs + 11] * id
    v4 = src[bs + 12] * id
    v5 = src[bs + 13] * id
    v6 = src[bs + 14] * id
    v7 = src[bs + 15] * id
    dstInt8[qo + 8] = v0 > 0 ? (v0 + 0.5) | 0 : (v0 - 0.5) | 0
    dstInt8[qo + 9] = v1 > 0 ? (v1 + 0.5) | 0 : (v1 - 0.5) | 0
    dstInt8[qo + 10] = v2 > 0 ? (v2 + 0.5) | 0 : (v2 - 0.5) | 0
    dstInt8[qo + 11] = v3 > 0 ? (v3 + 0.5) | 0 : (v3 - 0.5) | 0
    dstInt8[qo + 12] = v4 > 0 ? (v4 + 0.5) | 0 : (v4 - 0.5) | 0
    dstInt8[qo + 13] = v5 > 0 ? (v5 + 0.5) | 0 : (v5 - 0.5) | 0
    dstInt8[qo + 14] = v6 > 0 ? (v6 + 0.5) | 0 : (v6 - 0.5) | 0
    dstInt8[qo + 15] = v7 > 0 ? (v7 + 0.5) | 0 : (v7 - 0.5) | 0

    v0 = src[bs + 16] * id
    v1 = src[bs + 17] * id
    v2 = src[bs + 18] * id
    v3 = src[bs + 19] * id
    v4 = src[bs + 20] * id
    v5 = src[bs + 21] * id
    v6 = src[bs + 22] * id
    v7 = src[bs + 23] * id
    dstInt8[qo + 16] = v0 > 0 ? (v0 + 0.5) | 0 : (v0 - 0.5) | 0
    dstInt8[qo + 17] = v1 > 0 ? (v1 + 0.5) | 0 : (v1 - 0.5) | 0
    dstInt8[qo + 18] = v2 > 0 ? (v2 + 0.5) | 0 : (v2 - 0.5) | 0
    dstInt8[qo + 19] = v3 > 0 ? (v3 + 0.5) | 0 : (v3 - 0.5) | 0
    dstInt8[qo + 20] = v4 > 0 ? (v4 + 0.5) | 0 : (v4 - 0.5) | 0
    dstInt8[qo + 21] = v5 > 0 ? (v5 + 0.5) | 0 : (v5 - 0.5) | 0
    dstInt8[qo + 22] = v6 > 0 ? (v6 + 0.5) | 0 : (v6 - 0.5) | 0
    dstInt8[qo + 23] = v7 > 0 ? (v7 + 0.5) | 0 : (v7 - 0.5) | 0

    v0 = src[bs + 24] * id
    v1 = src[bs + 25] * id
    v2 = src[bs + 26] * id
    v3 = src[bs + 27] * id
    v4 = src[bs + 28] * id
    v5 = src[bs + 29] * id
    v6 = src[bs + 30] * id
    v7 = src[bs + 31] * id
    dstInt8[qo + 24] = v0 > 0 ? (v0 + 0.5) | 0 : (v0 - 0.5) | 0
    dstInt8[qo + 25] = v1 > 0 ? (v1 + 0.5) | 0 : (v1 - 0.5) | 0
    dstInt8[qo + 26] = v2 > 0 ? (v2 + 0.5) | 0 : (v2 - 0.5) | 0
    dstInt8[qo + 27] = v3 > 0 ? (v3 + 0.5) | 0 : (v3 - 0.5) | 0
    dstInt8[qo + 28] = v4 > 0 ? (v4 + 0.5) | 0 : (v4 - 0.5) | 0
    dstInt8[qo + 29] = v5 > 0 ? (v5 + 0.5) | 0 : (v5 - 0.5) | 0
    dstInt8[qo + 30] = v6 > 0 ? (v6 + 0.5) | 0 : (v6 - 0.5) | 0
    dstInt8[qo + 31] = v7 > 0 ? (v7 + 0.5) | 0 : (v7 - 0.5) | 0

    bo = bo + Q8_0_BLOCK_SIZE
  }
}

// Compute dot product of float vector with Q8_0 cached vector
// x: Float32Array query vector, xOffset: start index
// Accumulate weighted Q8_0 cached vector to output
// out: Float32Array output, outOffset: start index
// cache: Uint8Array Q8_0 cache, cacheInt8: Int8Array view
// cacheOffset: byte offset in cache
// weight: scalar weight to multiply
// count: number of elements (must be multiple of 32)
function accumQ8_0Cache(
  out,
  outOffset,
  cache,
  cacheInt8,
  cacheOffset,
  weight,
  count
) {
  // Skip near-zero attention weights
  if (weight > -1e-8 && weight < 1e-8) {
    return
  }

  var nb = count >> 5
  var bo = cacheOffset
  var ob = outOffset

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(cache[bo] | (cache[bo + 1] << 8))
    var scale = d * weight
    var qOff = bo + 2

    // Unrolled inner loop for JIT optimization
    out[ob] = out[ob] + cacheInt8[qOff] * scale
    out[ob + 1] = out[ob + 1] + cacheInt8[qOff + 1] * scale
    out[ob + 2] = out[ob + 2] + cacheInt8[qOff + 2] * scale
    out[ob + 3] = out[ob + 3] + cacheInt8[qOff + 3] * scale
    out[ob + 4] = out[ob + 4] + cacheInt8[qOff + 4] * scale
    out[ob + 5] = out[ob + 5] + cacheInt8[qOff + 5] * scale
    out[ob + 6] = out[ob + 6] + cacheInt8[qOff + 6] * scale
    out[ob + 7] = out[ob + 7] + cacheInt8[qOff + 7] * scale
    out[ob + 8] = out[ob + 8] + cacheInt8[qOff + 8] * scale
    out[ob + 9] = out[ob + 9] + cacheInt8[qOff + 9] * scale
    out[ob + 10] = out[ob + 10] + cacheInt8[qOff + 10] * scale
    out[ob + 11] = out[ob + 11] + cacheInt8[qOff + 11] * scale
    out[ob + 12] = out[ob + 12] + cacheInt8[qOff + 12] * scale
    out[ob + 13] = out[ob + 13] + cacheInt8[qOff + 13] * scale
    out[ob + 14] = out[ob + 14] + cacheInt8[qOff + 14] * scale
    out[ob + 15] = out[ob + 15] + cacheInt8[qOff + 15] * scale
    out[ob + 16] = out[ob + 16] + cacheInt8[qOff + 16] * scale
    out[ob + 17] = out[ob + 17] + cacheInt8[qOff + 17] * scale
    out[ob + 18] = out[ob + 18] + cacheInt8[qOff + 18] * scale
    out[ob + 19] = out[ob + 19] + cacheInt8[qOff + 19] * scale
    out[ob + 20] = out[ob + 20] + cacheInt8[qOff + 20] * scale
    out[ob + 21] = out[ob + 21] + cacheInt8[qOff + 21] * scale
    out[ob + 22] = out[ob + 22] + cacheInt8[qOff + 22] * scale
    out[ob + 23] = out[ob + 23] + cacheInt8[qOff + 23] * scale
    out[ob + 24] = out[ob + 24] + cacheInt8[qOff + 24] * scale
    out[ob + 25] = out[ob + 25] + cacheInt8[qOff + 25] * scale
    out[ob + 26] = out[ob + 26] + cacheInt8[qOff + 26] * scale
    out[ob + 27] = out[ob + 27] + cacheInt8[qOff + 27] * scale
    out[ob + 28] = out[ob + 28] + cacheInt8[qOff + 28] * scale
    out[ob + 29] = out[ob + 29] + cacheInt8[qOff + 29] * scale
    out[ob + 30] = out[ob + 30] + cacheInt8[qOff + 30] * scale
    out[ob + 31] = out[ob + 31] + cacheInt8[qOff + 31] * scale

    bo = bo + Q8_0_BLOCK_SIZE
    ob = ob + 32
  }
}

// Compute dot product of two Q8_0 cached vectors (int8 * int8)
// Used for Q8-quantized Q heads against Q8_0 KV cache
function dotQ8_0_Q8_0Cache(aQ8, aI8, aOff, bQ8, bI8, bOff, count) {
  var nb = count >> 5
  var sum = 0.0
  var ao = aOff
  var bo = bOff

  for (var i = 0; i < nb; i = i + 1) {
    var da = fp16ToFp32(aQ8[ao] | (aQ8[ao + 1] << 8))
    var db = fp16ToFp32(bQ8[bo] | (bQ8[bo + 1] << 8))
    var qa = ao + 2
    var qb = bo + 2

    // Two independent 16-term partial sums for superscalar ILP
    var isum1 =
      aI8[qa] * bI8[qb] +
      aI8[qa + 1] * bI8[qb + 1] +
      aI8[qa + 2] * bI8[qb + 2] +
      aI8[qa + 3] * bI8[qb + 3] +
      aI8[qa + 4] * bI8[qb + 4] +
      aI8[qa + 5] * bI8[qb + 5] +
      aI8[qa + 6] * bI8[qb + 6] +
      aI8[qa + 7] * bI8[qb + 7] +
      aI8[qa + 8] * bI8[qb + 8] +
      aI8[qa + 9] * bI8[qb + 9] +
      aI8[qa + 10] * bI8[qb + 10] +
      aI8[qa + 11] * bI8[qb + 11] +
      aI8[qa + 12] * bI8[qb + 12] +
      aI8[qa + 13] * bI8[qb + 13] +
      aI8[qa + 14] * bI8[qb + 14] +
      aI8[qa + 15] * bI8[qb + 15]

    var isum2 =
      aI8[qa + 16] * bI8[qb + 16] +
      aI8[qa + 17] * bI8[qb + 17] +
      aI8[qa + 18] * bI8[qb + 18] +
      aI8[qa + 19] * bI8[qb + 19] +
      aI8[qa + 20] * bI8[qb + 20] +
      aI8[qa + 21] * bI8[qb + 21] +
      aI8[qa + 22] * bI8[qb + 22] +
      aI8[qa + 23] * bI8[qb + 23] +
      aI8[qa + 24] * bI8[qb + 24] +
      aI8[qa + 25] * bI8[qb + 25] +
      aI8[qa + 26] * bI8[qb + 26] +
      aI8[qa + 27] * bI8[qb + 27] +
      aI8[qa + 28] * bI8[qb + 28] +
      aI8[qa + 29] * bI8[qb + 29] +
      aI8[qa + 30] * bI8[qb + 30] +
      aI8[qa + 31] * bI8[qb + 31]

    sum = sum + da * db * (isum1 + isum2)
    ao = ao + Q8_0_BLOCK_SIZE
    bo = bo + Q8_0_BLOCK_SIZE
  }
  return sum
}

// ----------------------------------------------------------------------------
// Dequantization functions

function dequantizeF16(srcOffset, dst, dstOffset, count) {
  var src = getUint16ArrayAt(srcOffset, count)
  for (var i = 0; i < count; i = i + 1) {
    dst[dstOffset + i] = fp16ToFp32(src[i])
  }
}

function dequantizeBF16(srcOffset, dst, dstOffset, count) {
  var src = getUint16ArrayAt(srcOffset, count)
  for (var i = 0; i < count; i = i + 1) {
    dst[dstOffset + i] = bf16ToFp32(src[i])
  }
}

function dequantizeF32(srcOffset, dst, dstOffset, count) {
  var src = getFloat32ArrayAt(srcOffset, count)
  dst.set(src, dstOffset)
}

function dequantizeQ4_0(srcOffset, dst, dstOffset, count) {
  var nb = count >> 5
  var blockSize = 2 + QK4_0 / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))

    for (var j = 0; j < QK4_0 / 2; j = j + 1) {
      var qsByte = src[blockOffset + 2 + j]
      var x0 = (qsByte & 0x0f) - 8
      var x1 = (qsByte >> 4) - 8

      dst[dstOffset + i * QK4_0 + j] = x0 * d
      dst[dstOffset + i * QK4_0 + j + QK4_0 / 2] = x1 * d
    }
  }
}

function dequantizeQ4_1(srcOffset, dst, dstOffset, count) {
  var nb = count >> 5
  var blockSize = 2 + 2 + QK4_1 / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
    var m = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))

    for (var j = 0; j < QK4_1 / 2; j = j + 1) {
      var qsByte = src[blockOffset + 4 + j]
      var x0 = qsByte & 0x0f
      var x1 = qsByte >> 4

      dst[dstOffset + i * QK4_1 + j] = x0 * d + m
      dst[dstOffset + i * QK4_1 + j + QK4_1 / 2] = x1 * d + m
    }
  }
}

function dequantizeQ8_0(srcOffset, dst, dstOffset, count) {
  var nb = count >> 5
  var blockSize = 2 + QK8_0
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)
  var srcSigned = getInt8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))

    for (var j = 0; j < QK8_0; j = j + 1) {
      dst[dstOffset + i * QK8_0 + j] = srcSigned[blockOffset + 2 + j] * d
    }
  }
}

function dequantizeQ5_0(srcOffset, dst, dstOffset, count) {
  var nb = count >> 5
  var blockSize = 2 + 4 + QK5_0 / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
    var qh =
      src[blockOffset + 2] |
      (src[blockOffset + 3] << 8) |
      (src[blockOffset + 4] << 16) |
      (src[blockOffset + 5] << 24)

    for (var j = 0; j < QK5_0 / 2; j = j + 1) {
      var xh_0 = ((qh >> j) & 1) << 4
      var xh_1 = ((qh >> (j + 16)) & 1) << 4

      var qsByte = src[blockOffset + 6 + j]
      var x0 = (qsByte & 0x0f) | xh_0
      var x1 = (qsByte >> 4) | xh_1

      dst[dstOffset + i * QK5_0 + j] = (x0 - 16) * d
      dst[dstOffset + i * QK5_0 + j + QK5_0 / 2] = (x1 - 16) * d
    }
  }
}

function dequantizeQ5_1(srcOffset, dst, dstOffset, count) {
  var nb = count >> 5
  var blockSize = 2 + 2 + 4 + QK5_1 / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
    var m = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))
    var qh =
      src[blockOffset + 4] |
      (src[blockOffset + 5] << 8) |
      (src[blockOffset + 6] << 16) |
      (src[blockOffset + 7] << 24)

    for (var j = 0; j < QK5_1 / 2; j = j + 1) {
      var xh_0 = ((qh >> j) & 1) << 4
      var xh_1 = ((qh >> (j + 16)) & 1) << 4

      var qsByte = src[blockOffset + 8 + j]
      var x0 = (qsByte & 0x0f) | xh_0
      var x1 = (qsByte >> 4) | xh_1

      dst[dstOffset + i * QK5_1 + j] = x0 * d + m
      dst[dstOffset + i * QK5_1 + j + QK5_1 / 2] = x1 * d + m
    }
  }
}

function dequantizeQ2_K(srcOffset, dst, dstOffset, count) {
  var nb = count >> 8
  var blockSize = QK_K / 16 + QK_K / 4 + 2 + 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var scOff = blockOffset
    var qsOff = blockOffset + 16
    var dOffset = blockOffset + 80
    var d = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))
    var dmin = fp16ToFp32(src[dOffset + 2] | (src[dOffset + 3] << 8))

    var y = dstOffset + i * QK_K
    var is = 0
    var qIdx = 0

    for (var n = 0; n < QK_K; n = n + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var sc = src[scOff + is]
        is = is + 1
        var dl = d * (sc & 0xf)
        var ml = dmin * (sc >> 4)
        for (var l = 0; l < 16; l = l + 1) {
          dst[y] = dl * ((src[qsOff + qIdx + l] >> shift) & 3) - ml
          y = y + 1
        }

        sc = src[scOff + is]
        is = is + 1
        dl = d * (sc & 0xf)
        ml = dmin * (sc >> 4)
        for (var l = 0; l < 16; l = l + 1) {
          dst[y] = dl * ((src[qsOff + qIdx + l + 16] >> shift) & 3) - ml
          y = y + 1
        }

        shift = shift + 2
      }
      qIdx = qIdx + 32
    }
  }
}

function dequantizeQ3_K(srcOffset, dst, dstOffset, count) {
  var nb = count >> 8
  var blockSize = QK_K / 8 + QK_K / 4 + 12 + 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  var kmask1 = 0x03030303
  var kmask2 = 0x0f0f0f0f
  var scales = new Int8Array(16)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var hmOff = blockOffset
    var qsOff = blockOffset + 32
    var scRawOff = blockOffset + 96
    var dOffset = blockOffset + 108
    var d_all = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))

    var aux0 =
      src[scRawOff] |
      (src[scRawOff + 1] << 8) |
      (src[scRawOff + 2] << 16) |
      (src[scRawOff + 3] << 24)
    var aux1 =
      src[scRawOff + 4] |
      (src[scRawOff + 5] << 8) |
      (src[scRawOff + 6] << 16) |
      (src[scRawOff + 7] << 24)
    var aux2 =
      src[scRawOff + 8] |
      (src[scRawOff + 9] << 8) |
      (src[scRawOff + 10] << 16) |
      (src[scRawOff + 11] << 24)

    var tmp = aux2
    var s0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    var s1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
    var s2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    var s3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)

    scales[0] = s0 & 0xff
    scales[1] = (s0 >> 8) & 0xff
    scales[2] = (s0 >> 16) & 0xff
    scales[3] = (s0 >> 24) & 0xff
    scales[4] = s1 & 0xff
    scales[5] = (s1 >> 8) & 0xff
    scales[6] = (s1 >> 16) & 0xff
    scales[7] = (s1 >> 24) & 0xff
    scales[8] = s2 & 0xff
    scales[9] = (s2 >> 8) & 0xff
    scales[10] = (s2 >> 16) & 0xff
    scales[11] = (s2 >> 24) & 0xff
    scales[12] = s3 & 0xff
    scales[13] = (s3 >> 8) & 0xff
    scales[14] = (s3 >> 16) & 0xff
    scales[15] = (s3 >> 24) & 0xff

    for (var si = 0; si < 16; si = si + 1) {
      if (scales[si] > 127) {
        scales[si] = scales[si] - 256
      }
    }

    var y = dstOffset + i * QK_K
    var is = 0
    var m = 1
    var qIdx = 0

    for (var n = 0; n < QK_K; n = n + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var dl = d_all * (scales[is] - 32)
        is = is + 1
        for (var l = 0; l < 16; l = l + 1) {
          var q = (src[qsOff + qIdx + l] >> shift) & 3
          var h = src[hmOff + l] & m ? 0 : 4
          dst[y] = dl * (q - h)
          y = y + 1
        }

        dl = d_all * (scales[is] - 32)
        is = is + 1
        for (var l = 0; l < 16; l = l + 1) {
          var q = (src[qsOff + qIdx + l + 16] >> shift) & 3
          var h = src[hmOff + l + 16] & m ? 0 : 4
          dst[y] = dl * (q - h)
          y = y + 1
        }

        shift = shift + 2
        m <<= 1
      }
      qIdx = qIdx + 32
    }
  }
}

function dequantizeQ5_K(srcOffset, dst, dstOffset, count) {
  var nb = count >> 8
  var blockSize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
    var dmin = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))
    var scOff = blockOffset + 4
    var qhOff = blockOffset + 16
    var qlOff = blockOffset + 48

    var y = dstOffset + i * QK_K
    var is = 0
    var u1 = 1
    var u2 = 2
    var qlIdx = 0

    for (var j = 0; j < QK_K; j = j + 64) {
      var sc
      var m
      if (is < 4) {
        sc = src[scOff + is] & 63
        m = src[scOff + is + 4] & 63
      } else {
        sc = (src[scOff + is + 4] & 0xf) | ((src[scOff + is - 4] >> 6) << 4)
        m = (src[scOff + is + 4] >> 4) | ((src[scOff + is] >> 6) << 4)
      }
      var d1 = d * sc
      var m1 = dmin * m

      is = is + 1
      if (is < 4) {
        sc = src[scOff + is] & 63
        m = src[scOff + is + 4] & 63
      } else {
        sc = (src[scOff + is + 4] & 0xf) | ((src[scOff + is - 4] >> 6) << 4)
        m = (src[scOff + is + 4] >> 4) | ((src[scOff + is] >> 6) << 4)
      }
      var d2 = d * sc
      var m2 = dmin * m
      is = is + 1

      for (var l = 0; l < 32; l = l + 1) {
        dst[y + j + l] =
          d1 * ((src[qlOff + qlIdx + l] & 0xf) + (src[qhOff + l] & u1 ? 16 : 0)) - m1
      }
      for (var l = 0; l < 32; l = l + 1) {
        dst[y + j + l + 32] =
          d2 * ((src[qlOff + qlIdx + l] >> 4) + (src[qhOff + l] & u2 ? 16 : 0)) - m2
      }

      qlIdx = qlIdx + 32
      u1 <<= 2
      u2 <<= 2
    }
  }
}

function dequantizeQ4_K(srcOffset, dst, dstOffset, count) {
  var nb = count >> 8
  var blockSize = 2 + 2 + 12 + QK_K / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
    var dmin = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))
    var scOff = blockOffset + 4
    var qsOff = blockOffset + 16

    var is = 0
    var y = dstOffset + i * QK_K

    for (var j = 0; j < QK_K; j = j + 64) {
      var sc
      var m
      if (is < 4) {
        sc = src[scOff + is] & 63
        m = src[scOff + is + 4] & 63
      } else {
        sc = (src[scOff + is + 4] & 0xf) | ((src[scOff + is - 4] >> 6) << 4)
        m = (src[scOff + is + 4] >> 4) | ((src[scOff + is] >> 6) << 4)
      }
      var d1 = d * sc
      var m1 = dmin * m

      is = is + 1
      if (is < 4) {
        sc = src[scOff + is] & 63
        m = src[scOff + is + 4] & 63
      } else {
        sc = (src[scOff + is + 4] & 0xf) | ((src[scOff + is - 4] >> 6) << 4)
        m = (src[scOff + is + 4] >> 4) | ((src[scOff + is] >> 6) << 4)
      }
      var d2 = d * sc
      var m2 = dmin * m
      is = is + 1

      var qIdx = j / 2
      for (var l = 0; l < 32; l = l + 1) {
        var qByte = src[qsOff + qIdx + l]
        dst[y + j + l] = d1 * (qByte & 0xf) - m1
        dst[y + j + l + 32] = d2 * (qByte >> 4) - m2
      }
    }
  }
}

function dequantizeQ6_K(srcOffset, dst, dstOffset, count) {
  var nb = count >> 8
  var blockSize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)
  var srcSigned = getInt8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var qlOff = blockOffset
    var qhOff = blockOffset + 128
    var scalesOffset = blockOffset + 192
    var dOffset = blockOffset + 208
    var d = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))

    var y = dstOffset + i * QK_K

    for (var n = 0; n < QK_K; n = n + 128) {
      var qlBase = qlOff + (n >> 1)
      var qhBase = qhOff + (n >> 2)
      for (var l = 0; l < 32; l = l + 1) {
        var is = l >> 4
        var scBase = scalesOffset + (n >> 7) * 8
        var q1 = ((src[qlBase + l] & 0xf) | (((src[qhBase + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((src[qlBase + l + 32] & 0xf) | (((src[qhBase + l] >> 2) & 3) << 4)) - 32
        var q3 = ((src[qlBase + l] >> 4) | (((src[qhBase + l] >> 4) & 3) << 4)) - 32
        var q4 =
          ((src[qlBase + l + 32] >> 4) | (((src[qhBase + l] >> 6) & 3) << 4)) - 32

        dst[y + n + l] = d * srcSigned[scBase + is + 0] * q1
        dst[y + n + l + 32] = d * srcSigned[scBase + is + 2] * q2
        dst[y + n + l + 64] = d * srcSigned[scBase + is + 4] * q3
        dst[y + n + l + 96] = d * srcSigned[scBase + is + 6] * q4
      }
    }
  }
}

function dequantizeIQ4_NL(srcOffset, dst, dstOffset, count) {
  var nb = count >> 5
  var blockSize = 2 + QK4_NL / 2
  var totalBytes = nb * blockSize
  var src = getUint8ArrayAt(srcOffset, totalBytes)

  for (var i = 0; i < nb; i = i + 1) {
    var blockOffset = i * blockSize
    var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))

    for (var j = 0; j < QK4_NL / 2; j = j + 1) {
      var qsByte = src[blockOffset + 2 + j]
      dst[dstOffset + i * QK4_NL + j] = d * kvalues_iq4nl[qsByte & 0xf]
      dst[dstOffset + i * QK4_NL + j + QK4_NL / 2] = d * kvalues_iq4nl[qsByte >> 4]
    }
  }
}

function dequantizeTensor(srcOffset, count, type) {
  var dst = new Float32Array(count)

  switch (type) {
    case GGML_TYPE.F32:
      dequantizeF32(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.F16:
      dequantizeF16(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.BF16:
    case 30:
      dequantizeBF16(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q4_0:
      dequantizeQ4_0(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q4_1:
      dequantizeQ4_1(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q5_0:
      dequantizeQ5_0(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q5_1:
      dequantizeQ5_1(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q8_0:
      dequantizeQ8_0(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q2_K:
      dequantizeQ2_K(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q3_K:
      dequantizeQ3_K(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q4_K:
      dequantizeQ4_K(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q5_K:
      dequantizeQ5_K(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.Q6_K:
      dequantizeQ6_K(srcOffset, dst, 0, count)
      break
    case GGML_TYPE.IQ4_NL:
      dequantizeIQ4_NL(srcOffset, dst, 0, count)
      break
    default:
      throw new Error("Unsupported quantization type: " + type)
  }

  return dst
}

// ----------------------------------------------------------------------------
// Fused quantized vector-matrix multiplication
// These compute dot products directly from quantized weights without full dequantization

// Get block size for quantization type
function getBlockSize(type) {
  switch (type) {
    case GGML_TYPE.F32:
      return 1
    case GGML_TYPE.F16:
      return 1
    case GGML_TYPE.BF16:
      return 1
    case 30:
      return 1
    case GGML_TYPE.Q4_0:
      return QK4_0
    case GGML_TYPE.Q4_1:
      return QK4_1
    case GGML_TYPE.Q5_0:
      return QK5_0
    case GGML_TYPE.Q5_1:
      return QK5_1
    case GGML_TYPE.Q8_0:
      return QK8_0
    case GGML_TYPE.Q2_K:
      return QK_K
    case GGML_TYPE.Q3_K:
      return QK_K
    case GGML_TYPE.Q4_K:
      return QK_K
    case GGML_TYPE.Q5_K:
      return QK_K
    case GGML_TYPE.Q6_K:
      return QK_K
    case GGML_TYPE.IQ4_NL:
      return QK4_NL
    default:
      return 1
  }
}

// Get bytes per block for quantization type
function getTypeSize(type) {
  switch (type) {
    case GGML_TYPE.F32:
      return 4
    case GGML_TYPE.F16:
      return 2
    case GGML_TYPE.BF16:
      return 2
    case 30:
      return 2
    case GGML_TYPE.Q4_0:
      return 2 + QK4_0 / 2
    case GGML_TYPE.Q4_1:
      return 2 + 2 + QK4_1 / 2
    case GGML_TYPE.Q5_0:
      return 2 + 4 + QK5_0 / 2
    case GGML_TYPE.Q5_1:
      return 2 + 2 + 4 + QK5_1 / 2
    case GGML_TYPE.Q8_0:
      return 2 + QK8_0
    case GGML_TYPE.Q2_K:
      return QK_K / 16 + QK_K / 4 + 2 + 2
    case GGML_TYPE.Q3_K:
      return QK_K / 8 + QK_K / 4 + 12 + 2
    case GGML_TYPE.Q4_K:
      return 2 + 2 + 12 + QK_K / 2
    case GGML_TYPE.Q5_K:
      return 2 + 2 + 12 + QK_K / 8 + QK_K / 2
    case GGML_TYPE.Q6_K:
      return QK_K / 2 + QK_K / 4 + QK_K / 16 + 2
    case GGML_TYPE.IQ4_NL:
      return 2 + QK4_NL / 2
    default:
      return 0
  }
}

// Get row size in bytes
function getRowSize(nCols, type) {
  var blockSize = getBlockSize(type)
  var typeSize = getTypeSize(type)
  return ((nCols / blockSize) | 0) * typeSize
}

// Dequantize a single row from quantized tensor into destination array
// Used for on-demand embedding lookup to avoid storing full dequantized embeddings
function dequantizeRow(dst, srcOffset, nCols, type) {
  switch (type) {
    case GGML_TYPE.F32:
      dequantizeF32(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.F16:
      dequantizeF16(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.BF16:
    case 30:
      dequantizeBF16(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q4_0:
      dequantizeQ4_0(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q4_1:
      dequantizeQ4_1(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q5_0:
      dequantizeQ5_0(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q5_1:
      dequantizeQ5_1(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q8_0:
      dequantizeQ8_0(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q2_K:
      dequantizeQ2_K(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q3_K:
      dequantizeQ3_K(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q4_K:
      dequantizeQ4_K(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q5_K:
      dequantizeQ5_K(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.Q6_K:
      dequantizeQ6_K(srcOffset, dst, 0, nCols)
      break
    case GGML_TYPE.IQ4_NL:
      dequantizeIQ4_NL(srcOffset, dst, 0, nCols)
      break
    default:
      throw new Error("Unsupported embedding type: " + type)
  }
}

// Fused dot product for Q4_0
function vecDotQ4_0(x, srcOffset, n) {
  var nb = n >> 5 // n / 32
  var sum = 0.0
  var bo = srcOffset // block offset in buffer
  var xb = 0 // x offset
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))

    var blockSum = 0.0
    for (var j = 0; j < 16; j = j + 1) {
      var qsByte = u8[bo + 2 + j]
      var x0 = (qsByte & 0x0f) - 8
      var x1 = (qsByte >> 4) - 8
      blockSum = blockSum + x[xb + j] * x0 + x[xb + j + 16] * x1
    }
    sum = sum + blockSum * d
    bo = bo + 18 // 2 + 16
    xb = xb + 32
  }
  return sum
}

// Fused dot product for Q4_1
function vecDotQ4_1(x, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var m = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))

    var blockSum = 0.0
    var xSum = 0.0
    for (var j = 0; j < 16; j = j + 1) {
      var qsByte = u8[bo + 4 + j]
      var x0 = qsByte & 0x0f
      var x1 = qsByte >> 4
      blockSum = blockSum + x[xb + j] * x0 + x[xb + j + 16] * x1
      xSum = xSum + x[xb + j] + x[xb + j + 16]
    }
    sum = sum + blockSum * d + xSum * m
    bo = bo + 20 // 2 + 2 + 16
    xb = xb + 32
  }
  return sum
}

// Fused dot product for Q5_0
function vecDotQ5_0(x, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var qh = u8[bo + 2] | (u8[bo + 3] << 8) | (u8[bo + 4] << 16) | (u8[bo + 5] << 24)

    var blockSum = 0.0
    for (var j = 0; j < 16; j = j + 1) {
      var xh_0 = ((qh >> j) & 1) << 4
      var xh_1 = ((qh >> (j + 16)) & 1) << 4
      var qsByte = u8[bo + 6 + j]
      var x0 = ((qsByte & 0x0f) | xh_0) - 16
      var x1 = ((qsByte >> 4) | xh_1) - 16
      blockSum = blockSum + x[xb + j] * x0 + x[xb + j + 16] * x1
    }
    sum = sum + blockSum * d
    bo = bo + 22 // 2 + 4 + 16
    xb = xb + 32
  }
  return sum
}

// Fused dot product for Q5_1
function vecDotQ5_1(x, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var m = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
    var qh = u8[bo + 4] | (u8[bo + 5] << 8) | (u8[bo + 6] << 16) | (u8[bo + 7] << 24)

    var blockSum = 0.0
    var xSum = 0.0
    for (var j = 0; j < 16; j = j + 1) {
      var xh_0 = ((qh >> j) & 1) << 4
      var xh_1 = ((qh >> (j + 16)) & 1) << 4
      var qsByte = u8[bo + 8 + j]
      var x0 = (qsByte & 0x0f) | xh_0
      var x1 = (qsByte >> 4) | xh_1
      blockSum = blockSum + x[xb + j] * x0 + x[xb + j + 16] * x1
      xSum = xSum + x[xb + j] + x[xb + j + 16]
    }
    sum = sum + blockSum * d + xSum * m
    bo = bo + 24 // 2 + 2 + 4 + 16
    xb = xb + 32
  }
  return sum
}

// Fused dot product for Q8_0 - JIT optimized
function vecDotQ8_0(x, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  // Cache typed array references for JIT
  var u8 = ggufUint8
  var i8 = ggufInt8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var qOff = bo + 2

    // Unrolled inner loop with cached offset
    var blockSum =
      x[xb] * i8[qOff] +
      x[xb + 1] * i8[qOff + 1] +
      x[xb + 2] * i8[qOff + 2] +
      x[xb + 3] * i8[qOff + 3] +
      x[xb + 4] * i8[qOff + 4] +
      x[xb + 5] * i8[qOff + 5] +
      x[xb + 6] * i8[qOff + 6] +
      x[xb + 7] * i8[qOff + 7] +
      x[xb + 8] * i8[qOff + 8] +
      x[xb + 9] * i8[qOff + 9] +
      x[xb + 10] * i8[qOff + 10] +
      x[xb + 11] * i8[qOff + 11] +
      x[xb + 12] * i8[qOff + 12] +
      x[xb + 13] * i8[qOff + 13] +
      x[xb + 14] * i8[qOff + 14] +
      x[xb + 15] * i8[qOff + 15] +
      x[xb + 16] * i8[qOff + 16] +
      x[xb + 17] * i8[qOff + 17] +
      x[xb + 18] * i8[qOff + 18] +
      x[xb + 19] * i8[qOff + 19] +
      x[xb + 20] * i8[qOff + 20] +
      x[xb + 21] * i8[qOff + 21] +
      x[xb + 22] * i8[qOff + 22] +
      x[xb + 23] * i8[qOff + 23] +
      x[xb + 24] * i8[qOff + 24] +
      x[xb + 25] * i8[qOff + 25] +
      x[xb + 26] * i8[qOff + 26] +
      x[xb + 27] * i8[qOff + 27] +
      x[xb + 28] * i8[qOff + 28] +
      x[xb + 29] * i8[qOff + 29] +
      x[xb + 30] * i8[qOff + 30] +
      x[xb + 31] * i8[qOff + 31]

    sum = sum + blockSum * d
    bo = bo + 34
    xb = xb + 32
  }
  return sum
}

// Fused dot product for Q2_K - JIT optimized
function vecDotQ2_K(x, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  // Cache typed array reference for JIT
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var scOff = bo
    var qsOff = bo + 16
    var dOff = bo + 80
    var d = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))
    var dmin = fp16ToFp32(u8[dOff + 2] | (u8[dOff + 3] << 8))

    var is = 0
    var qIdx = 0
    var blockSum = 0.0

    for (var nOuter = 0; nOuter < 256; nOuter = nOuter + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var sc = u8[scOff + is]
        is = is + 1
        var dl = d * (sc & 0xf)
        var ml = dmin * (sc >> 4)
        var baseIdx = xb + nOuter + j * 32
        var qBase = qsOff + qIdx
        for (var l = 0; l < 16; l = l + 1) {
          blockSum =
            blockSum + x[baseIdx + l] * (dl * ((u8[qBase + l] >> shift) & 3) - ml)
        }

        sc = u8[scOff + is]
        is = is + 1
        dl = d * (sc & 0xf)
        ml = dmin * (sc >> 4)
        for (var l = 0; l < 16; l = l + 1) {
          blockSum =
            blockSum +
            x[baseIdx + 16 + l] * (dl * ((u8[qBase + l + 16] >> shift) & 3) - ml)
        }
        shift = shift + 2
      }
      qIdx = qIdx + 32
    }
    sum = sum + blockSum
    bo = bo + 84
    xb = xb + 256
  }
  return sum
}

// Pre-allocated scales array for Q3_K
var q3kScales = new Int8Array(16)

// Fused dot product for Q3_K
function vecDotQ3_K(x, srcOffset, n) {
  var nb = n >> 8
  var kmask1 = 0x03030303
  var kmask2 = 0x0f0f0f0f
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var hmOff = bo
    var qsOff = bo + 32
    var scOff = bo + 96
    var dOff = bo + 108
    var dAll = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))

    var aux0 =
      u8[scOff] |
      (u8[scOff + 1] << 8) |
      (u8[scOff + 2] << 16) |
      (u8[scOff + 3] << 24)
    var aux1 =
      u8[scOff + 4] |
      (u8[scOff + 5] << 8) |
      (u8[scOff + 6] << 16) |
      (u8[scOff + 7] << 24)
    var aux2 =
      u8[scOff + 8] |
      (u8[scOff + 9] << 8) |
      (u8[scOff + 10] << 16) |
      (u8[scOff + 11] << 24)

    var s0 = (aux0 & kmask2) | (((aux2 >> 0) & kmask1) << 4)
    var s1 = (aux1 & kmask2) | (((aux2 >> 2) & kmask1) << 4)
    var s2 = ((aux0 >> 4) & kmask2) | (((aux2 >> 4) & kmask1) << 4)
    var s3 = ((aux1 >> 4) & kmask2) | (((aux2 >> 6) & kmask1) << 4)

    q3kScales[0] = s0 & 0xff
    q3kScales[1] = (s0 >> 8) & 0xff
    q3kScales[2] = (s0 >> 16) & 0xff
    q3kScales[3] = (s0 >> 24) & 0xff
    q3kScales[4] = s1 & 0xff
    q3kScales[5] = (s1 >> 8) & 0xff
    q3kScales[6] = (s1 >> 16) & 0xff
    q3kScales[7] = (s1 >> 24) & 0xff
    q3kScales[8] = s2 & 0xff
    q3kScales[9] = (s2 >> 8) & 0xff
    q3kScales[10] = (s2 >> 16) & 0xff
    q3kScales[11] = (s2 >> 24) & 0xff
    q3kScales[12] = s3 & 0xff
    q3kScales[13] = (s3 >> 8) & 0xff
    q3kScales[14] = (s3 >> 16) & 0xff
    q3kScales[15] = (s3 >> 24) & 0xff

    for (var si = 0; si < 16; si = si + 1) {
      if (q3kScales[si] > 127) {
        q3kScales[si] = q3kScales[si] - 256
      }
    }

    var is = 0
    var m = 1
    var qIdx = 0
    var blockSum = 0.0

    for (var nOuter = 0; nOuter < 256; nOuter = nOuter + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var dl = dAll * (q3kScales[is] - 32)
        is = is + 1
        for (var l = 0; l < 16; l = l + 1) {
          var q = (u8[qsOff + qIdx + l] >> shift) & 3
          var h = u8[hmOff + l] & m ? 0 : 4
          blockSum = blockSum + x[xb + nOuter + j * 32 + l] * dl * (q - h)
        }

        dl = dAll * (q3kScales[is] - 32)
        is = is + 1
        for (var l = 0; l < 16; l = l + 1) {
          var q = (u8[qsOff + qIdx + l + 16] >> shift) & 3
          var h = u8[hmOff + l + 16] & m ? 0 : 4
          blockSum = blockSum + x[xb + nOuter + j * 32 + 16 + l] * dl * (q - h)
        }
        shift = shift + 2
        m <<= 1
      }
      qIdx = qIdx + 32
    }
    sum = sum + blockSum
    bo = bo + 110 // 32 + 64 + 12 + 2
    xb = xb + 256
  }
  return sum
}

// Fused dot product for Q4_K - JIT optimized
function vecDotQ4_K(x, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  // Cache typed array reference for JIT
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var dmin = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
    var scOff = bo + 4
    var qsOff = bo + 16

    var blockSum = 0.0

    // Unrolled: j=0 (is=0,1)
    var sc0 = u8[scOff] & 63
    var m0 = u8[scOff + 4] & 63
    var sc1 = u8[scOff + 1] & 63
    var m1 = u8[scOff + 5] & 63
    var d1 = d * sc0
    var dm1 = dmin * m0
    var d2 = d * sc1
    var dm2 = dmin * m1
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + l]
      blockSum = blockSum + x[xb + l] * (d1 * (qByte & 0xf) - dm1)
      blockSum = blockSum + x[xb + l + 32] * (d2 * (qByte >> 4) - dm2)
    }

    // Unrolled: j=64 (is=2,3)
    sc0 = u8[scOff + 2] & 63
    m0 = u8[scOff + 6] & 63
    sc1 = u8[scOff + 3] & 63
    m1 = u8[scOff + 7] & 63
    d1 = d * sc0
    dm1 = dmin * m0
    d2 = d * sc1
    dm2 = dmin * m1
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + 32 + l]
      blockSum = blockSum + x[xb + 64 + l] * (d1 * (qByte & 0xf) - dm1)
      blockSum = blockSum + x[xb + 64 + l + 32] * (d2 * (qByte >> 4) - dm2)
    }

    // Unrolled: j=128 (is=4,5)
    sc0 = (u8[scOff + 8] & 0xf) | ((u8[scOff] >> 6) << 4)
    m0 = (u8[scOff + 8] >> 4) | ((u8[scOff + 4] >> 6) << 4)
    sc1 = (u8[scOff + 9] & 0xf) | ((u8[scOff + 1] >> 6) << 4)
    m1 = (u8[scOff + 9] >> 4) | ((u8[scOff + 5] >> 6) << 4)
    d1 = d * sc0
    dm1 = dmin * m0
    d2 = d * sc1
    dm2 = dmin * m1
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + 64 + l]
      blockSum = blockSum + x[xb + 128 + l] * (d1 * (qByte & 0xf) - dm1)
      blockSum = blockSum + x[xb + 128 + l + 32] * (d2 * (qByte >> 4) - dm2)
    }

    // Unrolled: j=192 (is=6,7)
    sc0 = (u8[scOff + 10] & 0xf) | ((u8[scOff + 2] >> 6) << 4)
    m0 = (u8[scOff + 10] >> 4) | ((u8[scOff + 6] >> 6) << 4)
    sc1 = (u8[scOff + 11] & 0xf) | ((u8[scOff + 3] >> 6) << 4)
    m1 = (u8[scOff + 11] >> 4) | ((u8[scOff + 7] >> 6) << 4)
    d1 = d * sc0
    dm1 = dmin * m0
    d2 = d * sc1
    dm2 = dmin * m1
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + 96 + l]
      blockSum = blockSum + x[xb + 192 + l] * (d1 * (qByte & 0xf) - dm1)
      blockSum = blockSum + x[xb + 192 + l + 32] * (d2 * (qByte >> 4) - dm2)
    }

    sum = sum + blockSum
    bo = bo + 144
    xb = xb + 256
  }
  return sum
}

// Fused dot product for Q5_K
function vecDotQ5_K(x, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var dmin = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
    var scOff = bo + 4
    var qhOff = bo + 16
    var qlOff = bo + 48

    var is = 0
    var u1 = 1
    var u2 = 2
    var qlIdx = 0
    var blockSum = 0.0

    for (var j = 0; j < 256; j = j + 64) {
      var sc
      var m
      if (is < 4) {
        sc = u8[scOff + is] & 63
        m = u8[scOff + is + 4] & 63
      } else {
        sc = (u8[scOff + is + 4] & 0xf) | ((u8[scOff + is - 4] >> 6) << 4)
        m = (u8[scOff + is + 4] >> 4) | ((u8[scOff + is] >> 6) << 4)
      }
      var d1 = d * sc
      var m1 = dmin * m
      is = is + 1

      if (is < 4) {
        sc = u8[scOff + is] & 63
        m = u8[scOff + is + 4] & 63
      } else {
        sc = (u8[scOff + is + 4] & 0xf) | ((u8[scOff + is - 4] >> 6) << 4)
        m = (u8[scOff + is + 4] >> 4) | ((u8[scOff + is] >> 6) << 4)
      }
      var d2 = d * sc
      var m2 = dmin * m
      is = is + 1

      for (var l = 0; l < 32; l = l + 1) {
        blockSum =
          blockSum +
          x[xb + j + l] *
            (d1 * ((u8[qlOff + qlIdx + l] & 0xf) + (u8[qhOff + l] & u1 ? 16 : 0)) -
              m1)
        blockSum =
          blockSum +
          x[xb + j + l + 32] *
            (d2 * ((u8[qlOff + qlIdx + l] >> 4) + (u8[qhOff + l] & u2 ? 16 : 0)) -
              m2)
      }
      qlIdx = qlIdx + 32
      u1 <<= 2
      u2 <<= 2
    }
    sum = sum + blockSum
    bo = bo + 176 // 2 + 2 + 12 + 32 + 128
    xb = xb + 256
  }
  return sum
}

// Fused dot product for Q6_K
function vecDotQ6_K(x, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8
  var i8 = ggufInt8

  for (var i = 0; i < nb; i = i + 1) {
    var qlOff = bo
    var qhOff = bo + 128
    var scOff = bo + 192
    var dOff = bo + 208
    var d = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))

    var blockSum = 0.0
    for (var nOuter = 0; nOuter < 256; nOuter = nOuter + 128) {
      var scBase = scOff + (nOuter >> 7) * 8
      var qlBase = qlOff + (nOuter >> 1)
      var qhBase = qhOff + (nOuter >> 2)
      // Pre-compute scale multipliers (avoids redundant d * i8[scBase+is] in inner loop)
      var ds0 = d * i8[scBase]
      var ds2 = d * i8[scBase + 2]
      var ds4 = d * i8[scBase + 4]
      var ds6 = d * i8[scBase + 6]
      var ds1 = d * i8[scBase + 1]
      var ds3 = d * i8[scBase + 3]
      var ds5 = d * i8[scBase + 5]
      var ds7 = d * i8[scBase + 7]
      // l = 0..15 (is = 0)
      for (var l = 0; l < 16; l = l + 1) {
        var q1 = ((u8[qlBase + l] & 0xf) | (((u8[qhBase + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((u8[qlBase + l + 32] & 0xf) | (((u8[qhBase + l] >> 2) & 3) << 4)) - 32
        var q3 = ((u8[qlBase + l] >> 4) | (((u8[qhBase + l] >> 4) & 3) << 4)) - 32
        var q4 =
          ((u8[qlBase + l + 32] >> 4) | (((u8[qhBase + l] >> 6) & 3) << 4)) - 32
        blockSum = blockSum + x[xb + nOuter + l] * ds0 * q1
        blockSum = blockSum + x[xb + nOuter + l + 32] * ds2 * q2
        blockSum = blockSum + x[xb + nOuter + l + 64] * ds4 * q3
        blockSum = blockSum + x[xb + nOuter + l + 96] * ds6 * q4
      }
      // l = 16..31 (is = 1)
      for (var l = 16; l < 32; l = l + 1) {
        var q1 = ((u8[qlBase + l] & 0xf) | (((u8[qhBase + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((u8[qlBase + l + 32] & 0xf) | (((u8[qhBase + l] >> 2) & 3) << 4)) - 32
        var q3 = ((u8[qlBase + l] >> 4) | (((u8[qhBase + l] >> 4) & 3) << 4)) - 32
        var q4 =
          ((u8[qlBase + l + 32] >> 4) | (((u8[qhBase + l] >> 6) & 3) << 4)) - 32
        blockSum = blockSum + x[xb + nOuter + l] * ds1 * q1
        blockSum = blockSum + x[xb + nOuter + l + 32] * ds3 * q2
        blockSum = blockSum + x[xb + nOuter + l + 64] * ds5 * q3
        blockSum = blockSum + x[xb + nOuter + l + 96] * ds7 * q4
      }
    }
    sum = sum + blockSum
    bo = bo + 210 // 128 + 64 + 16 + 2
    xb = xb + 256
  }
  return sum
}

// Fused dot product for IQ4_NL
function vecDotIQ4_NL(x, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var bo = srcOffset
  var xb = 0
  var u8 = ggufUint8

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))

    var blockSum = 0.0
    for (var j = 0; j < 16; j = j + 1) {
      var qsByte = u8[bo + 2 + j]
      blockSum = blockSum + x[xb + j] * kvalues_iq4nl[qsByte & 0xf]
      blockSum = blockSum + x[xb + j + 16] * kvalues_iq4nl[qsByte >> 4]
    }
    sum = sum + blockSum * d
    bo = bo + 18 // 2 + 16
    xb = xb + 32
  }
  return sum
}

// Fused dot product for F16 - unrolled by 8
function vecDotF16(x, srcOffset, n) {
  var sum = 0.0
  var bo = srcOffset
  var u8 = ggufUint8
  var n8 = n & ~7
  var i = 0
  for (; i < n8; i = i + 8) {
    sum =
      sum +
      x[i] * fp16Table[u8[bo] | (u8[bo + 1] << 8)] +
      x[i + 1] * fp16Table[u8[bo + 2] | (u8[bo + 3] << 8)] +
      x[i + 2] * fp16Table[u8[bo + 4] | (u8[bo + 5] << 8)] +
      x[i + 3] * fp16Table[u8[bo + 6] | (u8[bo + 7] << 8)] +
      x[i + 4] * fp16Table[u8[bo + 8] | (u8[bo + 9] << 8)] +
      x[i + 5] * fp16Table[u8[bo + 10] | (u8[bo + 11] << 8)] +
      x[i + 6] * fp16Table[u8[bo + 12] | (u8[bo + 13] << 8)] +
      x[i + 7] * fp16Table[u8[bo + 14] | (u8[bo + 15] << 8)]
    bo = bo + 16
  }
  for (; i < n; i = i + 1) {
    sum = sum + x[i] * fp16Table[u8[bo] | (u8[bo + 1] << 8)]
    bo = bo + 2
  }
  return sum
}

// Fused dot product for BF16 - unrolled by 8
function vecDotBF16(x, srcOffset, n) {
  var sum = 0.0
  var bo = srcOffset
  var u8 = ggufUint8
  var n8 = n & ~7
  var i = 0
  for (; i < n8; i = i + 8) {
    sum =
      sum +
      x[i] * bf16Table[u8[bo] | (u8[bo + 1] << 8)] +
      x[i + 1] * bf16Table[u8[bo + 2] | (u8[bo + 3] << 8)] +
      x[i + 2] * bf16Table[u8[bo + 4] | (u8[bo + 5] << 8)] +
      x[i + 3] * bf16Table[u8[bo + 6] | (u8[bo + 7] << 8)] +
      x[i + 4] * bf16Table[u8[bo + 8] | (u8[bo + 9] << 8)] +
      x[i + 5] * bf16Table[u8[bo + 10] | (u8[bo + 11] << 8)] +
      x[i + 6] * bf16Table[u8[bo + 12] | (u8[bo + 13] << 8)] +
      x[i + 7] * bf16Table[u8[bo + 14] | (u8[bo + 15] << 8)]
    bo = bo + 16
  }
  for (; i < n; i = i + 1) {
    sum = sum + x[i] * bf16Table[u8[bo] | (u8[bo + 1] << 8)]
    bo = bo + 2
  }
  return sum
}

// Fused dot product for F32
function vecDotF32(x, srcOffset, n) {
  var sum = 0.0
  var bo = srcOffset
  for (var i = 0; i < n; i = i + 1) {
    sum = sum + x[i] * dataView.getFloat32(bo, true)
    bo = bo + 4
  }
  return sum
}

// Fused quantized matrix-vector multiplication
// Computes out = W @ x where W is quantized (rows x cols)
// Get vec_dot function for a type (avoids switch in hot loop)
function getVecDotFunc(type) {
  switch (type) {
    case GGML_TYPE.Q4_0:
      return vecDotQ4_0
    case GGML_TYPE.Q4_1:
      return vecDotQ4_1
    case GGML_TYPE.Q5_0:
      return vecDotQ5_0
    case GGML_TYPE.Q5_1:
      return vecDotQ5_1
    case GGML_TYPE.Q8_0:
      return vecDotQ8_0
    case GGML_TYPE.Q2_K:
      return vecDotQ2_K
    case GGML_TYPE.Q3_K:
      return vecDotQ3_K
    case GGML_TYPE.Q4_K:
      return vecDotQ4_K
    case GGML_TYPE.Q5_K:
      return vecDotQ5_K
    case GGML_TYPE.Q6_K:
      return vecDotQ6_K
    case GGML_TYPE.IQ4_NL:
      return vecDotIQ4_NL
    case GGML_TYPE.F16:
      return vecDotF16
    case GGML_TYPE.BF16:
    case 30:
      return vecDotBF16
    case GGML_TYPE.F32:
      return vecDotF32
    default:
      return null
  }
}

// ----------------------------------------------------------------------------
// Q8_0-input vec_dot functions (integer inner loops)
// These take Q8_0-quantized x instead of float x for faster dot products.
// Signature: (xQ8, xQ8i8, srcOffset, n) where xQ8 is Uint8Array, xQ8i8 is Int8Array view

function vecDotQ4_0_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = wOff + 2
    var qx = xOff + 2
    var isum = 0
    for (var j = 0; j < 16; j = j + 1) {
      var qByte = u8[qw + j]
      isum =
        isum +
        xQ8i8[qx + j] * ((qByte & 0x0f) - 8) +
        xQ8i8[qx + j + 16] * ((qByte >> 4) - 8)
    }
    sum = sum + dw * dx * isum
    wOff = wOff + 18
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ4_1_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var mw = fp16ToFp32(u8[wOff + 2] | (u8[wOff + 3] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = wOff + 4
    var qx = xOff + 2
    var isum = 0
    var xsum = 0
    for (var j = 0; j < 16; j = j + 1) {
      var qByte = u8[qw + j]
      isum =
        isum + xQ8i8[qx + j] * (qByte & 0x0f) + xQ8i8[qx + j + 16] * (qByte >> 4)
      xsum = xsum + xQ8i8[qx + j] + xQ8i8[qx + j + 16]
    }
    sum = sum + dw * dx * isum + mw * dx * xsum
    wOff = wOff + 20
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ5_0_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var qh =
      u8[wOff + 2] |
      (u8[wOff + 3] << 8) |
      (u8[wOff + 4] << 16) |
      (u8[wOff + 5] << 24)
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = wOff + 6
    var qx = xOff + 2
    var isum = 0
    for (var j = 0; j < 16; j = j + 1) {
      var xh_0 = ((qh >> j) & 1) << 4
      var xh_1 = ((qh >> (j + 16)) & 1) << 4
      var qByte = u8[qw + j]
      isum =
        isum +
        xQ8i8[qx + j] * (((qByte & 0x0f) | xh_0) - 16) +
        xQ8i8[qx + j + 16] * (((qByte >> 4) | xh_1) - 16)
    }
    sum = sum + dw * dx * isum
    wOff = wOff + 22
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ5_1_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var mw = fp16ToFp32(u8[wOff + 2] | (u8[wOff + 3] << 8))
    var qh =
      u8[wOff + 4] |
      (u8[wOff + 5] << 8) |
      (u8[wOff + 6] << 16) |
      (u8[wOff + 7] << 24)
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = wOff + 8
    var qx = xOff + 2
    var isum = 0
    var xsum = 0
    for (var j = 0; j < 16; j = j + 1) {
      var xh_0 = ((qh >> j) & 1) << 4
      var xh_1 = ((qh >> (j + 16)) & 1) << 4
      var qByte = u8[qw + j]
      isum =
        isum +
        xQ8i8[qx + j] * ((qByte & 0x0f) | xh_0) +
        xQ8i8[qx + j + 16] * ((qByte >> 4) | xh_1)
      xsum = xsum + xQ8i8[qx + j] + xQ8i8[qx + j + 16]
    }
    sum = sum + dw * dx * isum + mw * dx * xsum
    wOff = wOff + 24
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ2_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var scOff = wOff
    var qsOff = wOff + 16
    var dOff = wOff + 80
    var d = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))
    var dmin = fp16ToFp32(u8[dOff + 2] | (u8[dOff + 3] << 8))
    var is = 0
    var qIdx = 0
    var blockSum = 0.0
    for (var nOuter = 0; nOuter < 256; nOuter = nOuter + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
        var qxBase = xOff + 2
        var qBase = qsOff + qIdx
        var sc = u8[scOff + is]
        is = is + 1
        var dl = d * (sc & 0xf)
        var ml = dmin * (sc >> 4)
        var isum1 = 0
        var xsum1 = 0
        for (var l = 0; l < 16; l = l + 1) {
          var qx = xQ8i8[qxBase + l]
          isum1 = isum1 + qx * ((u8[qBase + l] >> shift) & 3)
          xsum1 = xsum1 + qx
        }
        sc = u8[scOff + is]
        is = is + 1
        var dl2 = d * (sc & 0xf)
        var ml2 = dmin * (sc >> 4)
        var isum2 = 0
        var xsum2 = 0
        for (var l = 0; l < 16; l = l + 1) {
          var qx = xQ8i8[qxBase + 16 + l]
          isum2 = isum2 + qx * ((u8[qBase + l + 16] >> shift) & 3)
          xsum2 = xsum2 + qx
        }
        blockSum =
          blockSum + dx * (dl * isum1 - ml * xsum1 + dl2 * isum2 - ml2 * xsum2)
        shift = shift + 2
        xOff = xOff + 34
      }
      qIdx = qIdx + 32
    }
    sum = sum + blockSum
    wOff = wOff + 84
  }
  return sum
}

function vecDotQ3_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var kmask1 = 0x03030303
  var kmask2 = 0x0f0f0f0f
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var hmOff = wOff
    var qsOff = wOff + 32
    var scOff = wOff + 96
    var dOff = wOff + 108
    var dAll = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))
    var aux0 =
      u8[scOff] |
      (u8[scOff + 1] << 8) |
      (u8[scOff + 2] << 16) |
      (u8[scOff + 3] << 24)
    var aux1 =
      u8[scOff + 4] |
      (u8[scOff + 5] << 8) |
      (u8[scOff + 6] << 16) |
      (u8[scOff + 7] << 24)
    var aux2 =
      u8[scOff + 8] |
      (u8[scOff + 9] << 8) |
      (u8[scOff + 10] << 16) |
      (u8[scOff + 11] << 24)
    var s0 = (aux0 & kmask2) | (((aux2 >> 0) & kmask1) << 4)
    var s1 = (aux1 & kmask2) | (((aux2 >> 2) & kmask1) << 4)
    var s2 = ((aux0 >> 4) & kmask2) | (((aux2 >> 4) & kmask1) << 4)
    var s3 = ((aux1 >> 4) & kmask2) | (((aux2 >> 6) & kmask1) << 4)
    q3kScales[0] = s0 & 0xff
    q3kScales[1] = (s0 >> 8) & 0xff
    q3kScales[2] = (s0 >> 16) & 0xff
    q3kScales[3] = (s0 >> 24) & 0xff
    q3kScales[4] = s1 & 0xff
    q3kScales[5] = (s1 >> 8) & 0xff
    q3kScales[6] = (s1 >> 16) & 0xff
    q3kScales[7] = (s1 >> 24) & 0xff
    q3kScales[8] = s2 & 0xff
    q3kScales[9] = (s2 >> 8) & 0xff
    q3kScales[10] = (s2 >> 16) & 0xff
    q3kScales[11] = (s2 >> 24) & 0xff
    q3kScales[12] = s3 & 0xff
    q3kScales[13] = (s3 >> 8) & 0xff
    q3kScales[14] = (s3 >> 16) & 0xff
    q3kScales[15] = (s3 >> 24) & 0xff
    for (var si = 0; si < 16; si = si + 1) {
      if (q3kScales[si] > 127) {
        q3kScales[si] = q3kScales[si] - 256
      }
    }
    var is = 0
    var m = 1
    var qIdx = 0
    var blockSum = 0.0
    for (var nOuter = 0; nOuter < 256; nOuter = nOuter + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
        var qxBase = xOff + 2
        var dl = dAll * (q3kScales[is] - 32)
        is = is + 1
        var isum1 = 0
        for (var l = 0; l < 16; l = l + 1) {
          var q = (u8[qsOff + qIdx + l] >> shift) & 3
          var h = u8[hmOff + l] & m ? 0 : 4
          isum1 = isum1 + xQ8i8[qxBase + l] * (q - h)
        }
        var dl2 = dAll * (q3kScales[is] - 32)
        is = is + 1
        var isum2 = 0
        for (var l = 0; l < 16; l = l + 1) {
          var q = (u8[qsOff + qIdx + l + 16] >> shift) & 3
          var h = u8[hmOff + l + 16] & m ? 0 : 4
          isum2 = isum2 + xQ8i8[qxBase + 16 + l] * (q - h)
        }
        blockSum = blockSum + dx * (dl * isum1 + dl2 * isum2)
        shift = shift + 2
        m = m << 1
        xOff = xOff + 34
      }
      qIdx = qIdx + 32
    }
    sum = sum + blockSum
    wOff = wOff + 110
  }
  return sum
}

function vecDotQ4_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var dmin = fp16ToFp32(u8[wOff + 2] | (u8[wOff + 3] << 8))
    var scOff = wOff + 4
    var qsOff = wOff + 16
    var blockSum = 0.0

    // Section 0 (is=0,1)
    var sc0 = u8[scOff] & 63
    var m0 = u8[scOff + 4] & 63
    var sc1 = u8[scOff + 1] & 63
    var m1 = u8[scOff + 5] & 63
    var d1 = d * sc0
    var dm1 = dmin * m0
    var d2 = d * sc1
    var dm2 = dmin * m1
    var dx0 = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qx0 = xOff + 2
    var dx1 = fp16ToFp32(xQ8[xOff + 34] | (xQ8[xOff + 35] << 8))
    var qx1 = xOff + 36
    var isum0 = 0
    var xsum0 = 0
    var isum1 = 0
    var xsum1 = 0
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + l]
      var xv0 = xQ8i8[qx0 + l]
      var xv1 = xQ8i8[qx1 + l]
      isum0 = isum0 + xv0 * (qByte & 0xf)
      xsum0 = xsum0 + xv0
      isum1 = isum1 + xv1 * (qByte >> 4)
      xsum1 = xsum1 + xv1
    }
    blockSum =
      blockSum +
      d1 * dx0 * isum0 -
      dm1 * dx0 * xsum0 +
      d2 * dx1 * isum1 -
      dm2 * dx1 * xsum1

    // Section 1 (is=2,3)
    sc0 = u8[scOff + 2] & 63
    m0 = u8[scOff + 6] & 63
    sc1 = u8[scOff + 3] & 63
    m1 = u8[scOff + 7] & 63
    d1 = d * sc0
    dm1 = dmin * m0
    d2 = d * sc1
    dm2 = dmin * m1
    dx0 = fp16ToFp32(xQ8[xOff + 68] | (xQ8[xOff + 69] << 8))
    qx0 = xOff + 70
    dx1 = fp16ToFp32(xQ8[xOff + 102] | (xQ8[xOff + 103] << 8))
    qx1 = xOff + 104
    isum0 = 0
    xsum0 = 0
    isum1 = 0
    xsum1 = 0
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + 32 + l]
      var xv0 = xQ8i8[qx0 + l]
      var xv1 = xQ8i8[qx1 + l]
      isum0 = isum0 + xv0 * (qByte & 0xf)
      xsum0 = xsum0 + xv0
      isum1 = isum1 + xv1 * (qByte >> 4)
      xsum1 = xsum1 + xv1
    }
    blockSum =
      blockSum +
      d1 * dx0 * isum0 -
      dm1 * dx0 * xsum0 +
      d2 * dx1 * isum1 -
      dm2 * dx1 * xsum1

    // Section 2 (is=4,5)
    sc0 = (u8[scOff + 8] & 0xf) | ((u8[scOff] >> 6) << 4)
    m0 = (u8[scOff + 8] >> 4) | ((u8[scOff + 4] >> 6) << 4)
    sc1 = (u8[scOff + 9] & 0xf) | ((u8[scOff + 1] >> 6) << 4)
    m1 = (u8[scOff + 9] >> 4) | ((u8[scOff + 5] >> 6) << 4)
    d1 = d * sc0
    dm1 = dmin * m0
    d2 = d * sc1
    dm2 = dmin * m1
    dx0 = fp16ToFp32(xQ8[xOff + 136] | (xQ8[xOff + 137] << 8))
    qx0 = xOff + 138
    dx1 = fp16ToFp32(xQ8[xOff + 170] | (xQ8[xOff + 171] << 8))
    qx1 = xOff + 172
    isum0 = 0
    xsum0 = 0
    isum1 = 0
    xsum1 = 0
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + 64 + l]
      var xv0 = xQ8i8[qx0 + l]
      var xv1 = xQ8i8[qx1 + l]
      isum0 = isum0 + xv0 * (qByte & 0xf)
      xsum0 = xsum0 + xv0
      isum1 = isum1 + xv1 * (qByte >> 4)
      xsum1 = xsum1 + xv1
    }
    blockSum =
      blockSum +
      d1 * dx0 * isum0 -
      dm1 * dx0 * xsum0 +
      d2 * dx1 * isum1 -
      dm2 * dx1 * xsum1

    // Section 3 (is=6,7)
    sc0 = (u8[scOff + 10] & 0xf) | ((u8[scOff + 2] >> 6) << 4)
    m0 = (u8[scOff + 10] >> 4) | ((u8[scOff + 6] >> 6) << 4)
    sc1 = (u8[scOff + 11] & 0xf) | ((u8[scOff + 3] >> 6) << 4)
    m1 = (u8[scOff + 11] >> 4) | ((u8[scOff + 7] >> 6) << 4)
    d1 = d * sc0
    dm1 = dmin * m0
    d2 = d * sc1
    dm2 = dmin * m1
    dx0 = fp16ToFp32(xQ8[xOff + 204] | (xQ8[xOff + 205] << 8))
    qx0 = xOff + 206
    dx1 = fp16ToFp32(xQ8[xOff + 238] | (xQ8[xOff + 239] << 8))
    qx1 = xOff + 240
    isum0 = 0
    xsum0 = 0
    isum1 = 0
    xsum1 = 0
    for (var l = 0; l < 32; l = l + 1) {
      var qByte = u8[qsOff + 96 + l]
      var xv0 = xQ8i8[qx0 + l]
      var xv1 = xQ8i8[qx1 + l]
      isum0 = isum0 + xv0 * (qByte & 0xf)
      xsum0 = xsum0 + xv0
      isum1 = isum1 + xv1 * (qByte >> 4)
      xsum1 = xsum1 + xv1
    }
    blockSum =
      blockSum +
      d1 * dx0 * isum0 -
      dm1 * dx0 * xsum0 +
      d2 * dx1 * isum1 -
      dm2 * dx1 * xsum1

    sum = sum + blockSum
    wOff = wOff + 144
    xOff = xOff + 272
  }
  return sum
}

function vecDotQ5_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var dmin = fp16ToFp32(u8[wOff + 2] | (u8[wOff + 3] << 8))
    var scOff = wOff + 4
    var qhOff = wOff + 16
    var qlOff = wOff + 48
    var is = 0
    var u1 = 1
    var u2 = 2
    var qlIdx = 0
    var blockSum = 0.0
    for (var j = 0; j < 256; j = j + 64) {
      var sc
      var m
      if (is < 4) {
        sc = u8[scOff + is] & 63
        m = u8[scOff + is + 4] & 63
      } else {
        sc = (u8[scOff + is + 4] & 0xf) | ((u8[scOff + is - 4] >> 6) << 4)
        m = (u8[scOff + is + 4] >> 4) | ((u8[scOff + is] >> 6) << 4)
      }
      var d1 = d * sc
      var m1 = dmin * m
      is = is + 1
      if (is < 4) {
        sc = u8[scOff + is] & 63
        m = u8[scOff + is + 4] & 63
      } else {
        sc = (u8[scOff + is + 4] & 0xf) | ((u8[scOff + is - 4] >> 6) << 4)
        m = (u8[scOff + is + 4] >> 4) | ((u8[scOff + is] >> 6) << 4)
      }
      var d2 = d * sc
      var m2 = dmin * m
      is = is + 1
      var dx0 = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
      var qx0 = xOff + 2
      var dx1 = fp16ToFp32(xQ8[xOff + 34] | (xQ8[xOff + 35] << 8))
      var qx1 = xOff + 36
      var isum0 = 0
      var xsum0 = 0
      var isum1 = 0
      var xsum1 = 0
      for (var l = 0; l < 32; l = l + 1) {
        var qw_lo = (u8[qlOff + qlIdx + l] & 0xf) + (u8[qhOff + l] & u1 ? 16 : 0)
        var qw_hi = (u8[qlOff + qlIdx + l] >> 4) + (u8[qhOff + l] & u2 ? 16 : 0)
        var xv0 = xQ8i8[qx0 + l]
        var xv1 = xQ8i8[qx1 + l]
        isum0 = isum0 + xv0 * qw_lo
        xsum0 = xsum0 + xv0
        isum1 = isum1 + xv1 * qw_hi
        xsum1 = xsum1 + xv1
      }
      blockSum =
        blockSum +
        d1 * dx0 * isum0 -
        m1 * dx0 * xsum0 +
        d2 * dx1 * isum1 -
        m2 * dx1 * xsum1
      qlIdx = qlIdx + 32
      u1 = u1 << 2
      u2 = u2 << 2
      xOff = xOff + 68
    }
    sum = sum + blockSum
    wOff = wOff + 176
  }
  return sum
}

function vecDotQ6_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  var i8 = ggufInt8
  for (var i = 0; i < nb; i = i + 1) {
    var qlOff = wOff
    var qhOff = wOff + 128
    var scOff = wOff + 192
    var dOff = wOff + 208
    var d = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))
    var blockSum = 0.0
    for (var nOuter = 0; nOuter < 256; nOuter = nOuter + 128) {
      var scBase = scOff + (nOuter >> 7) * 8
      var qlBase = qlOff + (nOuter >> 1)
      var qhBase = qhOff + (nOuter >> 2)
      var dx0 = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
      var qx0 = xOff + 2
      var dx1 = fp16ToFp32(xQ8[xOff + 34] | (xQ8[xOff + 35] << 8))
      var qx1 = xOff + 36
      var dx2 = fp16ToFp32(xQ8[xOff + 68] | (xQ8[xOff + 69] << 8))
      var qx2 = xOff + 70
      var dx3 = fp16ToFp32(xQ8[xOff + 102] | (xQ8[xOff + 103] << 8))
      var qx3 = xOff + 104
      var ds0 = d * i8[scBase]
      var ds2 = d * i8[scBase + 2]
      var ds4 = d * i8[scBase + 4]
      var ds6 = d * i8[scBase + 6]
      var ds1 = d * i8[scBase + 1]
      var ds3 = d * i8[scBase + 3]
      var ds5 = d * i8[scBase + 5]
      var ds7 = d * i8[scBase + 7]
      var isum0a = 0
      var isum1a = 0
      var isum2a = 0
      var isum3a = 0
      for (var l = 0; l < 16; l = l + 1) {
        var q1 = ((u8[qlBase + l] & 0xf) | (((u8[qhBase + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((u8[qlBase + l + 32] & 0xf) | (((u8[qhBase + l] >> 2) & 3) << 4)) - 32
        var q3 = ((u8[qlBase + l] >> 4) | (((u8[qhBase + l] >> 4) & 3) << 4)) - 32
        var q4 =
          ((u8[qlBase + l + 32] >> 4) | (((u8[qhBase + l] >> 6) & 3) << 4)) - 32
        isum0a = isum0a + xQ8i8[qx0 + l] * q1
        isum1a = isum1a + xQ8i8[qx1 + l] * q2
        isum2a = isum2a + xQ8i8[qx2 + l] * q3
        isum3a = isum3a + xQ8i8[qx3 + l] * q4
      }
      var isum0b = 0
      var isum1b = 0
      var isum2b = 0
      var isum3b = 0
      for (var l = 16; l < 32; l = l + 1) {
        var q1 = ((u8[qlBase + l] & 0xf) | (((u8[qhBase + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((u8[qlBase + l + 32] & 0xf) | (((u8[qhBase + l] >> 2) & 3) << 4)) - 32
        var q3 = ((u8[qlBase + l] >> 4) | (((u8[qhBase + l] >> 4) & 3) << 4)) - 32
        var q4 =
          ((u8[qlBase + l + 32] >> 4) | (((u8[qhBase + l] >> 6) & 3) << 4)) - 32
        isum0b = isum0b + xQ8i8[qx0 + l] * q1
        isum1b = isum1b + xQ8i8[qx1 + l] * q2
        isum2b = isum2b + xQ8i8[qx2 + l] * q3
        isum3b = isum3b + xQ8i8[qx3 + l] * q4
      }
      blockSum =
        blockSum +
        dx0 * (ds0 * isum0a + ds1 * isum0b) +
        dx1 * (ds2 * isum1a + ds3 * isum1b) +
        dx2 * (ds4 * isum2a + ds5 * isum2b) +
        dx3 * (ds6 * isum3a + ds7 * isum3b)
      xOff = xOff + 136
    }
    sum = sum + blockSum
    wOff = wOff + 210
  }
  return sum
}

function vecDotIQ4_NL_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var sum = 0.0
  var wOff = srcOffset
  var xOff = 0
  var u8 = ggufUint8
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[wOff] | (u8[wOff + 1] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = wOff + 2
    var qx = xOff + 2
    var isum = 0
    for (var j = 0; j < 16; j = j + 1) {
      var qByte = u8[qw + j]
      isum =
        isum +
        xQ8i8[qx + j] * kvalues_iq4nl[qByte & 0xf] +
        xQ8i8[qx + j + 16] * kvalues_iq4nl[qByte >> 4]
    }
    sum = sum + dw * dx * isum
    wOff = wOff + 18
    xOff = xOff + 34
  }
  return sum
}

// Get Q8_0-input vec_dot function for a type (null for float types)
function getVecDotQ8Func(type) {
  switch (type) {
    case GGML_TYPE.Q4_0:
      return vecDotQ4_0_Q8_0
    case GGML_TYPE.Q4_1:
      return vecDotQ4_1_Q8_0
    case GGML_TYPE.Q5_0:
      return vecDotQ5_0_Q8_0
    case GGML_TYPE.Q5_1:
      return vecDotQ5_1_Q8_0
    case GGML_TYPE.Q8_0:
      return null // Float×Q8 path is faster in JS (no quantization overhead)
    case GGML_TYPE.Q2_K:
      return vecDotQ2_K_Q8_0
    case GGML_TYPE.Q3_K:
      return vecDotQ3_K_Q8_0
    case GGML_TYPE.Q4_K:
      return vecDotQ4_K_Q8_0
    case GGML_TYPE.Q5_K:
      return vecDotQ5_K_Q8_0
    case GGML_TYPE.Q6_K:
      return vecDotQ6_K_Q8_0
    case GGML_TYPE.IQ4_NL:
      return vecDotIQ4_NL_Q8_0
    default:
      return null
  }
}

function matmulQuantized(out, x, qw) {
  var rows = qw.rows
  var cols = qw.cols
  var baseOffset = qw.dataOffset
  var rowSize = qw.rowSize
  var dotQ8Func = qw.dotQ8Func

  if (qw.type === GGML_TYPE.Q8_0) {
    // Per-matrix local views for better V8 bounds check elimination
    matmulQ8_0Local(out, x, qw.localU8, qw.localI8, rows, cols, rowSize)
  } else if (dotQ8Func) {
    // Quantize x to Q8_0 once, then use integer dot products
    quantizeToQ8_0Cache(x, 0, xQ8Buf, xQ8Int8Buf, 0, cols)
    for (var i = 0; i < rows; i = i + 1) {
      out[i] = dotQ8Func(xQ8Buf, xQ8Int8Buf, baseOffset + i * rowSize, cols)
    }
  } else {
    // Float weight types - use original float dot
    var dotFunc = qw.dotFunc
    for (var i = 0; i < rows; i = i + 1) {
      out[i] = dotFunc(x, baseOffset + i * rowSize, cols)
    }
  }
}

// Quantized matmul using pre-quantized x (avoids redundant quantization)
// Caller must have already quantized x into xQ8Buf/xQ8Int8Buf
function matmulQuantizedPreQ8(out, qw) {
  var rows = qw.rows
  var dotQ8Func = qw.dotQ8Func
  var baseOffset = qw.dataOffset
  var rowSize = qw.rowSize
  var cols = qw.cols
  for (var i = 0; i < rows; i = i + 1) {
    out[i] = dotQ8Func(xQ8Buf, xQ8Int8Buf, baseOffset + i * rowSize, cols)
  }
}

// Batched matmul: process multiple input vectors against same weight matrix
// Weight data is read once per row and reused across all batch elements
var PREFILL_BATCH_SIZE = 32

function matmulQuantizedBatch(outs, xs, qw, batchSize) {
  var rows = qw.rows
  var cols = qw.cols
  var baseOffset = qw.dataOffset
  var rowSize = qw.rowSize
  var dotQ8Func = qw.dotQ8Func

  if (qw.type === GGML_TYPE.Q8_0) {
    matmulQ8_0LocalBatch(
      outs,
      xs,
      qw.localU8,
      qw.localI8,
      rows,
      cols,
      rowSize,
      batchSize
    )
  } else if (dotQ8Func) {
    var bQ8 = state.batchQ8
    var bQ8i8 = state.batchQ8i8
    for (var b = 0; b < batchSize; b = b + 1) {
      quantizeToQ8_0Cache(xs[b], 0, bQ8[b], bQ8i8[b], 0, cols)
    }
    for (var i = 0; i < rows; i = i + 1) {
      var rowOff = baseOffset + i * rowSize
      for (var b = 0; b < batchSize; b = b + 1) {
        outs[b][i] = dotQ8Func(bQ8[b], bQ8i8[b], rowOff, cols)
      }
    }
  } else {
    var dotFunc = qw.dotFunc
    for (var i = 0; i < rows; i = i + 1) {
      var rowOff = baseOffset + i * rowSize
      for (var b = 0; b < batchSize; b = b + 1) {
        outs[b][i] = dotFunc(xs[b], rowOff, cols)
      }
    }
  }
}

// Q8_0 batch matmul: dequantize 4 rows into reusable buffer, then flat dot product
// Dequantization cost is amortized across all batch elements (32x reuse)
function matmulQ8_0LocalBatch(
  outs,
  xs,
  localU8,
  localI8,
  rows,
  cols,
  rowSize,
  batchSize
) {
  var nb = cols >> 5
  var rows4 = rows & ~3
  var buf = matmulDeqBuf
  var off1 = cols
  var off2 = cols + cols
  var off3 = off2 + cols
  var cols4 = cols & ~3
  for (var i = 0; i < rows4; i = i + 4) {
    // Dequantize 4 weight rows into the reusable Float64 buffer
    var bo = i * rowSize
    for (var r = 0; r < 4; r = r + 1) {
      var bOff = r * cols
      var bk = bo
      var idx = bOff
      for (var b = 0; b < nb; b = b + 1) {
        var d = fp16Table[localU8[bk] | (localU8[bk + 1] << 8)]
        var qo = bk + 2
        buf[idx] = d * localI8[qo]
        buf[idx + 1] = d * localI8[qo + 1]
        buf[idx + 2] = d * localI8[qo + 2]
        buf[idx + 3] = d * localI8[qo + 3]
        buf[idx + 4] = d * localI8[qo + 4]
        buf[idx + 5] = d * localI8[qo + 5]
        buf[idx + 6] = d * localI8[qo + 6]
        buf[idx + 7] = d * localI8[qo + 7]
        buf[idx + 8] = d * localI8[qo + 8]
        buf[idx + 9] = d * localI8[qo + 9]
        buf[idx + 10] = d * localI8[qo + 10]
        buf[idx + 11] = d * localI8[qo + 11]
        buf[idx + 12] = d * localI8[qo + 12]
        buf[idx + 13] = d * localI8[qo + 13]
        buf[idx + 14] = d * localI8[qo + 14]
        buf[idx + 15] = d * localI8[qo + 15]
        buf[idx + 16] = d * localI8[qo + 16]
        buf[idx + 17] = d * localI8[qo + 17]
        buf[idx + 18] = d * localI8[qo + 18]
        buf[idx + 19] = d * localI8[qo + 19]
        buf[idx + 20] = d * localI8[qo + 20]
        buf[idx + 21] = d * localI8[qo + 21]
        buf[idx + 22] = d * localI8[qo + 22]
        buf[idx + 23] = d * localI8[qo + 23]
        buf[idx + 24] = d * localI8[qo + 24]
        buf[idx + 25] = d * localI8[qo + 25]
        buf[idx + 26] = d * localI8[qo + 26]
        buf[idx + 27] = d * localI8[qo + 27]
        buf[idx + 28] = d * localI8[qo + 28]
        buf[idx + 29] = d * localI8[qo + 29]
        buf[idx + 30] = d * localI8[qo + 30]
        buf[idx + 31] = d * localI8[qo + 31]
        bk = bk + 34
        idx = idx + 32
      }
      bo = bo + rowSize
    }
    // Process 3 batch elements at a time, sharing buf reads (12 independent chains)
    var batchTrips = batchSize - (batchSize % 3)
    for (var batch = 0; batch < batchTrips; batch = batch + 3) {
      var xA = xs[batch]
      var xB = xs[batch + 1]
      var xC = xs[batch + 2]
      var s0 = 0.0
      var s1 = 0.0
      var s2 = 0.0
      var s3 = 0.0
      var t0 = 0.0
      var t1 = 0.0
      var t2 = 0.0
      var t3 = 0.0
      var u0 = 0.0
      var u1 = 0.0
      var u2 = 0.0
      var u3 = 0.0
      for (var j = 0; j < cols4; j = j + 4) {
        var a = xA[j]
        var b = xA[j + 1]
        var c = xA[j + 2]
        var d = xA[j + 3]
        var e = xB[j]
        var f = xB[j + 1]
        var g = xB[j + 2]
        var h = xB[j + 3]
        var p = xC[j]
        var q = xC[j + 1]
        var r = xC[j + 2]
        var v = xC[j + 3]
        var w0 = buf[j]
        var w1 = buf[j + 1]
        var w2 = buf[j + 2]
        var w3 = buf[j + 3]
        s0 = s0 + a * w0 + b * w1 + c * w2 + d * w3
        t0 = t0 + e * w0 + f * w1 + g * w2 + h * w3
        u0 = u0 + p * w0 + q * w1 + r * w2 + v * w3
        w0 = buf[off1 + j]
        w1 = buf[off1 + j + 1]
        w2 = buf[off1 + j + 2]
        w3 = buf[off1 + j + 3]
        s1 = s1 + a * w0 + b * w1 + c * w2 + d * w3
        t1 = t1 + e * w0 + f * w1 + g * w2 + h * w3
        u1 = u1 + p * w0 + q * w1 + r * w2 + v * w3
        w0 = buf[off2 + j]
        w1 = buf[off2 + j + 1]
        w2 = buf[off2 + j + 2]
        w3 = buf[off2 + j + 3]
        s2 = s2 + a * w0 + b * w1 + c * w2 + d * w3
        t2 = t2 + e * w0 + f * w1 + g * w2 + h * w3
        u2 = u2 + p * w0 + q * w1 + r * w2 + v * w3
        w0 = buf[off3 + j]
        w1 = buf[off3 + j + 1]
        w2 = buf[off3 + j + 2]
        w3 = buf[off3 + j + 3]
        s3 = s3 + a * w0 + b * w1 + c * w2 + d * w3
        t3 = t3 + e * w0 + f * w1 + g * w2 + h * w3
        u3 = u3 + p * w0 + q * w1 + r * w2 + v * w3
      }
      outs[batch][i] = s0
      outs[batch][i + 1] = s1
      outs[batch][i + 2] = s2
      outs[batch][i + 3] = s3
      outs[batch + 1][i] = t0
      outs[batch + 1][i + 1] = t1
      outs[batch + 1][i + 2] = t2
      outs[batch + 1][i + 3] = t3
      outs[batch + 2][i] = u0
      outs[batch + 2][i + 1] = u1
      outs[batch + 2][i + 2] = u2
      outs[batch + 2][i + 3] = u3
    }
    // Handle remaining 1-2 batch elements one at a time
    for (var batch = batchTrips; batch < batchSize; batch = batch + 1) {
      var xArr = xs[batch]
      var s0 = 0.0
      var s1 = 0.0
      var s2 = 0.0
      var s3 = 0.0
      for (var j = 0; j < cols4; j = j + 4) {
        var a = xArr[j]
        var b = xArr[j + 1]
        var c = xArr[j + 2]
        var d = xArr[j + 3]
        s0 = s0 + a * buf[j] + b * buf[j + 1] + c * buf[j + 2] + d * buf[j + 3]
        s1 =
          s1 +
          a * buf[off1 + j] +
          b * buf[off1 + j + 1] +
          c * buf[off1 + j + 2] +
          d * buf[off1 + j + 3]
        s2 =
          s2 +
          a * buf[off2 + j] +
          b * buf[off2 + j + 1] +
          c * buf[off2 + j + 2] +
          d * buf[off2 + j + 3]
        s3 =
          s3 +
          a * buf[off3 + j] +
          b * buf[off3 + j + 1] +
          c * buf[off3 + j + 2] +
          d * buf[off3 + j + 3]
      }
      outs[batch][i] = s0
      outs[batch][i + 1] = s1
      outs[batch][i + 2] = s2
      outs[batch][i + 3] = s3
    }
  }
}

// Q8_0 matmul: 4 rows at a time, sharing x reads across rows
function matmulQ8_0Local(out, x, localU8, localI8, rows, cols, rowSize) {
  var nb = cols >> 5
  var rows4 = rows & ~3
  var rs2 = rowSize + rowSize
  var rs3 = rs2 + rowSize
  for (var i = 0; i < rows4; i = i + 4) {
    var sum0 = 0.0
    var sum1 = 0.0
    var sum2 = 0.0
    var sum3 = 0.0
    var bo0 = i * rowSize
    var bo1 = bo0 + rowSize
    var bo2 = bo0 + rs2
    var bo3 = bo0 + rs3
    var xb = 0
    for (var b = 0; b < nb; b = b + 1) {
      var x0 = x[xb]
      var x1 = x[xb + 1]
      var x2 = x[xb + 2]
      var x3 = x[xb + 3]
      var x4 = x[xb + 4]
      var x5 = x[xb + 5]
      var x6 = x[xb + 6]
      var x7 = x[xb + 7]
      var x8 = x[xb + 8]
      var x9 = x[xb + 9]
      var x10 = x[xb + 10]
      var x11 = x[xb + 11]
      var x12 = x[xb + 12]
      var x13 = x[xb + 13]
      var x14 = x[xb + 14]
      var x15 = x[xb + 15]
      var x16 = x[xb + 16]
      var x17 = x[xb + 17]
      var x18 = x[xb + 18]
      var x19 = x[xb + 19]
      var x20 = x[xb + 20]
      var x21 = x[xb + 21]
      var x22 = x[xb + 22]
      var x23 = x[xb + 23]
      var x24 = x[xb + 24]
      var x25 = x[xb + 25]
      var x26 = x[xb + 26]
      var x27 = x[xb + 27]
      var x28 = x[xb + 28]
      var x29 = x[xb + 29]
      var x30 = x[xb + 30]
      var x31 = x[xb + 31]
      // Row 0
      var d0 = fp16Table[localU8[bo0] | (localU8[bo0 + 1] << 8)]
      var q0 = bo0 + 2
      sum0 =
        sum0 +
        d0 *
          (x0 * localI8[q0] +
            x1 * localI8[q0 + 1] +
            x2 * localI8[q0 + 2] +
            x3 * localI8[q0 + 3] +
            x4 * localI8[q0 + 4] +
            x5 * localI8[q0 + 5] +
            x6 * localI8[q0 + 6] +
            x7 * localI8[q0 + 7] +
            x8 * localI8[q0 + 8] +
            x9 * localI8[q0 + 9] +
            x10 * localI8[q0 + 10] +
            x11 * localI8[q0 + 11] +
            x12 * localI8[q0 + 12] +
            x13 * localI8[q0 + 13] +
            x14 * localI8[q0 + 14] +
            x15 * localI8[q0 + 15] +
            x16 * localI8[q0 + 16] +
            x17 * localI8[q0 + 17] +
            x18 * localI8[q0 + 18] +
            x19 * localI8[q0 + 19] +
            x20 * localI8[q0 + 20] +
            x21 * localI8[q0 + 21] +
            x22 * localI8[q0 + 22] +
            x23 * localI8[q0 + 23] +
            x24 * localI8[q0 + 24] +
            x25 * localI8[q0 + 25] +
            x26 * localI8[q0 + 26] +
            x27 * localI8[q0 + 27] +
            x28 * localI8[q0 + 28] +
            x29 * localI8[q0 + 29] +
            x30 * localI8[q0 + 30] +
            x31 * localI8[q0 + 31])
      // Row 1
      var d1 = fp16Table[localU8[bo1] | (localU8[bo1 + 1] << 8)]
      var q1 = bo1 + 2
      sum1 =
        sum1 +
        d1 *
          (x0 * localI8[q1] +
            x1 * localI8[q1 + 1] +
            x2 * localI8[q1 + 2] +
            x3 * localI8[q1 + 3] +
            x4 * localI8[q1 + 4] +
            x5 * localI8[q1 + 5] +
            x6 * localI8[q1 + 6] +
            x7 * localI8[q1 + 7] +
            x8 * localI8[q1 + 8] +
            x9 * localI8[q1 + 9] +
            x10 * localI8[q1 + 10] +
            x11 * localI8[q1 + 11] +
            x12 * localI8[q1 + 12] +
            x13 * localI8[q1 + 13] +
            x14 * localI8[q1 + 14] +
            x15 * localI8[q1 + 15] +
            x16 * localI8[q1 + 16] +
            x17 * localI8[q1 + 17] +
            x18 * localI8[q1 + 18] +
            x19 * localI8[q1 + 19] +
            x20 * localI8[q1 + 20] +
            x21 * localI8[q1 + 21] +
            x22 * localI8[q1 + 22] +
            x23 * localI8[q1 + 23] +
            x24 * localI8[q1 + 24] +
            x25 * localI8[q1 + 25] +
            x26 * localI8[q1 + 26] +
            x27 * localI8[q1 + 27] +
            x28 * localI8[q1 + 28] +
            x29 * localI8[q1 + 29] +
            x30 * localI8[q1 + 30] +
            x31 * localI8[q1 + 31])
      // Row 2
      var d2 = fp16Table[localU8[bo2] | (localU8[bo2 + 1] << 8)]
      var q2 = bo2 + 2
      sum2 =
        sum2 +
        d2 *
          (x0 * localI8[q2] +
            x1 * localI8[q2 + 1] +
            x2 * localI8[q2 + 2] +
            x3 * localI8[q2 + 3] +
            x4 * localI8[q2 + 4] +
            x5 * localI8[q2 + 5] +
            x6 * localI8[q2 + 6] +
            x7 * localI8[q2 + 7] +
            x8 * localI8[q2 + 8] +
            x9 * localI8[q2 + 9] +
            x10 * localI8[q2 + 10] +
            x11 * localI8[q2 + 11] +
            x12 * localI8[q2 + 12] +
            x13 * localI8[q2 + 13] +
            x14 * localI8[q2 + 14] +
            x15 * localI8[q2 + 15] +
            x16 * localI8[q2 + 16] +
            x17 * localI8[q2 + 17] +
            x18 * localI8[q2 + 18] +
            x19 * localI8[q2 + 19] +
            x20 * localI8[q2 + 20] +
            x21 * localI8[q2 + 21] +
            x22 * localI8[q2 + 22] +
            x23 * localI8[q2 + 23] +
            x24 * localI8[q2 + 24] +
            x25 * localI8[q2 + 25] +
            x26 * localI8[q2 + 26] +
            x27 * localI8[q2 + 27] +
            x28 * localI8[q2 + 28] +
            x29 * localI8[q2 + 29] +
            x30 * localI8[q2 + 30] +
            x31 * localI8[q2 + 31])
      // Row 3
      var d3 = fp16Table[localU8[bo3] | (localU8[bo3 + 1] << 8)]
      var q3 = bo3 + 2
      sum3 =
        sum3 +
        d3 *
          (x0 * localI8[q3] +
            x1 * localI8[q3 + 1] +
            x2 * localI8[q3 + 2] +
            x3 * localI8[q3 + 3] +
            x4 * localI8[q3 + 4] +
            x5 * localI8[q3 + 5] +
            x6 * localI8[q3 + 6] +
            x7 * localI8[q3 + 7] +
            x8 * localI8[q3 + 8] +
            x9 * localI8[q3 + 9] +
            x10 * localI8[q3 + 10] +
            x11 * localI8[q3 + 11] +
            x12 * localI8[q3 + 12] +
            x13 * localI8[q3 + 13] +
            x14 * localI8[q3 + 14] +
            x15 * localI8[q3 + 15] +
            x16 * localI8[q3 + 16] +
            x17 * localI8[q3 + 17] +
            x18 * localI8[q3 + 18] +
            x19 * localI8[q3 + 19] +
            x20 * localI8[q3 + 20] +
            x21 * localI8[q3 + 21] +
            x22 * localI8[q3 + 22] +
            x23 * localI8[q3 + 23] +
            x24 * localI8[q3 + 24] +
            x25 * localI8[q3 + 25] +
            x26 * localI8[q3 + 26] +
            x27 * localI8[q3 + 27] +
            x28 * localI8[q3 + 28] +
            x29 * localI8[q3 + 29] +
            x30 * localI8[q3 + 30] +
            x31 * localI8[q3 + 31])
      bo0 = bo0 + 34
      bo1 = bo1 + 34
      bo2 = bo2 + 34
      bo3 = bo3 + 34
      xb = xb + 32
    }
    out[i] = sum0
    out[i + 1] = sum1
    out[i + 2] = sum2
    out[i + 3] = sum3
  }
}

// ----------------------------------------------------------------------------
// Math functions

// Fast tanh approximation using [3,3] Padé approximant
// Accurate to ~1e-7 for |x| < 4, exact ±1 beyond
function fastTanh(x) {
  if (x < -4.0) {
    return -1.0
  }
  if (x > 4.0) {
    return 1.0
  }
  var x2 = x * x
  return (
    (x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))) /
    (135135.0 + x2 * (62370.0 + x2 * (3150.0 + 28.0 * x2)))
  )
}

function rmsnorm(out, x, w, size, invSize, eps) {
  eps = eps || 1e-5
  invSize = invSize || 1.0 / size
  var ss = 0.0
  // Loop unrolling: process 4 elements at a time
  var size4 = size & ~3 // size - (size % 4)
  var i = 0
  for (; i < size4; i = i + 4) {
    var x0 = x[i]
    var x1 = x[i + 1]
    var x2 = x[i + 2]
    var x3 = x[i + 3]
    ss = ss + x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
  }
  for (; i < size; i = i + 1) {
    ss = ss + x[i] * x[i]
  }
  ss = 1.0 / Math.sqrt(ss * invSize + eps)
  i = 0
  for (; i < size4; i = i + 4) {
    out[i] = w[i] * ss * x[i]
    out[i + 1] = w[i + 1] * ss * x[i + 1]
    out[i + 2] = w[i + 2] * ss * x[i + 2]
    out[i + 3] = w[i + 3] * ss * x[i + 3]
  }
  for (; i < size; i = i + 1) {
    out[i] = w[i] * ss * x[i]
  }
}

function rmsnormGemma(out, x, w, size, eps, invSize) {
  // Note: GGUF conversion already adds +1 to Gemma norm weights
  invSize = invSize || 1.0 / size
  var ss = 0.0
  // Loop unrolling: process 4 elements at a time
  var size4 = size & ~3
  var i = 0
  for (; i < size4; i = i + 4) {
    var x0 = x[i]
    var x1 = x[i + 1]
    var x2 = x[i + 2]
    var x3 = x[i + 3]
    ss = ss + x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
  }
  for (; i < size; i = i + 1) {
    ss = ss + x[i] * x[i]
  }
  ss = 1.0 / Math.sqrt(ss * invSize + eps)
  i = 0
  for (; i < size4; i = i + 4) {
    out[i] = w[i] * ss * x[i]
    out[i + 1] = w[i + 1] * ss * x[i + 1]
    out[i + 2] = w[i + 2] * ss * x[i + 2]
    out[i + 3] = w[i + 3] * ss * x[i + 3]
  }
  for (; i < size; i = i + 1) {
    out[i] = w[i] * ss * x[i]
  }
}

// In-place RMS norm at an offset into arr (avoids subarray allocation)
function rmsnormGemmaAt(arr, arrOffset, w, size, eps, invSize) {
  var ss = 0.0
  var size4 = size & ~3
  var end4 = arrOffset + size4
  var end = arrOffset + size
  var i = arrOffset
  var wi = 0
  for (; i < end4; i = i + 4, wi = wi + 4) {
    var x0 = arr[i]
    var x1 = arr[i + 1]
    var x2 = arr[i + 2]
    var x3 = arr[i + 3]
    ss = ss + x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
  }
  for (; i < end; i = i + 1) {
    ss = ss + arr[i] * arr[i]
  }
  ss = 1.0 / Math.sqrt(ss * invSize + eps)
  i = arrOffset
  wi = 0
  for (; i < end4; i = i + 4, wi = wi + 4) {
    arr[i] = w[wi] * ss * arr[i]
    arr[i + 1] = w[wi + 1] * ss * arr[i + 1]
    arr[i + 2] = w[wi + 2] * ss * arr[i + 2]
    arr[i + 3] = w[wi + 3] * ss * arr[i + 3]
  }
  for (; i < end; i = i + 1, wi = wi + 1) {
    arr[i] = w[wi] * ss * arr[i]
  }
}

// Fused embedding scale + RMS norm for Gemma first layer
// Scales x in-place and computes rmsnorm in 2 passes instead of 3
function rmsnormGemmaFusedScale(out, x, w, size, eps, invSize, scale) {
  var ss = 0.0
  var size4 = size & ~3
  var i = 0
  // Pass 1: scale x in-place and accumulate sum of squares
  for (; i < size4; i = i + 4) {
    var x0 = x[i] * scale
    var x1 = x[i + 1] * scale
    var x2 = x[i + 2] * scale
    var x3 = x[i + 3] * scale
    x[i] = x0
    x[i + 1] = x1
    x[i + 2] = x2
    x[i + 3] = x3
    ss = ss + x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
  }
  for (; i < size; i = i + 1) {
    var xv = x[i] * scale
    x[i] = xv
    ss = ss + xv * xv
  }
  // Pass 2: normalize
  ss = 1.0 / Math.sqrt(ss * invSize + eps)
  i = 0
  for (; i < size4; i = i + 4) {
    out[i] = w[i] * ss * x[i]
    out[i + 1] = w[i + 1] * ss * x[i + 1]
    out[i + 2] = w[i + 2] * ss * x[i + 2]
    out[i + 3] = w[i + 3] * ss * x[i + 3]
  }
  for (; i < size; i = i + 1) {
    out[i] = w[i] * ss * x[i]
  }
}

function accum(a, b, size) {
  var size4 = size & ~3
  var i = 0
  for (; i < size4; i = i + 4) {
    a[i] = a[i] + b[i]
    a[i + 1] = a[i + 1] + b[i + 1]
    a[i + 2] = a[i + 2] + b[i + 2]
    a[i + 3] = a[i + 3] + b[i + 3]
  }
  for (; i < size; i = i + 1) {
    a[i] = a[i] + b[i]
  }
}

// ----------------------------------------------------------------------------
// GGUF parsing

function parseGGUF(arrayBuffer) {
  ggufData = arrayBuffer
  dataView = new DataView(arrayBuffer)
  offset = 0

  var magic = readUint32()
  if (magic !== GGUF_MAGIC) {
    throw new Error(
      "Invalid GGUF magic: expected " +
        GGUF_MAGIC.toString(16) +
        ", got " +
        magic.toString(16)
    )
  }

  var version = readUint32()
  var nTensors = readUint64()
  var nKV = readUint64()

  var metadata = {}
  var i
  var key
  var valueType

  for (i = 0; i < nKV; i = i + 1) {
    key = readString()
    valueType = readUint32()
    metadata[key] = readGGUFValue(valueType)
  }

  var tensors = {}
  for (i = 0; i < nTensors; i = i + 1) {
    var name = readString()
    var nDims = readUint32()
    var dims = []
    for (var d = 0; d < nDims; d = d + 1) {
      dims.push(readUint64())
    }
    var type = readUint32()
    var tensorOffset = readUint64()

    var nElements = 1
    for (var d = 0; d < dims.length; d = d + 1) {
      nElements *= dims[d]
    }

    tensors[name] = {
      dims: dims,
      type: type,
      offset: tensorOffset,
      nElements: nElements,
    }
  }

  var alignment = metadata["general.alignment"] || 32
  var tensorDataOffset = Math.ceil(offset / alignment) * alignment

  return {
    version: version,
    metadata: metadata,
    tensors: tensors,
    tensorDataOffset: tensorDataOffset,
  }
}

function readGGUFValue(type) {
  switch (type) {
    case GGUF_TYPE.UINT8:
      return readUint8()
    case GGUF_TYPE.INT8:
      return readInt8()
    case GGUF_TYPE.UINT16:
      return readUint16()
    case GGUF_TYPE.INT16:
      var int16Val = dataView.getInt16(offset, true)
      offset = offset + 2
      return int16Val
    case GGUF_TYPE.UINT32:
      return readUint32()
    case GGUF_TYPE.INT32:
      return readInt32()
    case GGUF_TYPE.FLOAT32:
      return readFloat32()
    case GGUF_TYPE.BOOL:
      return readUint8() !== 0
    case GGUF_TYPE.STRING:
      return readString()
    case GGUF_TYPE.UINT64:
      return readUint64()
    case GGUF_TYPE.INT64:
      return readInt64()
    case GGUF_TYPE.FLOAT64:
      return readFloat64()
    case GGUF_TYPE.ARRAY:
      var arrType = readUint32()
      var arrLen = readUint64()
      if (arrType === GGUF_TYPE.FLOAT32) {
        var arrBytes = arrLen * 4
        var arr
        if (offset % 4 === 0) {
          arr = new Float32Array(ggufData, offset, arrLen)
        } else {
          arr = new Float32Array(ggufData.slice(offset, offset + arrBytes))
        }
        offset = offset + arrBytes
        return arr
      }
      if (arrType === GGUF_TYPE.UINT32) {
        var arrBytes = arrLen * 4
        var arr
        if (offset % 4 === 0) {
          arr = new Uint32Array(ggufData, offset, arrLen)
        } else {
          arr = new Uint32Array(ggufData.slice(offset, offset + arrBytes))
        }
        offset = offset + arrBytes
        return arr
      }
      var arr = new Array(arrLen)
      for (var i = 0; i < arrLen; i = i + 1) {
        arr[i] = readGGUFValue(arrType)
      }
      return arr
    default:
      throw new Error("Unknown GGUF type: " + type)
  }
}

// ----------------------------------------------------------------------------
// Model loading

function loadModel(arrayBuffer) {
  // Reset vocab cache when loading a new model
  vocabTrie = null
  vocabMap = null

  // Initialize cached buffer views for fast matmul access
  ggufUint8 = new Uint8Array(arrayBuffer)
  ggufInt8 = new Int8Array(arrayBuffer)

  var gguf = parseGGUF(arrayBuffer)
  var meta = gguf.metadata

  var arch = meta["general.architecture"] || "llama"
  var keyPrefix = arch

  if (arch === "gemma3" || arch === "gemma2" || arch === "gemma") {
    keyPrefix = "gemma3"
    if (!meta["gemma3.embedding_length"]) {
      if (meta["gemma2.embedding_length"]) {
        keyPrefix = "gemma2"
      } else if (meta["gemma.embedding_length"]) {
        keyPrefix = "gemma"
      }
    }
  }

  var isGemma = arch === "gemma3" || arch === "gemma2" || arch === "gemma"

  postMessage({
    type: "progress",
  })

  // Limit context length to avoid massive KV cache allocations in browser
  var modelSeqLen = meta[keyPrefix + ".context_length"] || 2048

  config = {
    dim: meta[keyPrefix + ".embedding_length"] || 4096,
    hiddenDim: meta[keyPrefix + ".feed_forward_length"] || 11008,
    nLayers: meta[keyPrefix + ".block_count"] || 32,
    nHeads: meta[keyPrefix + ".attention.head_count"] || 32,
    nKvHeads:
      meta[keyPrefix + ".attention.head_count_kv"] ||
      meta[keyPrefix + ".attention.head_count"] ||
      32,
    vocabSize: meta[keyPrefix + ".vocab_size"] || 32000,
    seqLen: contextSize > 0 ? Math.min(modelSeqLen, contextSize) : modelSeqLen,
    ropeTheta:
      meta[keyPrefix + ".rope.freq_base"] || (isGemma ? 1000000.0 : 500000.0),
    headDim: meta[keyPrefix + ".attention.key_length"] || 0,
    isGemma: isGemma,
    rmsNormEps: meta[keyPrefix + ".attention.layer_norm_rms_epsilon"] || 1e-6,
    finalLogitSoftcapping: meta[keyPrefix + ".final_logit_softcapping"] || 0.0,
    // Sliding window attention parameters for Gemma3
    swaWindow: meta[keyPrefix + ".attention.sliding_window"] || 0,
    ropeThetaSwa: meta[keyPrefix + ".rope.freq_base_swa"] || 10000.0,
    swaPattern: 6, // Gemma3 uses pattern 6: layers 0-4 are SWA, layer 5 is dense, etc.
  }

  if (config.headDim === 0) {
    config.headDim = (config.dim / config.nHeads) | 0
  }

  var vocabTokens = meta["tokenizer.ggml.tokens"] || []
  var vocabScores = meta["tokenizer.ggml.scores"] || []

  if (vocabTokens.length > 0) {
    config.vocabSize = vocabTokens.length
  }

  tokenizer = {
    vocab: vocabTokens,
    scores: vocabScores,
    vocabSize: config.vocabSize,
    bosToken: meta["tokenizer.ggml.bos_token_id"] || 1,
    eosToken: meta["tokenizer.ggml.eos_token_id"] || 2,
    eotToken: -1,
  }

  // Look up model-specific end tokens once at init time
  if (config.isGemma) {
    var endTurn = findSpecialToken("<end_of_turn>")
    if (endTurn < 0) {
      endTurn = 107
    }
    tokenizer.eotToken = endTurn
  } else {
    var eot = findSpecialToken("<|eot_id|>")
    if (eot < 0) {
      eot = 128009
    }
    tokenizer.eotToken = eot
  }

  postMessage({ type: "progress", message: "Loading weights..." })

  weights = loadWeights(gguf)
  state = createRunState(config)

  postMessage({ type: "progress", message: "Model loaded!" })

  return {
    config: config,
    tokenizer: tokenizer,
  }
}

function loadWeights(gguf) {
  var tensors = gguf.tensors
  var baseOffset = gguf.tensorDataOffset
  var w = {}

  // Load tensor as dequantized float (for small tensors like norms and embeddings)
  function loadTensorFloat(name) {
    var t = tensors[name]
    if (!t) {
      return null
    }
    return dequantizeTensor(baseOffset + t.offset, t.nElements, t.type)
  }

  // Load tensor keeping it quantized (for large weight matrices)
  // Returns a QuantizedTensor object with pre-computed rowSize and dotFunc
  function loadTensorQuantized(name, rows, cols) {
    var t = tensors[name]
    if (!t) {
      return null
    }
    var rs = getRowSize(cols, t.type)
    var off = baseOffset + t.offset
    var totalBytes = rows * rs
    var result = {
      dataOffset: off,
      type: t.type,
      rows: rows,
      cols: cols,
      rowSize: rs,
      dotFunc: getVecDotFunc(t.type),
      dotQ8Func: getVecDotQ8Func(t.type),
    }
    // Per-matrix typed array views for Q8_0 (helps V8 bounds check elimination)
    if (t.type === GGML_TYPE.Q8_0) {
      result.localU8 = new Uint8Array(ggufData, off, totalBytes)
      result.localI8 = new Int8Array(ggufData, off, totalBytes)
    }
    return result
  }

  function loadLayerTensorFloat(layer, suffix) {
    var name = "blk." + layer + "." + suffix
    return loadTensorFloat(name)
  }

  function loadLayerTensorQuantized(layer, suffix, rows, cols) {
    var name = "blk." + layer + "." + suffix
    return loadTensorQuantized(name, rows, cols)
  }

  var headSize = config.headDim
  var kvDim = config.nKvHeads * headSize
  var qDim = config.nHeads * headSize

  postMessage({ type: "progress", message: "Loading embeddings..." })
  // Token embeddings - keep quantized to save memory, dequantize on-demand
  var embTensor = tensors["token_embd.weight"]
  w.tokenEmbedding = {
    dataOffset: baseOffset + embTensor.offset,
    type: embTensor.type,
    rows: config.vocabSize,
    cols: config.dim,
    rowSize: getRowSize(config.dim, embTensor.type),
  }

  // Use per-layer arrays
  w.layers = []

  for (var l = 0; l < config.nLayers; l = l + 1) {
    postMessage({
      type: "progress",
      message: "Loading layer " + (l + 1) + "/" + config.nLayers + "...",
    })

    var layer = {}

    // RMS norm weights - small, dequantize
    layer.rmsAttWeight = loadLayerTensorFloat(l, "attn_norm.weight")
    layer.rmsFfnWeight = loadLayerTensorFloat(l, "ffn_norm.weight")

    // Attention weights - KEEP QUANTIZED!
    // Weight matrices are (out_dim, in_dim) = (rows, cols)
    layer.wq = loadLayerTensorQuantized(l, "attn_q.weight", qDim, config.dim)
    layer.wk = loadLayerTensorQuantized(l, "attn_k.weight", kvDim, config.dim)
    layer.wv = loadLayerTensorQuantized(l, "attn_v.weight", kvDim, config.dim)
    layer.wo = loadLayerTensorQuantized(l, "attn_output.weight", config.dim, qDim)

    // FFN weights - KEEP QUANTIZED!
    layer.w1 = loadLayerTensorQuantized(
      l,
      "ffn_gate.weight",
      config.hiddenDim,
      config.dim
    )
    layer.w2 = loadLayerTensorQuantized(
      l,
      "ffn_down.weight",
      config.dim,
      config.hiddenDim
    )
    layer.w3 = loadLayerTensorQuantized(
      l,
      "ffn_up.weight",
      config.hiddenDim,
      config.dim
    )

    // Gemma-specific weights - small, dequantize
    if (config.isGemma) {
      layer.attnQNorm = loadLayerTensorFloat(l, "attn_q_norm.weight")
      layer.attnKNorm = loadLayerTensorFloat(l, "attn_k_norm.weight")
      layer.attnPostNorm = loadLayerTensorFloat(l, "post_attention_norm.weight")
      layer.ffnPostNorm = loadLayerTensorFloat(l, "post_ffw_norm.weight")
    }

    w.layers.push(layer)
  }

  // Final norm - small, dequantize
  w.rmsFinalWeight = loadTensorFloat("output_norm.weight")

  // Output projection - may be tied to embeddings or separate
  var outputTensor = tensors["output.weight"]
  if (outputTensor) {
    // Load as quantized tensor
    w.wcls = loadTensorQuantized("output.weight", config.vocabSize, config.dim)
  } else {
    // Use tied embeddings - reference the same quantized embedding tensor
    var embOff = w.tokenEmbedding.dataOffset
    var embRowSize = w.tokenEmbedding.rowSize
    var embType = w.tokenEmbedding.type
    var embTotalBytes = config.vocabSize * embRowSize
    w.wcls = {
      dataOffset: embOff,
      type: embType,
      rows: config.vocabSize,
      cols: config.dim,
      rowSize: embRowSize,
      dotFunc: getVecDotFunc(embType),
      dotQ8Func: getVecDotQ8Func(embType),
    }
    if (embType === GGML_TYPE.Q8_0) {
      w.wcls.localU8 = new Uint8Array(ggufData, embOff, embTotalBytes)
      w.wcls.localI8 = new Int8Array(ggufData, embOff, embTotalBytes)
    }
  }

  return w
}

function createRunState(p) {
  var headSize = p.headDim
  var kvDim = p.nKvHeads * headSize
  var qDim = p.nHeads * headSize
  var maxDim = Math.max(p.dim, qDim)

  // Pre-compute RoPE frequency table (once per frequency index)
  var ropeSize = headSize / 2
  var seqLen = p.seqLen
  var ropeFreqs = new Array(ropeSize)
  for (var i = 0; i < ropeSize; i = i + 1) {
    ropeFreqs[i] = 1.0 / Math.pow(p.ropeTheta, (i * 2) / headSize)
  }

  // Pre-compute full RoPE sin/cos tables for all positions
  var ropeCosAll = new Float32Array(seqLen * ropeSize)
  var ropeSinAll = new Float32Array(seqLen * ropeSize)
  for (var pos = 0; pos < seqLen; pos = pos + 1) {
    var base = pos * ropeSize
    for (var i = 0; i < ropeSize; i = i + 1) {
      var val = pos * ropeFreqs[i]
      ropeCosAll[base + i] = Math.cos(val)
      ropeSinAll[base + i] = Math.sin(val)
    }
  }

  // Pre-compute SWA RoPE tables (only for Gemma models)
  var ropeCosSwaAll
  var ropeSinSwaAll
  if (p.isGemma) {
    var swaTheta = p.ropeThetaSwa > 0 ? p.ropeThetaSwa : 10000.0
    var swaFreqs = new Array(ropeSize)
    for (var i = 0; i < ropeSize; i = i + 1) {
      swaFreqs[i] = 1.0 / Math.pow(swaTheta, (i * 2) / headSize)
    }
    ropeCosSwaAll = new Float32Array(seqLen * ropeSize)
    ropeSinSwaAll = new Float32Array(seqLen * ropeSize)
    for (var pos = 0; pos < seqLen; pos = pos + 1) {
      var base = pos * ropeSize
      for (var i = 0; i < ropeSize; i = i + 1) {
        var val = pos * swaFreqs[i]
        ropeCosSwaAll[base + i] = Math.cos(val)
        ropeSinSwaAll[base + i] = Math.sin(val)
      }
    }
  } else {
    ropeCosSwaAll = ropeCosAll
    ropeSinSwaAll = ropeSinAll
  }

  // Q8_0 KV cache: 34 bytes per 32 floats
  // Per-head layout: [layer][kv_head][position][head_data] for cache locality
  var headBytesQ8 = (headSize >> 5) * Q8_0_BLOCK_SIZE
  var headSeqBytes = seqLen * headBytesQ8
  var kvCacheLayerBytes = p.nKvHeads * headSeqBytes
  var kvCacheTotalBytes = p.nLayers * kvCacheLayerBytes

  // Create ArrayBuffer for KV caches with both Uint8 and Int8 views
  var keyCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)
  var valueCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)

  // Allocate Q8_0 buffer for quantizing x in matmulQuantized
  var maxCols = Math.max(p.dim, qDim, p.hiddenDim)
  var xQ8Size = (maxCols >> 5) * 34
  var xQ8Buffer = new ArrayBuffer(xQ8Size)
  xQ8Buf = new Uint8Array(xQ8Buffer)
  xQ8Int8Buf = new Int8Array(xQ8Buffer)

  // Reusable buffer for batch dequantize-then-dot: 4 rows × maxCols (~192KB)
  matmulDeqBuf = new Float64Array(4 * maxCols)

  // Q8_0 buffer for quantized Q heads (for Q8 attention scoring)
  var qQ8TotalBytes = p.nHeads * headBytesQ8
  var qQ8Buffer = new ArrayBuffer(qQ8TotalBytes)

  // Pre-compute head offset tables to avoid repeated multiplication in attention loop
  var kvMul = p.nHeads / p.nKvHeads
  var headQOffsets = new Int32Array(p.nHeads)
  var headKvIdx = new Int32Array(p.nHeads)
  var headAttOffsets = new Int32Array(p.nHeads)
  var headKvByteOffsets = new Int32Array(p.nHeads)
  for (var h = 0; h < p.nHeads; h = h + 1) {
    headQOffsets[h] = h * headSize
    headKvIdx[h] = (h / kvMul) | 0
    headAttOffsets[h] = h * p.seqLen
    headKvByteOffsets[h] = ((h / kvMul) | 0) * headSeqBytes
  }

  // Pre-compute per-layer RoPE table references (avoid modulo check per layer)
  var ropeCosLayer = new Array(p.nLayers)
  var ropeSinLayer = new Array(p.nLayers)
  for (var l = 0; l < p.nLayers; l = l + 1) {
    var isSwa = p.swaPattern > 0 && l % p.swaPattern < p.swaPattern - 1
    if (p.isGemma && isSwa) {
      ropeCosLayer[l] = ropeCosSwaAll
      ropeSinLayer[l] = ropeSinSwaAll
    } else {
      ropeCosLayer[l] = ropeCosAll
      ropeSinLayer[l] = ropeSinAll
    }
  }

  // Batch buffers for prefill
  var batchX = new Array(PREFILL_BATCH_SIZE)
  var batchXb = new Array(PREFILL_BATCH_SIZE)
  var batchXb2 = new Array(PREFILL_BATCH_SIZE)
  var batchQ = new Array(PREFILL_BATCH_SIZE)
  var batchK = new Array(PREFILL_BATCH_SIZE)
  var batchV = new Array(PREFILL_BATCH_SIZE)
  var batchHb = new Array(PREFILL_BATCH_SIZE)
  var batchHb2 = new Array(PREFILL_BATCH_SIZE)
  var batchQ8 = new Array(PREFILL_BATCH_SIZE)
  var batchQ8i8 = new Array(PREFILL_BATCH_SIZE)
  for (var b = 0; b < PREFILL_BATCH_SIZE; b = b + 1) {
    batchX[b] = new Float32Array(p.dim)
    batchXb[b] = new Float32Array(maxDim)
    batchXb2[b] = new Float32Array(p.dim)
    batchQ[b] = new Float32Array(qDim)
    batchK[b] = new Float32Array(kvDim)
    batchV[b] = new Float32Array(kvDim)
    batchHb[b] = new Float32Array(p.hiddenDim)
    batchHb2[b] = new Float32Array(p.hiddenDim)
    var bQ8Buf = new ArrayBuffer(xQ8Size)
    batchQ8[b] = new Uint8Array(bQ8Buf)
    batchQ8i8[b] = new Int8Array(bQ8Buf)
  }

  return {
    x: new Float32Array(p.dim),
    xb: new Float32Array(maxDim),
    xb2: new Float32Array(p.dim),
    hb: new Float32Array(p.hiddenDim),
    hb2: new Float32Array(p.hiddenDim),
    q: new Float32Array(qDim),
    k: new Float32Array(kvDim),
    v: new Float32Array(kvDim),
    att: new Float32Array(p.nHeads * p.seqLen),
    logits: new Float32Array(p.vocabSize),
    // Q8_0 KV cache - Uint8 view for reading FP16 scale
    keyCache: new Uint8Array(keyCacheBuffer),
    valueCache: new Uint8Array(valueCacheBuffer),
    // Int8 views for reading quantized values
    keyCacheInt8: new Int8Array(keyCacheBuffer),
    valueCacheInt8: new Int8Array(valueCacheBuffer),
    // Q8_0 quantized Q heads for attention scoring
    qQ8: new Uint8Array(qQ8Buffer),
    qQ8i8: new Int8Array(qQ8Buffer),
    // Cache layout info
    headSeqBytes: headSeqBytes,
    // Pre-computed RoPE sin/cos for all positions
    ropeCosAll: ropeCosAll,
    ropeSinAll: ropeSinAll,
    ropeCosSwaAll: ropeCosSwaAll,
    ropeSinSwaAll: ropeSinSwaAll,
    ropeSize: ropeSize,
    // Per-layer RoPE table references (avoid modulo check per layer)
    ropeCosLayer: ropeCosLayer,
    ropeSinLayer: ropeSinLayer,
    // Cached constants to avoid recomputation in transformer
    headSize: headSize,
    kvDim: kvDim,
    qDim: qDim,
    kvMul: p.nHeads / p.nKvHeads,
    // Q8_0 cache: layer size in bytes (nKvHeads * seqLen * headBytesQ8)
    kvCacheLayerSize: kvCacheLayerBytes,
    attnScale: 1.0 / Math.sqrt(headSize),
    embedScale: Math.sqrt(p.dim),
    // Cache config values to avoid property lookups in hot loops
    dim: p.dim,
    nHeads: p.nHeads,
    nKvHeads: p.nKvHeads,
    nLayers: p.nLayers,
    seqLen: p.seqLen,
    hiddenDim: p.hiddenDim,
    vocabSize: p.vocabSize,
    isGemma: p.isGemma,
    rmsNormEps: p.rmsNormEps,
    // Pre-computed values for rmsnorm
    invDim: 1.0 / p.dim,
    invHeadSize: 1.0 / headSize,
    // SWA pattern for per-layer RoPE frequency selection
    swaPattern: p.swaPattern,
    // Pre-allocated buffers for top-k sampling
    topKIndices: new Int32Array(topK),
    topKValues: new Float32Array(topK),
    // Pre-computed head offset tables
    headQOffsets: headQOffsets,
    headKvIdx: headKvIdx,
    headAttOffsets: headAttOffsets,
    headKvByteOffsets: headKvByteOffsets,
    headBytesQ8: headBytesQ8,
    // Batch buffers for prefill
    batchX: batchX,
    batchXb: batchXb,
    batchXb2: batchXb2,
    batchQ: batchQ,
    batchK: batchK,
    batchV: batchV,
    batchHb: batchHb,
    batchHb2: batchHb2,
    batchQ8: batchQ8,
    batchQ8i8: batchQ8i8,
  }
}

// ----------------------------------------------------------------------------
// Transformer forward pass

// Llama-optimized transformer: fused attnScale in RoPE, SiLU via tanh,
// per-head KV layout, Q8 attention, GQA batching, pre-quantized matmul
function transformerLlama(token, pos, computeLogits) {
  var w = weights
  var s = state
  var dim = s.dim
  var headSize = s.headSize
  var kvDim = s.kvDim
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var nLayers = s.nLayers
  var nKvHeads = s.nKvHeads
  var invDim = s.invDim
  var kvMul = s.kvMul
  var headBytesQ8 = s.headBytesQ8
  var headSeqBytes = s.headSeqBytes
  var attnScale = s.attnScale
  var seqLen = s.seqLen

  var xArr = s.x
  var xbArr = s.xb
  var xb2Arr = s.xb2
  var qArr = s.q
  var kArr = s.k
  var vArr = s.v
  var sAtt = s.att
  var keyCache = s.keyCache
  var valueCache = s.valueCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCacheInt8 = s.valueCacheInt8
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8

  // Embedding
  var emb = w.tokenEmbedding
  dequantizeRow(xArr, emb.dataOffset + token * emb.rowSize, dim, emb.type)

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l]

    rmsnorm(xbArr, xArr, lw.rmsAttWeight, dim, invDim)

    // QKV matmuls - quantize once, reuse (#1+2)
    if (lw.wq.dotQ8Func && lw.wk.dotQ8Func && lw.wv.dotQ8Func) {
      quantizeToQ8_0Cache(xbArr, 0, xQ8Buf, xQ8Int8Buf, 0, dim)
      matmulQuantizedPreQ8(qArr, lw.wq)
      matmulQuantizedPreQ8(kArr, lw.wk)
      matmulQuantizedPreQ8(vArr, lw.wv)
    } else {
      matmulQuantized(qArr, xbArr, lw.wq)
      matmulQuantized(kArr, xbArr, lw.wk)
      matmulQuantized(vArr, xbArr, lw.wv)
    }

    // RoPE with fused attnScale on Q (#18)
    var half = headSize >> 1
    var ropeBase = pos * s.ropeSize
    var ropeCos = s.ropeCosLayer[l]
    var ropeSin = s.ropeSinLayer[l]

    var kvDim4 = kvDim & ~3
    for (var i = 0; i < kvDim4; i = i + 4) {
      var fi0 = (i >> 1) % half
      var fi1 = ((i + 2) >> 1) % half
      var fcr0 = ropeCos[ropeBase + fi0]
      var fci0 = ropeSin[ropeBase + fi0]
      var fcr1 = ropeCos[ropeBase + fi1]
      var fci1 = ropeSin[ropeBase + fi1]
      var qv0 = qArr[i]
      var qv1 = qArr[i + 1]
      qArr[i] = (qv0 * fcr0 - qv1 * fci0) * attnScale
      qArr[i + 1] = (qv0 * fci0 + qv1 * fcr0) * attnScale
      var qv2 = qArr[i + 2]
      var qv3 = qArr[i + 3]
      qArr[i + 2] = (qv2 * fcr1 - qv3 * fci1) * attnScale
      qArr[i + 3] = (qv2 * fci1 + qv3 * fcr1) * attnScale
      var kv0 = kArr[i]
      var kv1 = kArr[i + 1]
      kArr[i] = kv0 * fcr0 - kv1 * fci0
      kArr[i + 1] = kv0 * fci0 + kv1 * fcr0
      var kv2 = kArr[i + 2]
      var kv3 = kArr[i + 3]
      kArr[i + 2] = kv2 * fcr1 - kv3 * fci1
      kArr[i + 3] = kv2 * fci1 + kv3 * fcr1
    }
    for (var i = kvDim4; i < kvDim; i = i + 2) {
      var freqIdx = (i >> 1) % half
      var fcr = ropeCos[ropeBase + freqIdx]
      var fci = ropeSin[ropeBase + freqIdx]
      var v0 = qArr[i]
      var v1 = qArr[i + 1]
      qArr[i] = (v0 * fcr - v1 * fci) * attnScale
      qArr[i + 1] = (v0 * fci + v1 * fcr) * attnScale
      v0 = kArr[i]
      v1 = kArr[i + 1]
      kArr[i] = v0 * fcr - v1 * fci
      kArr[i + 1] = v0 * fci + v1 * fcr
    }
    for (var i = kvDim; i < qDim; i = i + 2) {
      var freqIdx = (i >> 1) % half
      var fcr = ropeCos[ropeBase + freqIdx]
      var fci = ropeSin[ropeBase + freqIdx]
      var v0 = qArr[i]
      var v1 = qArr[i + 1]
      qArr[i] = (v0 * fcr - v1 * fci) * attnScale
      qArr[i + 1] = (v0 * fci + v1 * fcr) * attnScale
    }

    // Per-head KV cache write (#20)
    var loff = l * s.kvCacheLayerSize
    for (var h = 0; h < nKvHeads; h = h + 1) {
      var headOff = loff + h * headSeqBytes + pos * headBytesQ8
      quantizeToQ8_0Cache(
        kArr,
        h * headSize,
        keyCache,
        keyCacheInt8,
        headOff,
        headSize
      )
      quantizeToQ8_0Cache(
        vArr,
        h * headSize,
        valueCache,
        valueCacheInt8,
        headOff,
        headSize
      )
    }

    // Quantize all Q heads to Q8_0 in one batch call (#15)
    quantizeToQ8_0Cache(qArr, 0, qQ8, qQ8i8, 0, qDim)

    xbArr.fill(0, 0, qDim)

    // GQA-batched attention (#16) with Q8 scoring (#15)
    // Loop reordered: position-first for K/V cache locality (#1)
    for (var kvH = 0; kvH < nKvHeads; kvH = kvH + 1) {
      var kBase = loff + kvH * headSeqBytes

      // Score all Q heads in this GQA group against all K positions
      for (var t = 0; t <= pos; t = t + 1) {
        var kOff = kBase + t * headBytesQ8
        for (var mh = 0; mh < kvMul; mh = mh + 1) {
          var h = kvH * kvMul + mh
          sAtt[h * seqLen + t] = dotQ8_0_Q8_0Cache(
            qQ8,
            qQ8i8,
            h * headBytesQ8,
            keyCache,
            keyCacheInt8,
            kOff,
            headSize
          )
        }
      }

      // Softmax + value accumulation per Q head
      for (var mh = 0; mh < kvMul; mh = mh + 1) {
        var h = kvH * kvMul + mh
        var attOffset = h * seqLen

        // Softmax
        var softmaxEnd = attOffset + pos
        var maxVal = sAtt[attOffset]
        for (var i = attOffset + 1; i <= softmaxEnd; i = i + 1) {
          if (sAtt[i] > maxVal) {
            maxVal = sAtt[i]
          }
        }
        var expSum = 0.0
        for (var i = attOffset; i <= softmaxEnd; i = i + 1) {
          var e = Math.exp(sAtt[i] - maxVal)
          sAtt[i] = e
          expSum = expSum + e
        }
        var invSum = 1.0 / expSum
        for (var i = attOffset; i <= softmaxEnd; i = i + 1) {
          sAtt[i] = sAtt[i] * invSum
        }

        // Value accumulation - position-first for V cache locality
        var xbOffset = h * headSize
        for (var t = 0; t <= pos; t = t + 1) {
          accumQ8_0Cache(
            xbArr,
            xbOffset,
            valueCache,
            valueCacheInt8,
            kBase + t * headBytesQ8,
            sAtt[attOffset + t],
            headSize
          )
        }
      }
    }

    // Attention output
    matmulQuantized(xb2Arr, xbArr, lw.wo)
    accum(xArr, xb2Arr, dim)

    // FFN
    rmsnorm(xbArr, xArr, lw.rmsFfnWeight, dim, invDim)

    var hbArr = s.hb
    var hb2Arr = s.hb2

    // FFN gate and up - quantize once, reuse (#1+2)
    if (lw.w1.dotQ8Func && lw.w3.dotQ8Func) {
      quantizeToQ8_0Cache(xbArr, 0, xQ8Buf, xQ8Int8Buf, 0, dim)
      matmulQuantizedPreQ8(hbArr, lw.w1)
      matmulQuantizedPreQ8(hb2Arr, lw.w3)
    } else {
      matmulQuantized(hbArr, xbArr, lw.w1)
      matmulQuantized(hb2Arr, xbArr, lw.w3)
    }

    // SiLU via tanh: silu(x) = 0.5 * x * (1 + tanh(x/2)) (#17)
    var hd4 = hiddenDim & ~3
    for (var i = 0; i < hd4; i = i + 4) {
      var v0 = hbArr[i]
      var v1 = hbArr[i + 1]
      var v2 = hbArr[i + 2]
      var v3 = hbArr[i + 3]
      hbArr[i] = 0.5 * v0 * (1.0 + fastTanh(0.5 * v0)) * hb2Arr[i]
      hbArr[i + 1] = 0.5 * v1 * (1.0 + fastTanh(0.5 * v1)) * hb2Arr[i + 1]
      hbArr[i + 2] = 0.5 * v2 * (1.0 + fastTanh(0.5 * v2)) * hb2Arr[i + 2]
      hbArr[i + 3] = 0.5 * v3 * (1.0 + fastTanh(0.5 * v3)) * hb2Arr[i + 3]
    }
    for (var i = hd4; i < hiddenDim; i = i + 1) {
      var val = hbArr[i]
      hbArr[i] = 0.5 * val * (1.0 + fastTanh(0.5 * val)) * hb2Arr[i]
    }

    // FFN down
    matmulQuantized(xbArr, hbArr, lw.w2)
    accum(xArr, xbArr, dim)
  }

  // Final norm
  rmsnorm(xArr, xArr, w.rmsFinalWeight, dim, invDim)

  // Classifier into logits
  if (computeLogits !== false) {
    matmulQuantized(s.logits, xArr, w.wcls)
  }
}

// Batched prefill transformer for Llama: processes multiple prompt tokens per
// layer pass, reading weight data once and reusing across all batch elements
function transformerPrefillLlama(allTokens, startPos, batchSize) {
  var w = weights
  var s = state
  var dim = s.dim
  var headSize = s.headSize
  var kvDim = s.kvDim
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var nLayers = s.nLayers
  var nKvHeads = s.nKvHeads
  var invDim = s.invDim
  var kvMul = s.kvMul
  var headBytesQ8 = s.headBytesQ8
  var headSeqBytes = s.headSeqBytes
  var attnScale = s.attnScale
  var seqLen = s.seqLen

  var keyCache = s.keyCache
  var valueCache = s.valueCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCacheInt8 = s.valueCacheInt8
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8
  var sAtt = s.att

  var bX = s.batchX
  var bXb = s.batchXb
  var bXb2 = s.batchXb2
  var bQ = s.batchQ
  var bK = s.batchK
  var bV = s.batchV
  var bHb = s.batchHb
  var bHb2 = s.batchHb2

  // Embed all tokens in batch
  var emb = w.tokenEmbedding
  for (var b = 0; b < batchSize; b = b + 1) {
    dequantizeRow(
      bX[b],
      emb.dataOffset + allTokens[startPos + b] * emb.rowSize,
      dim,
      emb.type
    )
  }

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l]

    // Batch rmsnorm
    for (var b = 0; b < batchSize; b = b + 1) {
      rmsnorm(bXb[b], bX[b], lw.rmsAttWeight, dim, invDim)
    }

    // Batch QKV matmuls - read weights once, compute for all batch elements
    matmulQuantizedBatch(bQ, bXb, lw.wq, batchSize)
    matmulQuantizedBatch(bK, bXb, lw.wk, batchSize)
    matmulQuantizedBatch(bV, bXb, lw.wv, batchSize)

    // Per-token: RoPE, KV cache write, attention
    var half = headSize >> 1
    var ropeCos = s.ropeCosLayer[l]
    var ropeSin = s.ropeSinLayer[l]
    var loff = l * s.kvCacheLayerSize

    for (var b = 0; b < batchSize; b = b + 1) {
      var pos = startPos + b
      var qArr = bQ[b]
      var kArr = bK[b]
      var vArr = bV[b]
      var xbArr = bXb[b]

      // RoPE with fused attnScale on Q
      var ropeBase = pos * s.ropeSize
      var kvDim4 = kvDim & ~3
      for (var i = 0; i < kvDim4; i = i + 4) {
        var fi0 = (i >> 1) % half
        var fi1 = ((i + 2) >> 1) % half
        var fcr0 = ropeCos[ropeBase + fi0]
        var fci0 = ropeSin[ropeBase + fi0]
        var fcr1 = ropeCos[ropeBase + fi1]
        var fci1 = ropeSin[ropeBase + fi1]
        var qv0 = qArr[i]
        var qv1 = qArr[i + 1]
        qArr[i] = (qv0 * fcr0 - qv1 * fci0) * attnScale
        qArr[i + 1] = (qv0 * fci0 + qv1 * fcr0) * attnScale
        var qv2 = qArr[i + 2]
        var qv3 = qArr[i + 3]
        qArr[i + 2] = (qv2 * fcr1 - qv3 * fci1) * attnScale
        qArr[i + 3] = (qv2 * fci1 + qv3 * fcr1) * attnScale
        var kv0 = kArr[i]
        var kv1 = kArr[i + 1]
        kArr[i] = kv0 * fcr0 - kv1 * fci0
        kArr[i + 1] = kv0 * fci0 + kv1 * fcr0
        var kv2 = kArr[i + 2]
        var kv3 = kArr[i + 3]
        kArr[i + 2] = kv2 * fcr1 - kv3 * fci1
        kArr[i + 3] = kv2 * fci1 + kv3 * fcr1
      }
      for (var i = kvDim4; i < kvDim; i = i + 2) {
        var freqIdx = (i >> 1) % half
        var fcr = ropeCos[ropeBase + freqIdx]
        var fci = ropeSin[ropeBase + freqIdx]
        var v0 = qArr[i]
        var v1 = qArr[i + 1]
        qArr[i] = (v0 * fcr - v1 * fci) * attnScale
        qArr[i + 1] = (v0 * fci + v1 * fcr) * attnScale
        v0 = kArr[i]
        v1 = kArr[i + 1]
        kArr[i] = v0 * fcr - v1 * fci
        kArr[i + 1] = v0 * fci + v1 * fcr
      }
      for (var i = kvDim; i < qDim; i = i + 2) {
        var freqIdx = (i >> 1) % half
        var fcr = ropeCos[ropeBase + freqIdx]
        var fci = ropeSin[ropeBase + freqIdx]
        var v0 = qArr[i]
        var v1 = qArr[i + 1]
        qArr[i] = (v0 * fcr - v1 * fci) * attnScale
        qArr[i + 1] = (v0 * fci + v1 * fcr) * attnScale
      }

      // Per-head KV cache write
      for (var h = 0; h < nKvHeads; h = h + 1) {
        var headOff = loff + h * headSeqBytes + pos * headBytesQ8
        quantizeToQ8_0Cache(
          kArr,
          h * headSize,
          keyCache,
          keyCacheInt8,
          headOff,
          headSize
        )
        quantizeToQ8_0Cache(
          vArr,
          h * headSize,
          valueCache,
          valueCacheInt8,
          headOff,
          headSize
        )
      }

      // Quantize all Q heads to Q8_0
      quantizeToQ8_0Cache(qArr, 0, qQ8, qQ8i8, 0, qDim)

      xbArr.fill(0, 0, qDim)

      // GQA-batched attention with Q8 scoring
      for (var kvH = 0; kvH < nKvHeads; kvH = kvH + 1) {
        var kBase = loff + kvH * headSeqBytes

        for (var t = 0; t <= pos; t = t + 1) {
          var kOff = kBase + t * headBytesQ8
          for (var mh = 0; mh < kvMul; mh = mh + 1) {
            var h = kvH * kvMul + mh
            sAtt[h * seqLen + t] = dotQ8_0_Q8_0Cache(
              qQ8,
              qQ8i8,
              h * headBytesQ8,
              keyCache,
              keyCacheInt8,
              kOff,
              headSize
            )
          }
        }

        for (var mh = 0; mh < kvMul; mh = mh + 1) {
          var h = kvH * kvMul + mh
          var attOffset = h * seqLen
          var softmaxEnd = attOffset + pos
          var maxVal = sAtt[attOffset]
          for (var i = attOffset + 1; i <= softmaxEnd; i = i + 1) {
            if (sAtt[i] > maxVal) {
              maxVal = sAtt[i]
            }
          }
          var expSum = 0.0
          for (var i = attOffset; i <= softmaxEnd; i = i + 1) {
            var e = Math.exp(sAtt[i] - maxVal)
            sAtt[i] = e
            expSum = expSum + e
          }
          var invSum = 1.0 / expSum
          for (var i = attOffset; i <= softmaxEnd; i = i + 1) {
            sAtt[i] = sAtt[i] * invSum
          }

          var xbOffset = h * headSize
          for (var t = 0; t <= pos; t = t + 1) {
            accumQ8_0Cache(
              xbArr,
              xbOffset,
              valueCache,
              valueCacheInt8,
              kBase + t * headBytesQ8,
              sAtt[attOffset + t],
              headSize
            )
          }
        }
      }
    }

    // Batch wo matmul
    matmulQuantizedBatch(bXb2, bXb, lw.wo, batchSize)
    for (var b = 0; b < batchSize; b = b + 1) {
      accum(bX[b], bXb2[b], dim)
    }

    // Batch FFN rmsnorm
    for (var b = 0; b < batchSize; b = b + 1) {
      rmsnorm(bXb[b], bX[b], lw.rmsFfnWeight, dim, invDim)
    }

    // Batch FFN matmuls
    matmulQuantizedBatch(bHb, bXb, lw.w1, batchSize)
    matmulQuantizedBatch(bHb2, bXb, lw.w3, batchSize)

    // Per-token SiLU activation
    var hd4 = hiddenDim & ~3
    for (var b = 0; b < batchSize; b = b + 1) {
      var hbArr = bHb[b]
      var hb2Arr = bHb2[b]
      for (var i = 0; i < hd4; i = i + 4) {
        var v0 = hbArr[i]
        var v1 = hbArr[i + 1]
        var v2 = hbArr[i + 2]
        var v3 = hbArr[i + 3]
        hbArr[i] = 0.5 * v0 * (1.0 + fastTanh(0.5 * v0)) * hb2Arr[i]
        hbArr[i + 1] = 0.5 * v1 * (1.0 + fastTanh(0.5 * v1)) * hb2Arr[i + 1]
        hbArr[i + 2] = 0.5 * v2 * (1.0 + fastTanh(0.5 * v2)) * hb2Arr[i + 2]
        hbArr[i + 3] = 0.5 * v3 * (1.0 + fastTanh(0.5 * v3)) * hb2Arr[i + 3]
      }
      for (var i = hd4; i < hiddenDim; i = i + 1) {
        var val = hbArr[i]
        hbArr[i] = 0.5 * val * (1.0 + fastTanh(0.5 * val)) * hb2Arr[i]
      }
    }

    // Batch FFN down matmul
    matmulQuantizedBatch(bXb, bHb, lw.w2, batchSize)
    for (var b = 0; b < batchSize; b = b + 1) {
      accum(bX[b], bXb[b], dim)
    }
  }
}

// Gemma-optimized transformer: per-head KV layout, Q8 attention,
// GQA batching, pre-quantized matmul, SWA, QK norms, post-norms
function transformerGemma(token, pos, computeLogits) {
  var w = weights
  var s = state
  var dim = s.dim
  var headSize = s.headSize
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var eps = s.rmsNormEps
  var nLayers = s.nLayers
  var nHeads = s.nHeads
  var nKvHeads = s.nKvHeads
  var invDim = s.invDim
  var invHeadSize = s.invHeadSize
  var kvMul = s.kvMul
  var headBytesQ8 = s.headBytesQ8
  var headSeqBytes = s.headSeqBytes
  var attnScale = s.attnScale
  var seqLen = s.seqLen

  var xArr = s.x
  var xbArr = s.xb
  var xb2Arr = s.xb2
  var qArr = s.q
  var kArr = s.k
  var vArr = s.v
  var sAtt = s.att
  var keyCache = s.keyCache
  var valueCache = s.valueCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCacheInt8 = s.valueCacheInt8
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8

  // Embedding (scaling fused into first rmsnorm)
  var emb = w.tokenEmbedding
  dequantizeRow(xArr, emb.dataOffset + token * emb.rowSize, dim, emb.type)

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l]

    if (l === 0) {
      // First layer: fused embed scale + rmsnorm (3 passes → 2)
      rmsnormGemmaFusedScale(
        xbArr,
        xArr,
        lw.rmsAttWeight,
        dim,
        eps,
        invDim,
        s.embedScale
      )
    } else {
      rmsnormGemma(xbArr, xArr, lw.rmsAttWeight, dim, eps, invDim)
    }

    // QKV matmuls - quantize once, reuse (#1+2)
    if (lw.wq.dotQ8Func && lw.wk.dotQ8Func && lw.wv.dotQ8Func) {
      quantizeToQ8_0Cache(xbArr, 0, xQ8Buf, xQ8Int8Buf, 0, dim)
      matmulQuantizedPreQ8(qArr, lw.wq)
      matmulQuantizedPreQ8(kArr, lw.wk)
      matmulQuantizedPreQ8(vArr, lw.wv)
    } else {
      matmulQuantized(qArr, xbArr, lw.wq)
      matmulQuantized(kArr, xbArr, lw.wk)
      matmulQuantized(vArr, xbArr, lw.wv)
    }

    // Gemma QK norms
    if (lw.attnQNorm && lw.attnKNorm) {
      for (var h = 0; h < nHeads; h = h + 1) {
        rmsnormGemmaAt(qArr, h * headSize, lw.attnQNorm, headSize, eps, invHeadSize)
      }
      for (var h = 0; h < nKvHeads; h = h + 1) {
        rmsnormGemmaAt(kArr, h * headSize, lw.attnKNorm, headSize, eps, invHeadSize)
      }
    }

    // Fused RoPE + Q attention scaling
    var half = headSize >> 1
    var ropeBase = pos * s.ropeSize
    var ropeCos = s.ropeCosLayer[l]
    var ropeSin = s.ropeSinLayer[l]

    for (var h = 0; h < nHeads; h = h + 1) {
      var idx = h * headSize
      for (var i = 0; i < half; i = i + 1) {
        var fcr = ropeCos[ropeBase + i]
        var fci = ropeSin[ropeBase + i]
        var v0 = qArr[idx + i]
        var v1 = qArr[idx + i + half]
        qArr[idx + i] = (v0 * fcr - v1 * fci) * attnScale
        qArr[idx + i + half] = (v0 * fci + v1 * fcr) * attnScale
      }
    }
    for (var h = 0; h < nKvHeads; h = h + 1) {
      var idx = h * headSize
      for (var i = 0; i < half; i = i + 1) {
        var fcr = ropeCos[ropeBase + i]
        var fci = ropeSin[ropeBase + i]
        var v0 = kArr[idx + i]
        var v1 = kArr[idx + i + half]
        kArr[idx + i] = v0 * fcr - v1 * fci
        kArr[idx + i + half] = v0 * fci + v1 * fcr
      }
    }

    // Per-head KV cache write (#20)
    var loff = l * s.kvCacheLayerSize
    for (var h = 0; h < nKvHeads; h = h + 1) {
      var headOff = loff + h * headSeqBytes + pos * headBytesQ8
      quantizeToQ8_0Cache(
        kArr,
        h * headSize,
        keyCache,
        keyCacheInt8,
        headOff,
        headSize
      )
      quantizeToQ8_0Cache(
        vArr,
        h * headSize,
        valueCache,
        valueCacheInt8,
        headOff,
        headSize
      )
    }

    // Quantize all Q heads to Q8_0 in one batch call (#15)
    quantizeToQ8_0Cache(qArr, 0, qQ8, qQ8i8, 0, qDim)

    xbArr.fill(0, 0, qDim)

    // SWA window enforcement
    var isSwaLayer = s.swaPattern > 0 && l % s.swaPattern < s.swaPattern - 1
    var startT =
      isSwaLayer && config.swaWindow > 0
        ? Math.max(0, pos - config.swaWindow + 1)
        : 0

    // GQA-batched attention (#16) with Q8 scoring (#15)
    // Loop reordered: position-first for K/V cache locality (#1)
    for (var kvH = 0; kvH < nKvHeads; kvH = kvH + 1) {
      var kBase = loff + kvH * headSeqBytes

      // Score all Q heads in this GQA group against all K positions
      for (var t = startT; t <= pos; t = t + 1) {
        var kOff = kBase + t * headBytesQ8
        for (var mh = 0; mh < kvMul; mh = mh + 1) {
          var h = kvH * kvMul + mh
          sAtt[h * seqLen + t] = dotQ8_0_Q8_0Cache(
            qQ8,
            qQ8i8,
            h * headBytesQ8,
            keyCache,
            keyCacheInt8,
            kOff,
            headSize
          )
        }
      }

      // Softmax + value accumulation per Q head
      for (var mh = 0; mh < kvMul; mh = mh + 1) {
        var h = kvH * kvMul + mh
        var attOffset = h * seqLen

        // Softmax
        var softmaxStart = attOffset + startT
        var softmaxEnd = attOffset + pos
        var maxVal = sAtt[softmaxStart]
        for (var i = softmaxStart + 1; i <= softmaxEnd; i = i + 1) {
          if (sAtt[i] > maxVal) {
            maxVal = sAtt[i]
          }
        }
        var expSum = 0.0
        for (var i = softmaxStart; i <= softmaxEnd; i = i + 1) {
          var e = Math.exp(sAtt[i] - maxVal)
          sAtt[i] = e
          expSum = expSum + e
        }
        var invSum = 1.0 / expSum
        for (var i = softmaxStart; i <= softmaxEnd; i = i + 1) {
          sAtt[i] = sAtt[i] * invSum
        }

        // Value accumulation - position-first for V cache locality
        var xbOffset = h * headSize
        for (var t = startT; t <= pos; t = t + 1) {
          accumQ8_0Cache(
            xbArr,
            xbOffset,
            valueCache,
            valueCacheInt8,
            kBase + t * headBytesQ8,
            sAtt[attOffset + t],
            headSize
          )
        }
      }
    }

    // Attention output
    matmulQuantized(xb2Arr, xbArr, lw.wo)

    if (lw.attnPostNorm) {
      rmsnormGemma(xb2Arr, xb2Arr, lw.attnPostNorm, dim, eps, invDim)
    }

    accum(xArr, xb2Arr, dim)

    // FFN
    rmsnormGemma(xbArr, xArr, lw.rmsFfnWeight, dim, eps, invDim)

    var hbArr = s.hb
    var hb2Arr = s.hb2

    // FFN gate and up - quantize once, reuse (#1+2)
    if (lw.w1.dotQ8Func && lw.w3.dotQ8Func) {
      quantizeToQ8_0Cache(xbArr, 0, xQ8Buf, xQ8Int8Buf, 0, dim)
      matmulQuantizedPreQ8(hbArr, lw.w1)
      matmulQuantizedPreQ8(hb2Arr, lw.w3)
    } else {
      matmulQuantized(hbArr, xbArr, lw.w1)
      matmulQuantized(hb2Arr, xbArr, lw.w3)
    }

    // GELU activation
    var hd4 = hiddenDim & ~3
    var GELU_A = 0.7978845608
    var GELU_B = 0.035677408137
    for (var i = 0; i < hd4; i = i + 4) {
      var x0 = hbArr[i]
      var x1 = hbArr[i + 1]
      var x2 = hbArr[i + 2]
      var x3 = hbArr[i + 3]
      hbArr[i] =
        0.5 * x0 * (1.0 + fastTanh(x0 * (GELU_A + GELU_B * x0 * x0))) * hb2Arr[i]
      hbArr[i + 1] =
        0.5 * x1 * (1.0 + fastTanh(x1 * (GELU_A + GELU_B * x1 * x1))) * hb2Arr[i + 1]
      hbArr[i + 2] =
        0.5 * x2 * (1.0 + fastTanh(x2 * (GELU_A + GELU_B * x2 * x2))) * hb2Arr[i + 2]
      hbArr[i + 3] =
        0.5 * x3 * (1.0 + fastTanh(x3 * (GELU_A + GELU_B * x3 * x3))) * hb2Arr[i + 3]
    }
    for (var i = hd4; i < hiddenDim; i = i + 1) {
      var x = hbArr[i]
      hbArr[i] =
        0.5 * x * (1.0 + fastTanh(x * (GELU_A + GELU_B * x * x))) * hb2Arr[i]
    }

    // FFN down
    matmulQuantized(xbArr, hbArr, lw.w2)

    if (lw.ffnPostNorm) {
      rmsnormGemma(xbArr, xbArr, lw.ffnPostNorm, dim, eps, invDim)
    }

    accum(xArr, xbArr, dim)
  }

  // Final norm
  rmsnormGemma(xArr, xArr, w.rmsFinalWeight, dim, eps, invDim)

  // Classifier into logits
  if (computeLogits !== false) {
    matmulQuantized(s.logits, xArr, w.wcls)

    if (config.finalLogitSoftcapping > 0) {
      var cap = config.finalLogitSoftcapping
      var vocabSize = s.vocabSize
      for (var i = 0; i < vocabSize; i = i + 1) {
        s.logits[i] = cap * fastTanh(s.logits[i] / cap)
      }
    }
  }
}

// Batched prefill transformer for Gemma: same batching strategy with
// Gemma-specific features (QK norms, NEOX RoPE, SWA, GELU, post-norms)
function transformerPrefillGemma(allTokens, startPos, batchSize) {
  var w = weights
  var s = state
  var dim = s.dim
  var headSize = s.headSize
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var eps = s.rmsNormEps
  var nLayers = s.nLayers
  var nHeads = s.nHeads
  var nKvHeads = s.nKvHeads
  var invDim = s.invDim
  var invHeadSize = s.invHeadSize
  var kvMul = s.kvMul
  var headBytesQ8 = s.headBytesQ8
  var headSeqBytes = s.headSeqBytes
  var attnScale = s.attnScale
  var seqLen = s.seqLen

  var keyCache = s.keyCache
  var valueCache = s.valueCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCacheInt8 = s.valueCacheInt8
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8
  var sAtt = s.att

  var bX = s.batchX
  var bXb = s.batchXb
  var bXb2 = s.batchXb2
  var bQ = s.batchQ
  var bK = s.batchK
  var bV = s.batchV
  var bHb = s.batchHb
  var bHb2 = s.batchHb2

  // Embed all tokens in batch (scaling fused into first rmsnorm)
  var emb = w.tokenEmbedding
  for (var b = 0; b < batchSize; b = b + 1) {
    dequantizeRow(
      bX[b],
      emb.dataOffset + allTokens[startPos + b] * emb.rowSize,
      dim,
      emb.type
    )
  }

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l]

    // Batch rmsnorm (fused embed scale for first layer)
    if (l === 0) {
      for (var b = 0; b < batchSize; b = b + 1) {
        rmsnormGemmaFusedScale(
          bXb[b],
          bX[b],
          lw.rmsAttWeight,
          dim,
          eps,
          invDim,
          s.embedScale
        )
      }
    } else {
      for (var b = 0; b < batchSize; b = b + 1) {
        rmsnormGemma(bXb[b], bX[b], lw.rmsAttWeight, dim, eps, invDim)
      }
    }

    // Batch QKV matmuls
    matmulQuantizedBatch(bQ, bXb, lw.wq, batchSize)
    matmulQuantizedBatch(bK, bXb, lw.wk, batchSize)
    matmulQuantizedBatch(bV, bXb, lw.wv, batchSize)

    // Per-token: QK norms, RoPE, KV cache write, attention
    var half = headSize >> 1
    var ropeCos = s.ropeCosLayer[l]
    var ropeSin = s.ropeSinLayer[l]
    var loff = l * s.kvCacheLayerSize

    // SWA window enforcement
    var isSwaLayer = s.swaPattern > 0 && l % s.swaPattern < s.swaPattern - 1

    for (var b = 0; b < batchSize; b = b + 1) {
      var pos = startPos + b
      var qArr = bQ[b]
      var kArr = bK[b]
      var vArr = bV[b]
      var xbArr = bXb[b]

      // Gemma QK norms
      if (lw.attnQNorm && lw.attnKNorm) {
        for (var h = 0; h < nHeads; h = h + 1) {
          rmsnormGemmaAt(
            qArr,
            h * headSize,
            lw.attnQNorm,
            headSize,
            eps,
            invHeadSize
          )
        }
        for (var h = 0; h < nKvHeads; h = h + 1) {
          rmsnormGemmaAt(
            kArr,
            h * headSize,
            lw.attnKNorm,
            headSize,
            eps,
            invHeadSize
          )
        }
      }

      // NEOX RoPE with fused attnScale on Q
      var ropeBase = pos * s.ropeSize
      for (var h = 0; h < nHeads; h = h + 1) {
        var idx = h * headSize
        for (var i = 0; i < half; i = i + 1) {
          var fcr = ropeCos[ropeBase + i]
          var fci = ropeSin[ropeBase + i]
          var v0 = qArr[idx + i]
          var v1 = qArr[idx + i + half]
          qArr[idx + i] = (v0 * fcr - v1 * fci) * attnScale
          qArr[idx + i + half] = (v0 * fci + v1 * fcr) * attnScale
        }
      }
      for (var h = 0; h < nKvHeads; h = h + 1) {
        var idx = h * headSize
        for (var i = 0; i < half; i = i + 1) {
          var fcr = ropeCos[ropeBase + i]
          var fci = ropeSin[ropeBase + i]
          var v0 = kArr[idx + i]
          var v1 = kArr[idx + i + half]
          kArr[idx + i] = v0 * fcr - v1 * fci
          kArr[idx + i + half] = v0 * fci + v1 * fcr
        }
      }

      // Per-head KV cache write
      for (var h = 0; h < nKvHeads; h = h + 1) {
        var headOff = loff + h * headSeqBytes + pos * headBytesQ8
        quantizeToQ8_0Cache(
          kArr,
          h * headSize,
          keyCache,
          keyCacheInt8,
          headOff,
          headSize
        )
        quantizeToQ8_0Cache(
          vArr,
          h * headSize,
          valueCache,
          valueCacheInt8,
          headOff,
          headSize
        )
      }

      // Quantize all Q heads to Q8_0
      quantizeToQ8_0Cache(qArr, 0, qQ8, qQ8i8, 0, qDim)

      xbArr.fill(0, 0, qDim)

      // SWA start position
      var startT =
        isSwaLayer && config.swaWindow > 0
          ? Math.max(0, pos - config.swaWindow + 1)
          : 0

      // GQA-batched attention with Q8 scoring
      for (var kvH = 0; kvH < nKvHeads; kvH = kvH + 1) {
        var kBase = loff + kvH * headSeqBytes

        for (var t = startT; t <= pos; t = t + 1) {
          var kOff = kBase + t * headBytesQ8
          for (var mh = 0; mh < kvMul; mh = mh + 1) {
            var h = kvH * kvMul + mh
            sAtt[h * seqLen + t] = dotQ8_0_Q8_0Cache(
              qQ8,
              qQ8i8,
              h * headBytesQ8,
              keyCache,
              keyCacheInt8,
              kOff,
              headSize
            )
          }
        }

        for (var mh = 0; mh < kvMul; mh = mh + 1) {
          var h = kvH * kvMul + mh
          var attOffset = h * seqLen

          var softmaxStart = attOffset + startT
          var softmaxEnd = attOffset + pos
          var maxVal = sAtt[softmaxStart]
          for (var i = softmaxStart + 1; i <= softmaxEnd; i = i + 1) {
            if (sAtt[i] > maxVal) {
              maxVal = sAtt[i]
            }
          }
          var expSum = 0.0
          for (var i = softmaxStart; i <= softmaxEnd; i = i + 1) {
            var e = Math.exp(sAtt[i] - maxVal)
            sAtt[i] = e
            expSum = expSum + e
          }
          var invSum = 1.0 / expSum
          for (var i = softmaxStart; i <= softmaxEnd; i = i + 1) {
            sAtt[i] = sAtt[i] * invSum
          }

          var xbOffset = h * headSize
          for (var t = startT; t <= pos; t = t + 1) {
            accumQ8_0Cache(
              xbArr,
              xbOffset,
              valueCache,
              valueCacheInt8,
              kBase + t * headBytesQ8,
              sAtt[attOffset + t],
              headSize
            )
          }
        }
      }
    }

    // Batch wo matmul
    matmulQuantizedBatch(bXb2, bXb, lw.wo, batchSize)

    // Post-attention norm (per-token)
    if (lw.attnPostNorm) {
      for (var b = 0; b < batchSize; b = b + 1) {
        rmsnormGemma(bXb2[b], bXb2[b], lw.attnPostNorm, dim, eps, invDim)
      }
    }

    for (var b = 0; b < batchSize; b = b + 1) {
      accum(bX[b], bXb2[b], dim)
    }

    // Batch FFN rmsnorm
    for (var b = 0; b < batchSize; b = b + 1) {
      rmsnormGemma(bXb[b], bX[b], lw.rmsFfnWeight, dim, eps, invDim)
    }

    // Batch FFN matmuls
    matmulQuantizedBatch(bHb, bXb, lw.w1, batchSize)
    matmulQuantizedBatch(bHb2, bXb, lw.w3, batchSize)

    // Per-token GELU activation
    var hd4 = hiddenDim & ~3
    var GELU_A = 0.7978845608
    var GELU_B = 0.035677408137
    for (var b = 0; b < batchSize; b = b + 1) {
      var hbArr = bHb[b]
      var hb2Arr = bHb2[b]
      for (var i = 0; i < hd4; i = i + 4) {
        var x0 = hbArr[i]
        var x1 = hbArr[i + 1]
        var x2 = hbArr[i + 2]
        var x3 = hbArr[i + 3]
        hbArr[i] =
          0.5 * x0 * (1.0 + fastTanh(x0 * (GELU_A + GELU_B * x0 * x0))) * hb2Arr[i]
        hbArr[i + 1] =
          0.5 *
          x1 *
          (1.0 + fastTanh(x1 * (GELU_A + GELU_B * x1 * x1))) *
          hb2Arr[i + 1]
        hbArr[i + 2] =
          0.5 *
          x2 *
          (1.0 + fastTanh(x2 * (GELU_A + GELU_B * x2 * x2))) *
          hb2Arr[i + 2]
        hbArr[i + 3] =
          0.5 *
          x3 *
          (1.0 + fastTanh(x3 * (GELU_A + GELU_B * x3 * x3))) *
          hb2Arr[i + 3]
      }
      for (var i = hd4; i < hiddenDim; i = i + 1) {
        var x = hbArr[i]
        hbArr[i] =
          0.5 * x * (1.0 + fastTanh(x * (GELU_A + GELU_B * x * x))) * hb2Arr[i]
      }
    }

    // Batch FFN down matmul
    matmulQuantizedBatch(bXb, bHb, lw.w2, batchSize)

    // Post-FFN norm (per-token)
    if (lw.ffnPostNorm) {
      for (var b = 0; b < batchSize; b = b + 1) {
        rmsnormGemma(bXb[b], bXb[b], lw.ffnPostNorm, dim, eps, invDim)
      }
    }

    for (var b = 0; b < batchSize; b = b + 1) {
      accum(bX[b], bXb[b], dim)
    }
  }
}

// Dispatch to model-specific transformer (#21)
function transformer(token, pos, computeLogits) {
  if (state.isGemma) {
    transformerGemma(token, pos, computeLogits)
  } else {
    transformerLlama(token, pos, computeLogits)
  }
}

// Dispatch to model-specific batched prefill
function transformerPrefill(allTokens, startPos, batchSize) {
  if (state.isGemma) {
    transformerPrefillGemma(allTokens, startPos, batchSize)
  } else {
    transformerPrefillLlama(allTokens, startPos, batchSize)
  }
}

// ----------------------------------------------------------------------------
// Sampling

function randomF32() {
  // Use JavaScript's built-in Math.random() for simplicity
  return Math.random()
}

function sampleArgmax(logits, n) {
  var maxI = 0
  var maxP = logits[0]
  for (var i = 1; i < n; i = i + 1) {
    if (logits[i] > maxP) {
      maxI = i
      maxP = logits[i]
    }
  }
  return maxI
}

function sample(logits, temp) {
  if (temp === 0.0) {
    return sampleArgmax(logits, config.vocabSize)
  }

  var vocabSize = config.vocabSize
  var k = topK
  if (k > vocabSize) {
    k = vocabSize
  }
  var topKIdx = state.topKIndices
  var topKVal = state.topKValues

  // Initialize with first k logits
  for (var i = 0; i < k; i = i + 1) {
    topKIdx[i] = i
    topKVal[i] = logits[i]
  }

  // Find current min in top-k
  var minPos = 0
  var minVal = topKVal[0]
  for (var i = 1; i < k; i = i + 1) {
    if (topKVal[i] < minVal) {
      minVal = topKVal[i]
      minPos = i
    }
  }

  // Scan rest of vocab, replacing min when larger found
  for (var i = k; i < vocabSize; i = i + 1) {
    if (logits[i] > minVal) {
      topKVal[minPos] = logits[i]
      topKIdx[minPos] = i
      // Rescan for new min starting from replaced value as bound
      minVal = topKVal[0]
      minPos = 0
      for (var j = 1; j < k; j = j + 1) {
        if (topKVal[j] < minVal) {
          minVal = topKVal[j]
          minPos = j
        }
      }
    }
  }

  // Apply temperature and fused max+softmax over just k values
  var invTemp = 1.0 / temp
  var maxV = topKVal[0]
  for (var i = 1; i < k; i = i + 1) {
    if (topKVal[i] > maxV) {
      maxV = topKVal[i]
    }
  }
  var maxVT = maxV * invTemp
  var sum = 0.0
  for (var i = 0; i < k; i = i + 1) {
    var e = Math.exp(topKVal[i] * invTemp - maxVT)
    topKVal[i] = e
    sum = sum + e
  }

  // Apply top-P (nucleus) filtering: sort by probability descending,
  // then keep only tokens whose cumulative probability reaches topP
  var n = k
  if (topP < 1.0) {
    // Insertion sort by probability descending (k is small, ~40)
    for (var i = 1; i < k; i = i + 1) {
      var keyVal = topKVal[i]
      var keyIdx = topKIdx[i]
      var j = i - 1
      while (j >= 0 && topKVal[j] < keyVal) {
        topKVal[j + 1] = topKVal[j]
        topKIdx[j + 1] = topKIdx[j]
        j = j - 1
      }
      topKVal[j + 1] = keyVal
      topKIdx[j + 1] = keyIdx
    }

    // Accumulate probabilities until we reach the topP threshold
    var cumSum = 0.0
    var threshold = topP * sum
    n = k
    for (var i = 0; i < k; i = i + 1) {
      cumSum = cumSum + topKVal[i]
      if (cumSum >= threshold) {
        n = i + 1
        break
      }
    }

    // Recompute sum over the kept tokens
    sum = 0.0
    for (var i = 0; i < n; i = i + 1) {
      sum = sum + topKVal[i]
    }
  }

  // Sample from the filtered distribution
  var r = randomF32() * sum
  var cdf = 0.0
  for (var i = 0; i < n; i = i + 1) {
    cdf = cdf + topKVal[i]
    if (r < cdf) {
      return topKIdx[i]
    }
  }
  return topKIdx[n - 1]
}

// ----------------------------------------------------------------------------
// Tokenizer

var vocabMap = null
var vocabTrie = null

function buildSortedVocab() {
  if (vocabTrie) {
    return
  }

  vocabMap = {}
  // Build trie for O(max_token_len) longest-match lookup
  vocabTrie = {}

  for (var i = 0; i < tokenizer.vocab.length; i = i + 1) {
    var token = tokenizer.vocab[i]
    if (token && token.length > 0) {
      vocabMap[token] = i
      // Insert into trie
      var node = vocabTrie
      for (var j = 0; j < token.length; j = j + 1) {
        var ch = token.charCodeAt(j)
        if (!node[ch]) {
          node[ch] = {}
        }
        node = node[ch]
      }
      node.id = i
      node.len = token.length
    }
  }
}

function findSpecialToken(tokenStr) {
  buildSortedVocab()
  if (vocabMap && Object.prototype.hasOwnProperty.call(vocabMap, tokenStr)) {
    return vocabMap[tokenStr]
  }
  return -1
}

// Build tiktoken byte-to-unicode mapping (OpenAI's bytes_to_unicode)
var tiktokenByteToUnicode = null

function buildTiktokenByteToUnicodeMap() {
  if (tiktokenByteToUnicode) {
    return
  }
  tiktokenByteToUnicode = {}

  // This is OpenAI's bytes_to_unicode() function
  // Printable ASCII and some extended chars map to themselves
  var n = 0
  for (var b = 0; b < 256; b = b + 1) {
    // These byte ranges map directly: ! to ~, ¡ to ¬, ® to ÿ
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
      tiktokenByteToUnicode[b] = b
    } else {
      // Other bytes (0-32, 127-160, 173) map to 256+n, 257+n, etc.
      tiktokenByteToUnicode[b] = 256 + n
      n = n + 1
    }
  }
}

function textToTiktoken(text) {
  buildTiktokenByteToUnicodeMap()

  var parts = []
  for (var i = 0; i < text.length; i = i + 1) {
    var code = text.charCodeAt(i)

    // Handle UTF-16 surrogate pairs
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      var low = text.charCodeAt(i + 1)
      if (low >= 0xdc00 && low <= 0xdfff) {
        code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00)
        i = i + 1
      }
    }

    // Convert unicode to UTF-8 bytes, then map each byte to tiktoken unicode
    if (code < 0x80) {
      parts.push(String.fromCharCode(tiktokenByteToUnicode[code]))
    } else if (code < 0x800) {
      parts.push(String.fromCharCode(tiktokenByteToUnicode[0xc0 | (code >> 6)]))
      parts.push(String.fromCharCode(tiktokenByteToUnicode[0x80 | (code & 0x3f)]))
    } else if (code < 0x10000) {
      parts.push(String.fromCharCode(tiktokenByteToUnicode[0xe0 | (code >> 12)]))
      parts.push(
        String.fromCharCode(tiktokenByteToUnicode[0x80 | ((code >> 6) & 0x3f)])
      )
      parts.push(String.fromCharCode(tiktokenByteToUnicode[0x80 | (code & 0x3f)]))
    } else {
      parts.push(String.fromCharCode(tiktokenByteToUnicode[0xf0 | (code >> 18)]))
      parts.push(
        String.fromCharCode(tiktokenByteToUnicode[0x80 | ((code >> 12) & 0x3f)])
      )
      parts.push(
        String.fromCharCode(tiktokenByteToUnicode[0x80 | ((code >> 6) & 0x3f)])
      )
      parts.push(String.fromCharCode(tiktokenByteToUnicode[0x80 | (code & 0x3f)]))
    }
  }
  return parts.join("")
}

function textToSentencePiece(text) {
  var parts = []
  var needPrefix = true // Add ▁ before first alphanumeric char

  for (var i = 0; i < text.length; i = i + 1) {
    var c = text.charAt(i)
    var code = text.charCodeAt(i)

    if (c === " ") {
      // Space -> ▁ (U+2581)
      parts.push("\u2581")
      needPrefix = false // ▁ already added for the space
    } else if (c === "\n" || c === "\t" || c === "\r") {
      // Control characters are kept as-is
      parts.push(c)
      needPrefix = true // Next word needs prefix
    } else {
      // Regular character - add prefix if this is start of a word
      if (
        needPrefix &&
        ((code >= 65 && code <= 90) ||
          (code >= 97 && code <= 122) ||
          (code >= 48 && code <= 57))
      ) {
        parts.push("\u2581")
      }
      parts.push(c)
      needPrefix = false
    }
  }

  return parts.join("")
}

// Streaming UTF-8 decoder for Llama tokens
var utf8Decoder = new TextDecoder("utf-8")

// Build tiktoken unicode-to-byte mapping (inverse of bytes_to_unicode)
var tiktokenUnicodeToByte = null
// Pre-allocated buffer for tokenToBytes (max token length * 4 bytes per char)
var tokenToBytesBuffer = new Uint8Array(256)

function buildTiktokenMap() {
  if (tiktokenUnicodeToByte) {
    return
  }
  tiktokenUnicodeToByte = {}

  // This is the inverse of OpenAI's bytes_to_unicode() function
  // Printable ASCII and some extended chars map to themselves
  var n = 0
  for (var b = 0; b < 256; b = b + 1) {
    // These byte ranges map directly: ! to ~, ¡ to ¬, ® to ÿ
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
      tiktokenUnicodeToByte[b] = b
    } else {
      // Other bytes (0-32, 127-160, 173) map to 256+n, 257+n, etc.
      tiktokenUnicodeToByte[256 + n] = b
      n = n + 1
    }
  }
}

function tokenToBytes(token) {
  if (token < 0 || token >= tokenizer.vocab.length) {
    return new Uint8Array(0)
  }
  var piece = tokenizer.vocab[token]
  if (!piece) {
    return new Uint8Array(0)
  }

  // Handle <0xNN> byte tokens
  if (
    piece.length === 6 &&
    piece.charAt(0) === "<" &&
    piece.charAt(1) === "0" &&
    piece.charAt(2) === "x"
  ) {
    var hex = piece.substring(3, 5)
    var byte = parseInt(hex, 16)
    return new Uint8Array([byte])
  }

  buildTiktokenMap()

  // Pre-allocated buffer: each char maps to at most 4 UTF-8 bytes
  var buf = tokenToBytesBuffer
  var len = 0
  for (var i = 0; i < piece.length; i = i + 1) {
    var code = piece.charCodeAt(i)

    // Handle UTF-16 surrogate pairs
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < piece.length) {
      var low = piece.charCodeAt(i + 1)
      if (low >= 0xdc00 && low <= 0xdfff) {
        code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00)
        i = i + 1
      }
    }

    // Look up in tiktoken mapping
    if (tiktokenUnicodeToByte[code] !== undefined) {
      buf[len] = tiktokenUnicodeToByte[code]
      len = len + 1
    } else {
      // Fallback: encode unknown unicode as UTF-8
      if (code < 0x80) {
        buf[len] = code
        len = len + 1
      } else if (code < 0x800) {
        buf[len] = 0xc0 | (code >> 6)
        buf[len + 1] = 0x80 | (code & 0x3f)
        len = len + 2
      } else if (code < 0x10000) {
        buf[len] = 0xe0 | (code >> 12)
        buf[len + 1] = 0x80 | ((code >> 6) & 0x3f)
        buf[len + 2] = 0x80 | (code & 0x3f)
        len = len + 3
      } else {
        buf[len] = 0xf0 | (code >> 18)
        buf[len + 1] = 0x80 | ((code >> 12) & 0x3f)
        buf[len + 2] = 0x80 | ((code >> 6) & 0x3f)
        buf[len + 3] = 0x80 | (code & 0x3f)
        len = len + 4
      }
    }
  }

  return buf.subarray(0, len)
}

function decodeToken(token) {
  if (token < 0 || token >= tokenizer.vocab.length) {
    return ""
  }
  var piece = tokenizer.vocab[token]
  if (!piece) {
    return ""
  }

  // Handle <0xNN> byte tokens
  if (
    piece.length === 6 &&
    piece.charAt(0) === "<" &&
    piece.charAt(1) === "0" &&
    piece.charAt(2) === "x"
  ) {
    var hex = piece.substring(3, 5)
    var byte = parseInt(hex, 16)
    return String.fromCharCode(byte)
  }

  // For Gemma (SentencePiece), just replace ▁ with space
  if (config.isGemma) {
    return piece.replace(/\u2581/g, " ")
  }

  // For Llama (tiktoken), use TextDecoder for proper UTF-8 handling
  var bytes = tokenToBytes(token)
  return utf8Decoder.decode(bytes)
}

function bpeEncode(text) {
  buildSortedVocab()

  // Convert text based on tokenizer type
  var encodedText
  if (config.isGemma) {
    encodedText = textToSentencePiece(text)
  } else {
    encodedText = textToTiktoken(text)
  }

  var tokens = []
  var pos = 0
  var textLen = encodedText.length

  // Pre-convert to char codes for faster trie traversal
  var codes = new Uint16Array(textLen)
  for (var i = 0; i < textLen; i = i + 1) {
    codes[i] = encodedText.charCodeAt(i)
  }

  while (pos < textLen) {
    // Walk trie for longest match at current position
    var node = vocabTrie
    var bestId = -1
    var bestLen = 0
    for (var j = pos; j < textLen; j = j + 1) {
      node = node[codes[j]]
      if (!node) {
        break
      }
      if (node.id !== undefined) {
        bestId = node.id
        bestLen = node.len
      }
    }

    if (bestId !== -1) {
      tokens.push(bestId)
      pos = pos + bestLen
    } else {
      // No match found - try to find single character token
      var singleChar = encodedText.charAt(pos)
      var singleId = vocabMap[singleChar]
      if (singleId !== undefined) {
        tokens.push(singleId)
      }
      // Skip this character regardless
      pos = pos + 1
    }
  }

  return tokens
}

function encodeLlama3Chat(chatHistory, sysPrompt) {
  var tokens = []

  // Find special tokens
  var bosToken = findSpecialToken("<|begin_of_text|>")
  if (bosToken < 0) {
    bosToken = 128000
  }

  var startHeader = findSpecialToken("<|start_header_id|>")
  if (startHeader < 0) {
    startHeader = 128006
  }

  var endHeader = findSpecialToken("<|end_header_id|>")
  if (endHeader < 0) {
    endHeader = 128007
  }

  var eotToken = findSpecialToken("<|eot_id|>")
  if (eotToken < 0) {
    eotToken = 128009
  }

  // <|begin_of_text|>
  tokens.push(bosToken)

  // System prompt if provided
  if (sysPrompt && sysPrompt.length > 0) {
    tokens.push(startHeader)
    var sysTokens = bpeEncode("system")
    for (var i = 0; i < sysTokens.length; i = i + 1) {
      tokens.push(sysTokens[i])
    }
    tokens.push(endHeader)

    var sysTextTokens = bpeEncode("\n\n" + sysPrompt)
    for (var i = 0; i < sysTextTokens.length; i = i + 1) {
      tokens.push(sysTextTokens[i])
    }
    tokens.push(eotToken)
  }

  // Chat history messages
  for (var m = 0; m < chatHistory.length; m = m + 1) {
    var role = chatHistory[m].role
    var content = chatHistory[m].content

    tokens.push(startHeader)
    var roleTokens = bpeEncode(role)
    for (var i = 0; i < roleTokens.length; i = i + 1) {
      tokens.push(roleTokens[i])
    }
    tokens.push(endHeader)

    var contentTokens = bpeEncode("\n\n" + content)
    for (var i = 0; i < contentTokens.length; i = i + 1) {
      tokens.push(contentTokens[i])
    }
    tokens.push(eotToken)
  }

  // Assistant header for generation
  tokens.push(startHeader)
  var assistantTokens = bpeEncode("assistant")
  for (var i = 0; i < assistantTokens.length; i = i + 1) {
    tokens.push(assistantTokens[i])
  }
  tokens.push(endHeader)

  var newlineTokens = bpeEncode("\n\n")
  for (var i = 0; i < newlineTokens.length; i = i + 1) {
    tokens.push(newlineTokens[i])
  }

  return tokens
}

function encodeGemma3Chat(chatHistory, sysPrompt) {
  var tokens = []

  // Find special tokens
  var bosToken = findSpecialToken("<bos>")
  if (bosToken < 0) {
    // Default Gemma3 BOS
    bosToken = 2
  }

  var startTurn = findSpecialToken("<start_of_turn>")
  if (startTurn < 0) {
    // Default Gemma3 start_of_turn
    startTurn = 106
  }

  var endTurn = findSpecialToken("<end_of_turn>")
  if (endTurn < 0) {
    // Default Gemma3 end_of_turn
    endTurn = 107
  }

  // <bos>
  tokens.push(bosToken)

  // Chat history messages
  var systemUsed = false
  for (var m = 0; m < chatHistory.length; m = m + 1) {
    var role = chatHistory[m].role
    var content = chatHistory[m].content

    // Gemma uses "model" instead of "assistant"
    var gemmaRole = role === "assistant" ? "model" : role

    tokens.push(startTurn)

    var roleText = gemmaRole + "\n"
    // Merge system prompt into first user message
    if (!systemUsed && role === "user" && sysPrompt && sysPrompt.length > 0) {
      roleText = gemmaRole + "\n" + sysPrompt + "\n\n"
      systemUsed = true
    }

    var roleTokens = bpeEncode(roleText + content)
    for (var i = 0; i < roleTokens.length; i = i + 1) {
      tokens.push(roleTokens[i])
    }

    tokens.push(endTurn)

    var newlineTokens = bpeEncode("\n")
    for (var i = 0; i < newlineTokens.length; i = i + 1) {
      tokens.push(newlineTokens[i])
    }
  }

  // Model header for generation
  tokens.push(startTurn)

  var modelTokens = bpeEncode("model\n")
  for (var i = 0; i < modelTokens.length; i = i + 1) {
    tokens.push(modelTokens[i])
  }

  return tokens
}

// ----------------------------------------------------------------------------
// Generation

function generate(chatHistory) {
  var promptTokens
  if (config.isGemma) {
    promptTokens = encodeGemma3Chat(chatHistory, systemPrompt)
  } else {
    promptTokens = encodeLlama3Chat(chatHistory, systemPrompt)
  }

  if (promptTokens.length === 0) {
    promptTokens = [tokenizer.bosToken]
  }

  var token = promptTokens[0]
  var pos = 0
  var output = ""
  var numPromptTokens = promptTokens.length
  var pendingNewline = false

  // Token stream for repeat detection (avoids string search on growing output)
  var generatedTokens = []

  // Use streaming TextDecoder for Llama to handle UTF-8 across token boundaries
  var streamDecoder = config.isGemma
    ? null
    : new TextDecoder("utf-8", { fatal: false })

  var effectiveMaxTokens = maxTokens
  if (effectiveMaxTokens <= 0 || effectiveMaxTokens > config.seqLen) {
    effectiveMaxTokens = config.seqLen
  }

  // Batched prefill: process all prompt tokens except the last in batches
  if (numPromptTokens > 1) {
    var prefillEnd = numPromptTokens - 1
    while (pos < prefillEnd) {
      var bs = Math.min(PREFILL_BATCH_SIZE, prefillEnd - pos)
      transformerPrefill(promptTokens, pos, bs)
      pos = pos + bs
    }
    token = promptTokens[pos]
  }

  for (var step = pos; step < effectiveMaxTokens; step = step + 1) {
    transformer(token, pos, pos >= numPromptTokens - 1)

    var next
    if (pos < numPromptTokens - 1) {
      next = promptTokens[pos + 1]
    } else {
      next = sample(state.logits, temperature)
    }

    if (pos >= numPromptTokens - 1) {
      if (next === tokenizer.eosToken) {
        break
      }

      // Check for model-specific end token
      if (next === tokenizer.eotToken) {
        break
      }

      // Track generated tokens for repeat detection
      generatedTokens.push(next)

      var decoded
      if (config.isGemma) {
        decoded = decodeToken(next)
      } else {
        // For Llama, use streaming decoder to handle multi-byte UTF-8 across tokens
        var bytes = tokenToBytes(next)
        decoded = streamDecoder.decode(bytes, { stream: true })
      }

      // Buffer newlines - only output if followed by non-end token
      if (decoded === "\n") {
        pendingNewline = true
      } else {
        if (pendingNewline) {
          output = output + "\n"
          postMessage({ type: "token", token: "\n" })
          pendingNewline = false
        }
        if (decoded.length > 0) {
          output = output + decoded
          postMessage({ type: "token", token: decoded })
        }

        // Stop if the model is stuck repeating tokens (check every 10 steps)
        var gtLen = generatedTokens.length
        if (gtLen > 20 && step % 10 === 0) {
          // Check if the last 10 tokens repeat as a pattern in history
          var patLen = 10
          var repeats = 0
          var matched = true
          for (var r = 1; r <= 10 && matched; r = r + 1) {
            var off = gtLen - patLen - r * patLen
            if (off < 0) {
              break
            }
            matched = true
            for (var p = 0; p < patLen; p = p + 1) {
              if (generatedTokens[gtLen - patLen + p] !== generatedTokens[off + p]) {
                matched = false
                break
              }
            }
            if (matched) {
              repeats = repeats + 1
            }
          }
          if (repeats > 5) {
            break
          }
        }
      }
    }

    token = next
    pos = pos + 1
  }

  // Flush any remaining bytes in the decoder
  if (!config.isGemma) {
    var remaining = streamDecoder.decode()
    if (remaining.length > 0) {
      output = output + remaining
      postMessage({ type: "token", token: remaining })
    }
  }

  postMessage({ type: "complete", output: output })

  return output
}

// ----------------------------------------------------------------------------
// Message handler

var cbRender

function postMessage(message) {
  if (typeof self !== "undefined" && typeof self.postMessage === "function") {
    self.postMessage(message)
  }
  if (message.type === "token" && cbRender) {
    cbRender(message.token)
  }
}

function llama3pure(data) {
  try {
    switch (data.type) {
      case "load":
        if (data.maxTokens !== undefined) {
          maxTokens = data.maxTokens
        }
        if (data.contextSize !== undefined) {
          contextSize = data.contextSize
        }
        if (data.systemPrompt !== undefined) {
          systemPrompt = data.systemPrompt
        }
        if (data.temperature !== undefined) {
          temperature = data.temperature
        }
        if (data.topP !== undefined) {
          topP = data.topP
        }
        if (data.topK !== undefined) {
          topK = data.topK
        }
        if (typeof data.cbRender === "function") {
          cbRender = data.cbRender
        }
        if (data.model instanceof ArrayBuffer) {
          loadModel(data.model)
        } else {
          console.error(
            "The model parameter is required and must be an ArrayBuffer."
          )
          return
        }
        postMessage({
          type: "loaded",
        })
        break

      case "generate":
        if (ggufUint8) {
          generate(data.chatHistory)
        }
        break

      default:
        break
    }
  } catch (err) {
    console.error(err)
  }
}

// Web Worker mode
if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = function (e) {
    llama3pure(e.data)
  }
}

export default llama3pure
