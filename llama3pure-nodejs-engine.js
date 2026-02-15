/*
----------------------------------------------------------------------------

Designed by Leonardo Javier Russo
https://www.lrusso.com

Web Worker for LLM Inference - Llama-3 and Gemma-3 Transformer models.
Supports GGUF file format with various quantization types.

----------------------------------------------------------------------------
*/

"use strict"

import fs from "fs"

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
var fileFd = null
var offset = 0n // BigInt for >2GB file support

// Q8_0 buffers for quantizing x vector in matmulQuantized
var xQ8Buf = null
var xQ8Int8Buf = null

var temperature = 0.9
var topP = 0.9
var topK = 40
var systemPrompt = "You are a helpful assistant."
var maxTokens = 256
var contextSize = 2048
var modelFilename = ""

// QuantizedTensor structure: { dataOffset, type, rows, cols }
// Stores metadata to read quantized weights on-the-fly during matmul

// ----------------------------------------------------------------------------
// File reading helpers (supports >2GB files using BigInt offsets)

function readBytesFromFile(position, length) {
  var buffer = Buffer.alloc(length)
  fs.readSync(fileFd, buffer, 0, length, position)
  return buffer
}

function readUint8() {
  var buf = readBytesFromFile(offset, 1)
  offset = offset + 1n
  return buf.readUInt8(0)
}

function readUint16() {
  var buf = readBytesFromFile(offset, 2)
  offset = offset + 2n
  return buf.readUInt16LE(0)
}

function readUint32() {
  var buf = readBytesFromFile(offset, 4)
  offset = offset + 4n
  return buf.readUInt32LE(0)
}

function readUint64() {
  var buf = readBytesFromFile(offset, 8)
  offset = offset + 8n
  var low = buf.readUInt32LE(0)
  var high = buf.readUInt32LE(4)
  return low + high * 0x100000000
}

function readInt8() {
  var buf = readBytesFromFile(offset, 1)
  offset = offset + 1n
  return buf.readInt8(0)
}

function readInt32() {
  var buf = readBytesFromFile(offset, 4)
  offset = offset + 4n
  return buf.readInt32LE(0)
}

function readInt64() {
  var buf = readBytesFromFile(offset, 8)
  offset = offset + 8n
  var low = buf.readUInt32LE(0)
  var high = buf.readInt32LE(4)
  return low + high * 0x100000000
}

function readFloat32() {
  var buf = readBytesFromFile(offset, 4)
  offset = offset + 4n
  return buf.readFloatLE(0)
}

function readFloat64() {
  var buf = readBytesFromFile(offset, 8)
  offset = offset + 8n
  return buf.readDoubleLE(0)
}

function readString() {
  var len = readUint64()
  var buf = readBytesFromFile(offset, len)
  offset = offset + BigInt(len)
  return buf.toString("utf-8")
}

function getUint8ArrayAt(srcOffset, length) {
  var buf = readBytesFromFile(BigInt(srcOffset), length)
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
}

function getInt8ArrayAt(srcOffset, length) {
  var buf = readBytesFromFile(BigInt(srcOffset), length)
  return new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength)
}

function getUint16ArrayAt(srcOffset, count) {
  var buf = readBytesFromFile(BigInt(srcOffset), count * 2)
  return new Uint16Array(buf.buffer, buf.byteOffset, count)
}

function getFloat32ArrayAt(srcOffset, count) {
  var buf = readBytesFromFile(BigInt(srcOffset), count * 4)
  return new Float32Array(buf.buffer, buf.byteOffset, count)
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

function fp16ToFp32(h) {
  return fp16Table[h]
}

function bf16ToFp32(h) {
  convInt[0] = h << 16
  return convFloat[0]
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
    var blockStart = srcOffset + (i << 5) // i * 32

    // Find max absolute value in block
    var amax = 0.0
    for (var j = 0; j < 32; j = j + 1) {
      var av = src[blockStart + j]
      if (av < 0) {
        av = -av
      }
      if (av > amax) {
        amax = av
      }
    }

    // Compute scale
    var d = amax / 127.0
    var id = d > 0 ? 127.0 / amax : 0.0

    // Store scale as FP16
    var dFp16 = fp32ToFp16(d)
    dst[bo] = dFp16 & 0xff
    dst[bo + 1] = (dFp16 >> 8) & 0xff

    // Quantize and store values
    for (var j = 0; j < 32; j = j + 1) {
      var v = src[blockStart + j] * id
      // Round to nearest int8
      var q = v > 0 ? (v + 0.5) | 0 : (v - 0.5) | 0
      if (q > 127) {
        q = 127
      }
      if (q < -128) {
        q = -128
      }
      dstInt8[bo + 2 + j] = q
    }

    bo = bo + Q8_0_BLOCK_SIZE
  }
}

// Compute dot product of float vector with Q8_0 cached vector
// x: Float32Array query vector, xOffset: start index
// cache: Uint8Array Q8_0 cache, cacheInt8: Int8Array view of cache
// cacheOffset: byte offset in cache
// count: number of elements (must be multiple of 32)
function dotQ8_0Cache(x, xOffset, cache, cacheInt8, cacheOffset, count) {
  var nb = count >> 5
  var sum = 0.0
  var bo = cacheOffset
  var xb = xOffset

  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(cache[bo] | (cache[bo + 1] << 8))
    var qOff = bo + 2

    var blockSum =
      x[xb] * cacheInt8[qOff] +
      x[xb + 1] * cacheInt8[qOff + 1] +
      x[xb + 2] * cacheInt8[qOff + 2] +
      x[xb + 3] * cacheInt8[qOff + 3] +
      x[xb + 4] * cacheInt8[qOff + 4] +
      x[xb + 5] * cacheInt8[qOff + 5] +
      x[xb + 6] * cacheInt8[qOff + 6] +
      x[xb + 7] * cacheInt8[qOff + 7] +
      x[xb + 8] * cacheInt8[qOff + 8] +
      x[xb + 9] * cacheInt8[qOff + 9] +
      x[xb + 10] * cacheInt8[qOff + 10] +
      x[xb + 11] * cacheInt8[qOff + 11] +
      x[xb + 12] * cacheInt8[qOff + 12] +
      x[xb + 13] * cacheInt8[qOff + 13] +
      x[xb + 14] * cacheInt8[qOff + 14] +
      x[xb + 15] * cacheInt8[qOff + 15] +
      x[xb + 16] * cacheInt8[qOff + 16] +
      x[xb + 17] * cacheInt8[qOff + 17] +
      x[xb + 18] * cacheInt8[qOff + 18] +
      x[xb + 19] * cacheInt8[qOff + 19] +
      x[xb + 20] * cacheInt8[qOff + 20] +
      x[xb + 21] * cacheInt8[qOff + 21] +
      x[xb + 22] * cacheInt8[qOff + 22] +
      x[xb + 23] * cacheInt8[qOff + 23] +
      x[xb + 24] * cacheInt8[qOff + 24] +
      x[xb + 25] * cacheInt8[qOff + 25] +
      x[xb + 26] * cacheInt8[qOff + 26] +
      x[xb + 27] * cacheInt8[qOff + 27] +
      x[xb + 28] * cacheInt8[qOff + 28] +
      x[xb + 29] * cacheInt8[qOff + 29] +
      x[xb + 30] * cacheInt8[qOff + 30] +
      x[xb + 31] * cacheInt8[qOff + 31]
    sum = sum + blockSum * d
    bo = bo + Q8_0_BLOCK_SIZE
    xb = xb + 32
  }
  return sum
}

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
  for (var i = 0; i < count; i = i + 1) {
    dst[dstOffset + i] = src[i]
  }
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
    var scales = src.subarray(blockOffset, blockOffset + QK_K / 16)
    var qs = src.subarray(
      blockOffset + QK_K / 16,
      blockOffset + QK_K / 16 + QK_K / 4
    )
    var dOffset = blockOffset + QK_K / 16 + QK_K / 4
    var d = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))
    var dmin = fp16ToFp32(src[dOffset + 2] | (src[dOffset + 3] << 8))

    var y = dstOffset + i * QK_K
    var is = 0
    var qIdx = 0

    for (var n = 0; n < QK_K; n = n + 128) {
      var shift = 0
      for (var j = 0; j < 4; j = j + 1) {
        var sc = scales[is]
        is = is + 1
        var dl = d * (sc & 0xf)
        var ml = dmin * (sc >> 4)
        for (var l = 0; l < 16; l = l + 1) {
          dst[y] = dl * ((qs[qIdx + l] >> shift) & 3) - ml
          y = y + 1
        }

        sc = scales[is]
        is = is + 1
        dl = d * (sc & 0xf)
        ml = dmin * (sc >> 4)
        for (var l = 0; l < 16; l = l + 1) {
          dst[y] = dl * ((qs[qIdx + l + 16] >> shift) & 3) - ml
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
    var hmask = src.subarray(blockOffset, blockOffset + QK_K / 8)
    var qs = src.subarray(blockOffset + QK_K / 8, blockOffset + QK_K / 8 + QK_K / 4)
    var scalesRaw = src.subarray(
      blockOffset + QK_K / 8 + QK_K / 4,
      blockOffset + QK_K / 8 + QK_K / 4 + 12
    )
    var dOffset = blockOffset + QK_K / 8 + QK_K / 4 + 12
    var d_all = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))

    var aux0 =
      scalesRaw[0] |
      (scalesRaw[1] << 8) |
      (scalesRaw[2] << 16) |
      (scalesRaw[3] << 24)
    var aux1 =
      scalesRaw[4] |
      (scalesRaw[5] << 8) |
      (scalesRaw[6] << 16) |
      (scalesRaw[7] << 24)
    var aux2 =
      scalesRaw[8] |
      (scalesRaw[9] << 8) |
      (scalesRaw[10] << 16) |
      (scalesRaw[11] << 24)

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
          var q = (qs[qIdx + l] >> shift) & 3
          var h = hmask[l] & m ? 0 : 4
          dst[y] = dl * (q - h)
          y = y + 1
        }

        dl = d_all * (scales[is] - 32)
        is = is + 1
        for (var l = 0; l < 16; l = l + 1) {
          var q = (qs[qIdx + l + 16] >> shift) & 3
          var h = hmask[l + 16] & m ? 0 : 4
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
    var scales = src.subarray(blockOffset + 4, blockOffset + 16)
    var qh = src.subarray(blockOffset + 16, blockOffset + 16 + QK_K / 8)
    var ql = src.subarray(
      blockOffset + 16 + QK_K / 8,
      blockOffset + 16 + QK_K / 8 + QK_K / 2
    )

    var y = dstOffset + i * QK_K
    var is = 0
    var u1 = 1
    var u2 = 2
    var qlIdx = 0

    for (var j = 0; j < QK_K; j = j + 64) {
      var sc
      var m
      if (is < 4) {
        sc = scales[is] & 63
        m = scales[is + 4] & 63
      } else {
        sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
        m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
      }
      var d1 = d * sc
      var m1 = dmin * m

      is = is + 1
      if (is < 4) {
        sc = scales[is] & 63
        m = scales[is + 4] & 63
      } else {
        sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
        m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
      }
      var d2 = d * sc
      var m2 = dmin * m
      is = is + 1

      for (var l = 0; l < 32; l = l + 1) {
        dst[y + j + l] = d1 * ((ql[qlIdx + l] & 0xf) + (qh[l] & u1 ? 16 : 0)) - m1
      }
      for (var l = 0; l < 32; l = l + 1) {
        dst[y + j + l + 32] =
          d2 * ((ql[qlIdx + l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2
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
    var scales = src.subarray(blockOffset + 4, blockOffset + 16)
    var qs = src.subarray(blockOffset + 16, blockOffset + 16 + QK_K / 2)

    var is = 0
    var y = dstOffset + i * QK_K

    for (var j = 0; j < QK_K; j = j + 64) {
      var sc
      var m
      if (is < 4) {
        sc = scales[is] & 63
        m = scales[is + 4] & 63
      } else {
        sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
        m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
      }
      var d1 = d * sc
      var m1 = dmin * m

      is = is + 1
      if (is < 4) {
        sc = scales[is] & 63
        m = scales[is + 4] & 63
      } else {
        sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
        m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
      }
      var d2 = d * sc
      var m2 = dmin * m
      is = is + 1

      var qIdx = j / 2
      for (var l = 0; l < 32; l = l + 1) {
        var qByte = qs[qIdx + l]
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
    var ql = src.subarray(blockOffset, blockOffset + QK_K / 2)
    var qh = src.subarray(blockOffset + QK_K / 2, blockOffset + QK_K / 2 + QK_K / 4)
    var scalesOffset = blockOffset + QK_K / 2 + QK_K / 4
    var dOffset = blockOffset + QK_K / 2 + QK_K / 4 + QK_K / 16
    var d = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))

    var y = dstOffset + i * QK_K

    for (var n = 0; n < QK_K; n = n + 128) {
      for (var l = 0; l < 32; l = l + 1) {
        var is = l >> 4
        var scBase = scalesOffset + (n >> 7) * 8
        var q1 = ((ql[n / 2 + l] & 0xf) | (((qh[n / 4 + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((ql[n / 2 + l + 32] & 0xf) | (((qh[n / 4 + l] >> 2) & 3) << 4)) - 32
        var q3 = ((ql[n / 2 + l] >> 4) | (((qh[n / 4 + l] >> 4) & 3) << 4)) - 32
        var q4 = ((ql[n / 2 + l + 32] >> 4) | (((qh[n / 4 + l] >> 6) & 3) << 4)) - 32

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
  var blockSize = 18 // 2 + 16
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0 // block offset in local buffer
  var xb = 0 // x offset

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
  var blockSize = 20 // 2 + 2 + 16
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 22 // 2 + 4 + 16
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 24 // 2 + 2 + 4 + 16
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 34 // 2 + 32
  var totalBytes = nb * blockSize
  // Get both Uint8 and Int8 views of the same data
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var i8 = getInt8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 84 // Q2_K block size
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 110 // 32 + 64 + 12 + 2
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var kmask1 = 0x03030303
  var kmask2 = 0x0f0f0f0f
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 144
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 176 // 2 + 2 + 12 + 32 + 128
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
  var blockSize = 210 // 128 + 64 + 16 + 2
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var i8 = getInt8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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
      for (var l = 0; l < 32; l = l + 1) {
        var is = l >> 4
        var q1 = ((u8[qlBase + l] & 0xf) | (((u8[qhBase + l] >> 0) & 3) << 4)) - 32
        var q2 =
          ((u8[qlBase + l + 32] & 0xf) | (((u8[qhBase + l] >> 2) & 3) << 4)) - 32
        var q3 = ((u8[qlBase + l] >> 4) | (((u8[qhBase + l] >> 4) & 3) << 4)) - 32
        var q4 =
          ((u8[qlBase + l + 32] >> 4) | (((u8[qhBase + l] >> 6) & 3) << 4)) - 32

        blockSum = blockSum + x[xb + nOuter + l] * d * i8[scBase + is] * q1
        blockSum = blockSum + x[xb + nOuter + l + 32] * d * i8[scBase + is + 2] * q2
        blockSum = blockSum + x[xb + nOuter + l + 64] * d * i8[scBase + is + 4] * q3
        blockSum = blockSum + x[xb + nOuter + l + 96] * d * i8[scBase + is + 6] * q4
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
  var blockSize = 18 // 2 + 16
  var totalBytes = nb * blockSize
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xb = 0

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

// Fused dot product for F16
function vecDotF16(x, srcOffset, n) {
  var u8 = getUint8ArrayAt(srcOffset, n * 2)
  var sum = 0.0
  var bo = 0
  for (var i = 0; i < n; i = i + 1) {
    sum = sum + x[i] * fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    bo = bo + 2
  }
  return sum
}

// Fused dot product for BF16
function vecDotBF16(x, srcOffset, n) {
  var u8 = getUint8ArrayAt(srcOffset, n * 2)
  var sum = 0.0
  var bo = 0
  for (var i = 0; i < n; i = i + 1) {
    sum = sum + x[i] * bf16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    bo = bo + 2
  }
  return sum
}

// Fused dot product for F32
function vecDotF32(x, srcOffset, n) {
  var u8 = getUint8ArrayAt(srcOffset, n * 4)
  var sum = 0.0
  var bo = 0
  for (var i = 0; i < n; i = i + 1) {
    convInt[0] = u8[bo] | (u8[bo + 1] << 8) | (u8[bo + 2] << 16) | (u8[bo + 3] << 24)
    sum = sum + x[i] * convFloat[0]
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
  var totalBytes = nb * 18
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = bo + 2
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
    bo = bo + 18
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ4_1_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var totalBytes = nb * 20
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var mw = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = bo + 4
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
    bo = bo + 20
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ5_0_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var totalBytes = nb * 22
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var qh = u8[bo + 2] | (u8[bo + 3] << 8) | (u8[bo + 4] << 16) | (u8[bo + 5] << 24)
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = bo + 6
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
    bo = bo + 22
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ5_1_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var totalBytes = nb * 24
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var mw = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
    var qh = u8[bo + 4] | (u8[bo + 5] << 8) | (u8[bo + 6] << 16) | (u8[bo + 7] << 24)
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = bo + 8
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
    bo = bo + 24
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ8_0_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var totalBytes = nb * 34
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var i8 = getInt8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = bo + 2
    var qx = xOff + 2
    var isum =
      xQ8i8[qx] * i8[qw] +
      xQ8i8[qx + 1] * i8[qw + 1] +
      xQ8i8[qx + 2] * i8[qw + 2] +
      xQ8i8[qx + 3] * i8[qw + 3] +
      xQ8i8[qx + 4] * i8[qw + 4] +
      xQ8i8[qx + 5] * i8[qw + 5] +
      xQ8i8[qx + 6] * i8[qw + 6] +
      xQ8i8[qx + 7] * i8[qw + 7] +
      xQ8i8[qx + 8] * i8[qw + 8] +
      xQ8i8[qx + 9] * i8[qw + 9] +
      xQ8i8[qx + 10] * i8[qw + 10] +
      xQ8i8[qx + 11] * i8[qw + 11] +
      xQ8i8[qx + 12] * i8[qw + 12] +
      xQ8i8[qx + 13] * i8[qw + 13] +
      xQ8i8[qx + 14] * i8[qw + 14] +
      xQ8i8[qx + 15] * i8[qw + 15] +
      xQ8i8[qx + 16] * i8[qw + 16] +
      xQ8i8[qx + 17] * i8[qw + 17] +
      xQ8i8[qx + 18] * i8[qw + 18] +
      xQ8i8[qx + 19] * i8[qw + 19] +
      xQ8i8[qx + 20] * i8[qw + 20] +
      xQ8i8[qx + 21] * i8[qw + 21] +
      xQ8i8[qx + 22] * i8[qw + 22] +
      xQ8i8[qx + 23] * i8[qw + 23] +
      xQ8i8[qx + 24] * i8[qw + 24] +
      xQ8i8[qx + 25] * i8[qw + 25] +
      xQ8i8[qx + 26] * i8[qw + 26] +
      xQ8i8[qx + 27] * i8[qw + 27] +
      xQ8i8[qx + 28] * i8[qw + 28] +
      xQ8i8[qx + 29] * i8[qw + 29] +
      xQ8i8[qx + 30] * i8[qw + 30] +
      xQ8i8[qx + 31] * i8[qw + 31]
    sum = sum + dw * dx * isum
    bo = bo + 34
    xOff = xOff + 34
  }
  return sum
}

function vecDotQ2_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var totalBytes = nb * 84
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
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
    bo = bo + 84
  }
  return sum
}

function vecDotQ3_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var totalBytes = nb * 110
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var kmask1 = 0x03030303
  var kmask2 = 0x0f0f0f0f
  var sum = 0.0
  var bo = 0
  var xOff = 0
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
    bo = bo + 110
  }
  return sum
}

function vecDotQ4_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var totalBytes = nb * 144
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var dmin = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
    var scOff = bo + 4
    var qsOff = bo + 16
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
      isum0 = isum0 + xQ8i8[qx0 + l] * (qByte & 0xf)
      xsum0 = xsum0 + xQ8i8[qx0 + l]
      isum1 = isum1 + xQ8i8[qx1 + l] * (qByte >> 4)
      xsum1 = xsum1 + xQ8i8[qx1 + l]
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
      isum0 = isum0 + xQ8i8[qx0 + l] * (qByte & 0xf)
      xsum0 = xsum0 + xQ8i8[qx0 + l]
      isum1 = isum1 + xQ8i8[qx1 + l] * (qByte >> 4)
      xsum1 = xsum1 + xQ8i8[qx1 + l]
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
      isum0 = isum0 + xQ8i8[qx0 + l] * (qByte & 0xf)
      xsum0 = xsum0 + xQ8i8[qx0 + l]
      isum1 = isum1 + xQ8i8[qx1 + l] * (qByte >> 4)
      xsum1 = xsum1 + xQ8i8[qx1 + l]
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
      isum0 = isum0 + xQ8i8[qx0 + l] * (qByte & 0xf)
      xsum0 = xsum0 + xQ8i8[qx0 + l]
      isum1 = isum1 + xQ8i8[qx1 + l] * (qByte >> 4)
      xsum1 = xsum1 + xQ8i8[qx1 + l]
    }
    blockSum =
      blockSum +
      d1 * dx0 * isum0 -
      dm1 * dx0 * xsum0 +
      d2 * dx1 * isum1 -
      dm2 * dx1 * xsum1

    sum = sum + blockSum
    bo = bo + 144
    xOff = xOff + 272
  }
  return sum
}

function vecDotQ5_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var totalBytes = nb * 176
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
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
        isum0 = isum0 + xQ8i8[qx0 + l] * qw_lo
        xsum0 = xsum0 + xQ8i8[qx0 + l]
        isum1 = isum1 + xQ8i8[qx1 + l] * qw_hi
        xsum1 = xsum1 + xQ8i8[qx1 + l]
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
    bo = bo + 176
  }
  return sum
}

function vecDotQ6_K_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 8
  var totalBytes = nb * 210
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var i8 = getInt8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
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
    bo = bo + 210
  }
  return sum
}

function vecDotIQ4_NL_Q8_0(xQ8, xQ8i8, srcOffset, n) {
  var nb = n >> 5
  var totalBytes = nb * 18
  var u8 = getUint8ArrayAt(srcOffset, totalBytes)
  var sum = 0.0
  var bo = 0
  var xOff = 0
  for (var i = 0; i < nb; i = i + 1) {
    var dw = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
    var dx = fp16ToFp32(xQ8[xOff] | (xQ8[xOff + 1] << 8))
    var qw = bo + 2
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
    bo = bo + 18
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
      return vecDotQ8_0_Q8_0
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

  if (dotQ8Func) {
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

// Fast exp approximation for sigmoid range (|x| < ~10)
// Uses Schraudolph-style bit manipulation via polynomial
function fastExp(x) {
  if (x > 10.0) {
    return 22026.47
  }
  if (x < -10.0) {
    return 0.0
  }
  // Padé [3,3] approximation of exp(x) around 0
  // exp(x) ≈ (1 + x/2 + x²/10 + x³/120) / (1 - x/2 + x²/10 - x³/120)
  var x2 = x * x
  var x3 = x2 * x
  var num = 120.0 + 60.0 * x + 12.0 * x2 + x3
  var den = 120.0 - 60.0 * x + 12.0 * x2 - x3
  return num / den
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

function parseGGUF(filePath) {
  fileFd = fs.openSync(filePath, "r")
  offset = 0n

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
  var currentOffset = Number(offset)
  var tensorDataOffset = Math.ceil(currentOffset / alignment) * alignment

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
      var buf = readBytesFromFile(offset, 2)
      offset = offset + 2n
      return buf.readInt16LE(0)
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
      var arr = []
      for (var i = 0; i < arrLen; i = i + 1) {
        arr.push(readGGUFValue(arrType))
      }
      return arr
    default:
      throw new Error("Unknown GGUF type: " + type)
  }
}

// ----------------------------------------------------------------------------
// Model loading

function loadModel(filePath) {
  // Reset vocab cache when loading a new model
  vocabTrie = null
  vocabMap = null

  var gguf = parseGGUF(filePath)
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
    message:
      "Model architecture: " +
      arch +
      ", isGemma: " +
      isGemma +
      ", keyPrefix: " +
      keyPrefix,
  })

  var vocabTokens = meta["tokenizer.ggml.tokens"] || []

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
    seqLen: Math.min(modelSeqLen, contextSize),
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
    return {
      dataOffset: baseOffset + t.offset,
      type: t.type,
      rows: rows,
      cols: cols,
      rowSize: getRowSize(cols, t.type),
      dotFunc: getVecDotFunc(t.type),
      dotQ8Func: getVecDotQ8Func(t.type),
    }
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
    w.wcls = {
      dataOffset: w.tokenEmbedding.dataOffset,
      type: w.tokenEmbedding.type,
      rows: config.vocabSize,
      cols: config.dim,
      rowSize: w.tokenEmbedding.rowSize,
      dotFunc: getVecDotFunc(w.tokenEmbedding.type),
      dotQ8Func: getVecDotQ8Func(w.tokenEmbedding.type),
    }
  }

  return w
}

function createRunState(p) {
  var headSize = p.headDim
  var kvDim = p.nKvHeads * headSize
  var qDim = p.nHeads * headSize
  var maxDim = Math.max(p.dim, qDim)

  // Pre-compute full RoPE sin/cos tables for all positions
  var ropeSize = headSize / 2
  var seqLen = p.seqLen
  var ropeCosAll = new Float32Array(seqLen * ropeSize)
  var ropeSinAll = new Float32Array(seqLen * ropeSize)
  for (var pos = 0; pos < seqLen; pos = pos + 1) {
    var base = pos * ropeSize
    for (var i = 0; i < ropeSize; i = i + 1) {
      var val = pos / Math.pow(p.ropeTheta, (i * 2) / headSize)
      ropeCosAll[base + i] = Math.cos(val)
      ropeSinAll[base + i] = Math.sin(val)
    }
  }

  // Pre-compute SWA RoPE tables (for Gemma3 sliding window attention layers)
  var swaTheta = p.ropeThetaSwa > 0 ? p.ropeThetaSwa : 10000.0
  var ropeCosSwaAll = new Float32Array(seqLen * ropeSize)
  var ropeSinSwaAll = new Float32Array(seqLen * ropeSize)
  for (var pos = 0; pos < seqLen; pos = pos + 1) {
    var base = pos * ropeSize
    for (var i = 0; i < ropeSize; i = i + 1) {
      var val = pos / Math.pow(swaTheta, (i * 2) / headSize)
      ropeCosSwaAll[base + i] = Math.cos(val)
      ropeSinSwaAll[base + i] = Math.sin(val)
    }
  }

  // Q8_0 KV cache: 34 bytes per 32 floats
  // Number of Q8_0 blocks per KV vector = kvDim / 32
  var kvBlocksPerVec = kvDim >> 5 // kvDim / 32
  var kvCacheBytesPerVec = kvBlocksPerVec * Q8_0_BLOCK_SIZE // 34 bytes per block
  var kvCacheTotalBytes = p.nLayers * p.seqLen * kvCacheBytesPerVec

  // Create ArrayBuffer for KV caches with both Uint8 and Int8 views
  var keyCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)
  var valueCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)

  // Allocate Q8_0 buffer for quantizing x in matmulQuantized
  var maxCols = Math.max(p.dim, qDim, p.hiddenDim)
  var xQ8Size = (maxCols >> 5) * 34
  var xQ8Buffer = new ArrayBuffer(xQ8Size)
  xQ8Buf = new Uint8Array(xQ8Buffer)
  xQ8Int8Buf = new Int8Array(xQ8Buffer)

  // Pre-compute head offset tables to avoid repeated multiplication in attention loop
  var kvMul = p.nHeads / p.nKvHeads
  var headQOffsets = new Int32Array(p.nHeads)
  var headKvIdx = new Int32Array(p.nHeads)
  var headAttOffsets = new Int32Array(p.nHeads)
  for (var h = 0; h < p.nHeads; h = h + 1) {
    headQOffsets[h] = h * headSize
    headKvIdx[h] = (h / kvMul) | 0
    headAttOffsets[h] = h * p.seqLen
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
    // Cache layout info
    kvCacheBytesPerVec: kvCacheBytesPerVec,
    // Pre-computed RoPE sin/cos for all positions
    ropeCosAll: ropeCosAll,
    ropeSinAll: ropeSinAll,
    ropeCosSwaAll: ropeCosSwaAll,
    ropeSinSwaAll: ropeSinSwaAll,
    ropeSize: ropeSize,
    // Cached constants to avoid recomputation in transformer
    headSize: headSize,
    kvDim: kvDim,
    qDim: qDim,
    kvMul: p.nHeads / p.nKvHeads,
    // Q8_0 cache: layer size in bytes
    kvCacheLayerSize: p.seqLen * kvCacheBytesPerVec,
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
  }
}

// ----------------------------------------------------------------------------
// Transformer forward pass

function transformer(token, pos, computeLogits) {
  var w = weights
  var s = state
  // Use cached values from state instead of config lookups
  var dim = s.dim
  var headSize = s.headSize
  var kvDim = s.kvDim
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var eps = s.rmsNormEps
  var nLayers = s.nLayers
  var nHeads = s.nHeads
  var nKvHeads = s.nKvHeads
  var isGemma = s.isGemma
  var invDim = s.invDim
  var invHeadSize = s.invHeadSize

  // Cache array references for JIT optimization
  var xArr = s.x
  var xbArr = s.xb
  var xb2Arr = s.xb2

  // Dequantize token embedding on-demand (saves ~500MB+ for large vocab models)
  var emb = w.tokenEmbedding
  var embByteOffset = emb.dataOffset + token * emb.rowSize
  dequantizeRow(xArr, embByteOffset, dim, emb.type)

  // Gemma: Scale embeddings by sqrt(dim) - use pre-computed value
  if (isGemma) {
    var scale = s.embedScale
    var dim4 = dim & ~3
    var i = 0
    for (; i < dim4; i = i + 4) {
      xArr[i] *= scale
      xArr[i + 1] *= scale
      xArr[i + 2] *= scale
      xArr[i + 3] *= scale
    }
    for (; i < dim; i = i + 1) {
      xArr[i] *= scale
    }
  }

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l] // Layer weights

    if (isGemma) {
      rmsnormGemma(xbArr, xArr, lw.rmsAttWeight, dim, eps, invDim)
    } else {
      rmsnorm(xbArr, xArr, lw.rmsAttWeight, dim, invDim)
    }

    // QKV matmuls - using quantized weights
    matmulQuantized(s.q, xbArr, lw.wq)
    matmulQuantized(s.k, xbArr, lw.wk)
    matmulQuantized(s.v, xbArr, lw.wv)

    if (isGemma && lw.attnQNorm && lw.attnKNorm) {
      for (var h = 0; h < nHeads; h = h + 1) {
        rmsnormGemmaAt(s.q, h * headSize, lw.attnQNorm, headSize, eps, invHeadSize)
      }
      for (var h = 0; h < nKvHeads; h = h + 1) {
        rmsnormGemmaAt(s.k, h * headSize, lw.attnKNorm, headSize, eps, invHeadSize)
      }
    }

    // Apply RoPE using pre-computed frequencies and cached sin/cos
    // For Gemma3 with SWA: different layers use different RoPE frequencies
    var swaPattern = s.swaPattern
    var isSwaLayer = swaPattern > 0 && l % swaPattern < swaPattern - 1
    var half = headSize >> 1
    // Cache q and k array references for RoPE
    var qArr = s.q
    var kArr = s.k

    // Index into pre-computed RoPE tables for this position
    var ropeBase = pos * s.ropeSize
    var ropeCos
    var ropeSin
    if (isGemma && isSwaLayer) {
      ropeCos = s.ropeCosSwaAll
      ropeSin = s.ropeSinSwaAll
    } else {
      ropeCos = s.ropeCosAll
      ropeSin = s.ropeSinAll
    }

    if (isGemma) {
      // Fused RoPE + Q attention scaling (avoids separate scaling pass)
      var attnScale = s.attnScale
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
    } else {
      // Apply RoPE to Q and K with 4-way unrolling (Llama path)
      var kvDim4 = kvDim & ~3
      for (var i = 0; i < kvDim4; i = i + 4) {
        var fi0 = (i >> 1) % half
        var fi1 = ((i + 2) >> 1) % half
        var fcr0 = ropeCos[ropeBase + fi0]
        var fci0 = ropeSin[ropeBase + fi0]
        var fcr1 = ropeCos[ropeBase + fi1]
        var fci1 = ropeSin[ropeBase + fi1]
        // Process Q pair 0
        var qv0 = qArr[i]
        var qv1 = qArr[i + 1]
        qArr[i] = qv0 * fcr0 - qv1 * fci0
        qArr[i + 1] = qv0 * fci0 + qv1 * fcr0
        // Process Q pair 1
        var qv2 = qArr[i + 2]
        var qv3 = qArr[i + 3]
        qArr[i + 2] = qv2 * fcr1 - qv3 * fci1
        qArr[i + 3] = qv2 * fci1 + qv3 * fcr1
        // Process K pair 0
        var kv0 = kArr[i]
        var kv1 = kArr[i + 1]
        kArr[i] = kv0 * fcr0 - kv1 * fci0
        kArr[i + 1] = kv0 * fci0 + kv1 * fcr0
        // Process K pair 1
        var kv2 = kArr[i + 2]
        var kv3 = kArr[i + 3]
        kArr[i + 2] = kv2 * fcr1 - kv3 * fci1
        kArr[i + 3] = kv2 * fci1 + kv3 * fcr1
      }
      // kvDim remainder (if kvDim not multiple of 4)
      for (var i = kvDim4; i < kvDim; i = i + 2) {
        var freqIdx = (i >> 1) % half
        var fcr = ropeCos[ropeBase + freqIdx]
        var fci = ropeSin[ropeBase + freqIdx]
        var v0 = qArr[i]
        var v1 = qArr[i + 1]
        qArr[i] = v0 * fcr - v1 * fci
        qArr[i + 1] = v0 * fci + v1 * fcr
        v0 = kArr[i]
        v1 = kArr[i + 1]
        kArr[i] = v0 * fcr - v1 * fci
        kArr[i + 1] = v0 * fci + v1 * fcr
      }
      // Q-only remainder (qDim > kvDim for GQA)
      for (var i = kvDim; i < qDim; i = i + 2) {
        var freqIdx = (i >> 1) % half
        var fcr = ropeCos[ropeBase + freqIdx]
        var fci = ropeSin[ropeBase + freqIdx]
        var v0 = qArr[i]
        var v1 = qArr[i + 1]
        qArr[i] = v0 * fcr - v1 * fci
        qArr[i + 1] = v0 * fci + v1 * fcr
      }
    }

    // Q8_0 KV cache: use byte offsets
    var kvBytesPerVec = s.kvCacheBytesPerVec
    var loff = l * s.kvCacheLayerSize // layer offset in bytes
    var cacheOffset = loff + pos * kvBytesPerVec // byte offset for this position
    var vArr = s.v
    var keyCache = s.keyCache
    var valueCache = s.valueCache
    var keyCacheInt8 = s.keyCacheInt8
    var valueCacheInt8 = s.valueCacheInt8

    // Quantize K and V to Q8_0 format and store in cache
    quantizeToQ8_0Cache(kArr, 0, keyCache, keyCacheInt8, cacheOffset, kvDim)
    quantizeToQ8_0Cache(vArr, 0, valueCache, valueCacheInt8, cacheOffset, kvDim)

    // Use pre-computed attention scale
    var attScale = isGemma ? 1.0 : s.attnScale

    // Cache array references for JIT optimization
    var sAtt = s.att

    // Zero xb once for all heads
    xbArr.fill(0, 0, qDim)

    // Bytes per KV head = (headSize / 32) * 34
    var headBytesQ8 = (headSize >> 5) * Q8_0_BLOCK_SIZE

    // SWA window enforcement: restrict attention range for SWA layers
    var startT =
      isSwaLayer && config.swaWindow > 0
        ? Math.max(0, pos - config.swaWindow + 1)
        : 0

    // Cache head offset tables
    var headQOff = s.headQOffsets
    var headKvI = s.headKvIdx
    var headAttOff = s.headAttOffsets

    for (var h = 0; h < nHeads; h = h + 1) {
      var qOffset = headQOff[h]
      var attOffset = headAttOff[h]
      var kvHeadIdx = headKvI[h]
      var kvHeadByteOff = kvHeadIdx * headBytesQ8

      // Compute attention scores using Q8_0 dot product
      var kBase = loff + kvHeadByteOff
      for (var t = startT; t <= pos; t = t + 1) {
        var kByteOffset = kBase + t * kvBytesPerVec
        var score = dotQ8_0Cache(
          qArr,
          qOffset,
          keyCache,
          keyCacheInt8,
          kByteOffset,
          headSize
        )
        sAtt[attOffset + t] = score * attScale
      }

      // Fused softmax: pass 1 = find max, pass 2 = exp + sum + normalize
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

      // Accumulate weighted values using Q8_0 cache
      var xbOffset = h * headSize
      var vBase = loff + kvHeadByteOff
      for (var t = startT; t <= pos; t = t + 1) {
        var vByteOffset = vBase + t * kvBytesPerVec
        var a = sAtt[attOffset + t]
        accumQ8_0Cache(
          xbArr,
          xbOffset,
          valueCache,
          valueCacheInt8,
          vByteOffset,
          a,
          headSize
        )
      }
    }

    // Attention output - using quantized weights
    matmulQuantized(xb2Arr, xbArr, lw.wo)

    if (isGemma && lw.attnPostNorm) {
      rmsnormGemma(xb2Arr, xb2Arr, lw.attnPostNorm, dim, eps, invDim)
    }

    accum(xArr, xb2Arr, dim)

    if (isGemma) {
      rmsnormGemma(xbArr, xArr, lw.rmsFfnWeight, dim, eps, invDim)
    } else {
      rmsnorm(xbArr, xArr, lw.rmsFfnWeight, dim, invDim)
    }

    // FFN gate and up - using quantized weights
    var hbArr = s.hb
    var hb2Arr = s.hb2
    matmulQuantized(hbArr, xbArr, lw.w1)
    matmulQuantized(hb2Arr, xbArr, lw.w3)

    // Apply activation (GELU for Gemma, SiLU for Llama)
    var hd4 = hiddenDim & ~3
    // GELU constants: factor out 0.7978845608 * 0.044715 = 0.035677408137
    var GELU_A = 0.7978845608
    var GELU_B = 0.035677408137

    if (isGemma) {
      // GELU activation
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
    } else {
      // SiLU activation
      for (var i = 0; i < hd4; i = i + 4) {
        var v0 = hbArr[i]
        var v1 = hbArr[i + 1]
        var v2 = hbArr[i + 2]
        var v3 = hbArr[i + 3]
        hbArr[i] = (v0 / (1.0 + fastExp(-v0))) * hb2Arr[i]
        hbArr[i + 1] = (v1 / (1.0 + fastExp(-v1))) * hb2Arr[i + 1]
        hbArr[i + 2] = (v2 / (1.0 + fastExp(-v2))) * hb2Arr[i + 2]
        hbArr[i + 3] = (v3 / (1.0 + fastExp(-v3))) * hb2Arr[i + 3]
      }
      for (var i = hd4; i < hiddenDim; i = i + 1) {
        var val = hbArr[i]
        hbArr[i] = (val / (1.0 + fastExp(-val))) * hb2Arr[i]
      }
    }

    // FFN down - using quantized weights
    matmulQuantized(xbArr, hbArr, lw.w2)

    if (isGemma && lw.ffnPostNorm) {
      rmsnormGemma(xbArr, xbArr, lw.ffnPostNorm, dim, eps, invDim)
    }

    accum(xArr, xbArr, dim)
  }

  if (isGemma) {
    rmsnormGemma(xArr, xArr, w.rmsFinalWeight, dim, eps, invDim)
  } else {
    rmsnorm(xArr, xArr, w.rmsFinalWeight, dim, invDim)
  }

  // Classifier into logits - skip during prompt prefill for speed
  if (computeLogits !== false) {
    matmulQuantized(s.logits, xArr, w.wcls)

    if (isGemma && config.finalLogitSoftcapping > 0) {
      var cap = config.finalLogitSoftcapping
      var vocabSize = s.vocabSize
      for (var i = 0; i < vocabSize; i = i + 1) {
        s.logits[i] = cap * fastTanh(s.logits[i] / cap)
      }
    }
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
      // Re-find min
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

  for (var step = 0; step < maxTokens; step = step + 1) {
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

      // Check for model-specific end tokens
      if (config.isGemma) {
        if (tokenizer.vocab[next] === "<end_of_turn>") {
          break
        }
      } else {
        // Llama: check for EOT token
        if (tokenizer.vocab[next] === "<|eot_id|>") {
          break
        }
        if (next === 128009) {
          // Default EOT token ID
          break
        }
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
        if (data.filename !== undefined) {
          modelFilename = data.filename
        }
        var result = loadModel(data.filePath)
        cbRender = data.cbRender
        postMessage({
          type: "loaded",
          config: result.config,
          filename: modelFilename,
          maxTokens: maxTokens,
          contextSize: contextSize,
          topP: topP,
          topK: topK,
        })
        break

      case "generate":
        generate(data.chatHistory)
        break

      case "setTemperature":
        temperature = data.value
        break

      case "setSystemPrompt":
        systemPrompt = data.value
        break

      default:
        postMessage({
          type: "error",
          message: "Unknown message type: " + data.type,
        })
    }
  } catch (err) {
    postMessage({ type: "error", message: err.message })
  }
}

export default llama3pure
