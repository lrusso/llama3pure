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
var QK4_NL = 32

// IQ4_NL lookup table
var kvalues_iq4nl = [
  -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
]

// ----------------------------------------------------------------------------
// UTF-8 decoding

function decodeUTF8(bytes) {
  var str = ""
  var i = 0
  var len = bytes.length
  while (i < len) {
    var b0 = bytes[i]
    if (b0 < 0x80) {
      str = str + String.fromCharCode(b0)
      i = i + 1
    } else if ((b0 & 0xe0) === 0xc0) {
      if (i + 1 < len) {
        str = str + String.fromCharCode(((b0 & 0x1f) << 6) | (bytes[i + 1] & 0x3f))
        i = i + 2
      } else {
        i = i + 1
      }
    } else if ((b0 & 0xf0) === 0xe0) {
      if (i + 2 < len) {
        str =
          str +
          String.fromCharCode(
            ((b0 & 0x0f) << 12) |
              ((bytes[i + 1] & 0x3f) << 6) |
              (bytes[i + 2] & 0x3f)
          )
        i = i + 3
      } else {
        i = i + 1
      }
    } else if ((b0 & 0xf8) === 0xf0) {
      if (i + 3 < len) {
        var cp =
          ((b0 & 0x07) << 18) |
          ((bytes[i + 1] & 0x3f) << 12) |
          ((bytes[i + 2] & 0x3f) << 6) |
          (bytes[i + 3] & 0x3f)
        cp = cp - 0x10000
        str = str + String.fromCharCode(0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff))
        i = i + 4
      } else {
        i = i + 1
      }
    } else {
      i = i + 1
    }
  }
  return str
}

// Streaming decoder (Llama): a cut sequence waits for the next token's bytes
function createStreamingUTF8Decoder() {
  var pendingBytes = []
  return {
    decode: function decode(bytes, stream) {
      var i
      var combined
      if (!bytes) {
        bytes = new Uint8Array(0)
      }
      if (pendingBytes.length > 0) {
        combined = new Uint8Array(pendingBytes.length + bytes.length)
        for (i = 0; i < pendingBytes.length; i++) {
          combined[i] = pendingBytes[i]
        }
        for (i = 0; i < bytes.length; i++) {
          combined[pendingBytes.length + i] = bytes[i]
        }
        pendingBytes = []
      } else {
        combined = bytes
      }
      // Streaming: a cut sequence waits for the next token
      var len = combined.length
      var end = len
      if (stream) {
        i = 0
        while (i < len) {
          var b0 = combined[i]
          var seqLen = 1
          if ((b0 & 0xe0) === 0xc0) {
            seqLen = 2
          } else if ((b0 & 0xf0) === 0xe0) {
            seqLen = 3
          } else if ((b0 & 0xf8) === 0xf0) {
            seqLen = 4
          }
          if (i + seqLen > len) {
            end = i
            break
          }
          i = i + seqLen
        }
        for (i = end; i < len; i++) {
          pendingBytes.push(combined[i])
        }
      }
      return decodeUTF8(end === len ? combined : combined.subarray(0, end))
    },
  }
}
// ----------------------------------------------------------------------------
// State

var config = null
var weights = null
var state = null
var tokenizer = null
var ggufData = null
var dataView = null
var offset = 0

// Q8_0 buffers for the x vector in matmulQuantized (allocated on first use)
var xQ8Buf = null
var xQ8Int8Buf = null
var xQ8Size = 0
// Scratch of one generate() call: tokenizer hash table, then prefill batch
// buffers, then logits; released when the call ends
var genScratch = null
var matmulDeqBuf = null
// Four row views over matmulDeqBuf (Q8_0 prefill kernel)
var matmulDeqRows = null
// Int8 view over matmulDeqBuf (block-32 integer prefill kernel)
var matmulDeqI8 = null

var temperature = 0.9
var topP = 0.9
var topK = 40
var systemPrompt = "You are a helpful assistant."
var maxTokens = -1
var contextSize = 0

// QuantizedTensor records, read on the fly by the matmuls

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
  return decodeUTF8(bytes)
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

// FP16 to FP32 table (256 KB), filled on the first generate() (ensureFp16Table)
var fp16Table = new Float32Array(65536)
var fp16TableReady = false

function ensureFp16Table() {
  if (fp16TableReady) {
    return
  }
  fp16TableReady = true
  var t = fp16Table
  // e = 0: m * 2^-24; e = 31: Infinity or NaN; the upper half is the sign bit
  var scale = Math.pow(2, -24)
  for (var e = 0; e < 31; e = e + 1) {
    var base = e << 10
    var add = e === 0 ? 0 : 1024
    for (var m = 0; m < 1024; m = m + 1) {
      var v = (add + m) * scale
      t[base + m] = v
      t[base + m + 0x8000] = -v
    }
    if (e > 0) {
      scale = scale * 2
    }
  }
  t[0x7c00] = Infinity
  t[0xfc00] = -Infinity
  for (var m = 1; m < 1024; m = m + 1) {
    t[0x7c00 + m] = NaN
    t[0xfc00 + m] = NaN
  }
}

// BF16 to FP32 lookup table (256 KB), filled on first use (ensureBf16Table)
var bf16Table = null

function ensureBf16Table() {
  if (bf16Table !== null) {
    return
  }
  bf16Table = new Float32Array(65536)
  for (var h = 0; h < 65536; h = h + 1) {
    convInt[0] = h << 16
    bf16Table[h] = convFloat[0]
  }
}

function fp16ToFp32(h) {
  return fp16Table[h]
}

function bf16ToFp32(h) {
  if (bf16Table === null) {
    ensureBf16Table()
  }
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

// Quantize `count` floats (multiple of 32) of src into the Q8_0 cache dst
function quantizeToQ8_0Cache(src, srcOffset, dst, dstInt8, dstOffset, count) {
  var nb = count >> 5 // count / 32
  var bo = dstOffset // byte offset in destination

  for (var i = 0; i < nb; i = i + 1) {
    var bs = srcOffset + (i << 5) // i * 32

    // Find max absolute value in block
    var amax = 0.0
    for (var k = 0; k < 32; k = k + 1) {
      var av = src[bs + k]
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

    // Quantize and store values (round half away from zero)
    var qo = bo + 2
    for (var k = 0; k < 32; k = k + 1) {
      var v = src[bs + k] * id
      dstInt8[qo + k] = v > 0 ? (v + 0.5) | 0 : (v - 0.5) | 0
    }

    bo = bo + Q8_0_BLOCK_SIZE
  }
}

// out[outOffset...] += weight * Q8_0 cached vector (count: multiple of 32)
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
    for (var k = 0; k < 32; k = k + 1) {
      out[ob + k] = out[ob + k] + cacheInt8[qOff + k] * scale
    }
    bo = bo + Q8_0_BLOCK_SIZE
    ob = ob + 32
  }
}

// Dot product of two Q8_0 cached vectors (Q heads against the KV cache)
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

    // Exact integer dot product of the 32 int8 pairs
    var isum = 0
    for (var k = 0; k < 32; k = k + 1) {
      isum = (isum + aI8[qa + k] * bI8[qb + k]) | 0
    }

    sum = sum + da * db * isum
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

// Row dequantizers for F16 / BF16 (Int32 view, two values per word) and F32
function deqRowF16(I32, p, dst, dstOff, cols) {
  var tab = fp16Table
  var half = cols >> 1
  for (var k = 0; k < half; k = k + 1) {
    var v = I32[(p + k) | 0]
    var y = (dstOff + (k << 1)) | 0
    dst[y] = tab[v & 0xffff]
    dst[(y + 1) | 0] = tab[v >>> 16]
  }
}

function deqRowBF16(I32, p, dst, dstOff, cols) {
  var tab = bf16Table
  var half = cols >> 1
  for (var k = 0; k < half; k = k + 1) {
    var v = I32[(p + k) | 0]
    var y = (dstOff + (k << 1)) | 0
    dst[y] = tab[v & 0xffff]
    dst[(y + 1) | 0] = tab[v >>> 16]
  }
}

function deqRowF32(F32, p, dst, dstOff, cols) {
  for (var j = 0; j < cols; j = j + 1) {
    dst[(dstOff + j) | 0] = F32[(p + j) | 0]
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

// ----------------------------------------------------------------------------
// K-quant row dequantizers: (view, p, dst, dstOff, cols), view = Int32Array
// (Q2_K/Q4_K/Q5_K) or Uint16Array (Q3_K/Q6_K), p = row start in it

function deqRowQ2_K(I32, p, dst, dstOff, cols) {
  var nb = cols >> 8
  var y = dstOff
  for (var i = 0; i < nb; i = i + 1) {
    var w20 = I32[(p + 20) | 0]
    var d = fp16Table[w20 & 0xffff]
    var dmin = fp16Table[w20 >>> 16]
    var sw = p
    for (var h = 0; h < 2; h = h + 1) {
      // 32 quant bytes of this half, reused by the four 2-bit shifts
      var qp = (p + 4 + (h << 3)) | 0
      var v0 = I32[qp]
      var v1 = I32[(qp + 1) | 0]
      var v2 = I32[(qp + 2) | 0]
      var v3 = I32[(qp + 3) | 0]
      var v4 = I32[(qp + 4) | 0]
      var v5 = I32[(qp + 5) | 0]
      var v6 = I32[(qp + 6) | 0]
      var v7 = I32[(qp + 7) | 0]
      var scw0 = I32[(sw + (h << 1)) | 0]
      var scw1 = I32[(sw + (h << 1) + 1) | 0]
      for (var s = 0; s < 4; s = s + 1) {
        var shift = s << 1
        // scales 8h + 2s (first 16 values) and 8h + 2s + 1 (next 16)
        var scPair = s < 2 ? scw0 : scw1
        var scA = (scPair >>> ((s & 1) << 4)) & 0xff
        var scB = (scPair >>> (((s & 1) << 4) + 8)) & 0xff
        var dl = d * (scA & 0xf)
        var ml = dmin * (scA >> 4)
        dst[y] = dl * ((v0 >>> shift) & 3) - ml
        dst[(y + 1) | 0] = dl * ((v0 >>> (shift + 8)) & 3) - ml
        dst[(y + 2) | 0] = dl * ((v0 >>> (shift + 16)) & 3) - ml
        dst[(y + 3) | 0] = dl * ((v0 >>> (shift + 24)) & 3) - ml
        dst[(y + 4) | 0] = dl * ((v1 >>> shift) & 3) - ml
        dst[(y + 5) | 0] = dl * ((v1 >>> (shift + 8)) & 3) - ml
        dst[(y + 6) | 0] = dl * ((v1 >>> (shift + 16)) & 3) - ml
        dst[(y + 7) | 0] = dl * ((v1 >>> (shift + 24)) & 3) - ml
        dst[(y + 8) | 0] = dl * ((v2 >>> shift) & 3) - ml
        dst[(y + 9) | 0] = dl * ((v2 >>> (shift + 8)) & 3) - ml
        dst[(y + 10) | 0] = dl * ((v2 >>> (shift + 16)) & 3) - ml
        dst[(y + 11) | 0] = dl * ((v2 >>> (shift + 24)) & 3) - ml
        dst[(y + 12) | 0] = dl * ((v3 >>> shift) & 3) - ml
        dst[(y + 13) | 0] = dl * ((v3 >>> (shift + 8)) & 3) - ml
        dst[(y + 14) | 0] = dl * ((v3 >>> (shift + 16)) & 3) - ml
        dst[(y + 15) | 0] = dl * ((v3 >>> (shift + 24)) & 3) - ml
        dl = d * (scB & 0xf)
        ml = dmin * (scB >> 4)
        dst[(y + 16) | 0] = dl * ((v4 >>> shift) & 3) - ml
        dst[(y + 17) | 0] = dl * ((v4 >>> (shift + 8)) & 3) - ml
        dst[(y + 18) | 0] = dl * ((v4 >>> (shift + 16)) & 3) - ml
        dst[(y + 19) | 0] = dl * ((v4 >>> (shift + 24)) & 3) - ml
        dst[(y + 20) | 0] = dl * ((v5 >>> shift) & 3) - ml
        dst[(y + 21) | 0] = dl * ((v5 >>> (shift + 8)) & 3) - ml
        dst[(y + 22) | 0] = dl * ((v5 >>> (shift + 16)) & 3) - ml
        dst[(y + 23) | 0] = dl * ((v5 >>> (shift + 24)) & 3) - ml
        dst[(y + 24) | 0] = dl * ((v6 >>> shift) & 3) - ml
        dst[(y + 25) | 0] = dl * ((v6 >>> (shift + 8)) & 3) - ml
        dst[(y + 26) | 0] = dl * ((v6 >>> (shift + 16)) & 3) - ml
        dst[(y + 27) | 0] = dl * ((v6 >>> (shift + 24)) & 3) - ml
        dst[(y + 28) | 0] = dl * ((v7 >>> shift) & 3) - ml
        dst[(y + 29) | 0] = dl * ((v7 >>> (shift + 8)) & 3) - ml
        dst[(y + 30) | 0] = dl * ((v7 >>> (shift + 16)) & 3) - ml
        dst[(y + 31) | 0] = dl * ((v7 >>> (shift + 24)) & 3) - ml
        y = y + 32
      }
    }
    p = p + 21
  }
}

// Scales of the Q3_K / Q6_K blocks being read (16 per row)
var q3kScales = new Int8Array(64)

// 12 scale bytes of the Q3_K block at word p -> 16 scales at q3kScales[base..]
function q3kUnpackScales(U16, p, base) {
  var kmask1 = 0x03030303
  var kmask2 = 0x0f0f0f0f
  var aux0 = U16[(p + 48) | 0] | (U16[(p + 49) | 0] << 16)
  var aux1 = U16[(p + 50) | 0] | (U16[(p + 51) | 0] << 16)
  var aux2 = U16[(p + 52) | 0] | (U16[(p + 53) | 0] << 16)
  var s0 = (aux0 & kmask2) | (((aux2 >> 0) & kmask1) << 4)
  var s1 = (aux1 & kmask2) | (((aux2 >> 2) & kmask1) << 4)
  var s2 = ((aux0 >> 4) & kmask2) | (((aux2 >> 4) & kmask1) << 4)
  var s3 = ((aux1 >> 4) & kmask2) | (((aux2 >> 6) & kmask1) << 4)
  var sc = q3kScales
  for (var q = 0; q < 4; q = q + 1) {
    var sh = q << 3
    sc[base + q] = (s0 >> sh) & 0xff
    sc[base + 4 + q] = (s1 >> sh) & 0xff
    sc[base + 8 + q] = (s2 >> sh) & 0xff
    sc[base + 12 + q] = (s3 >> sh) & 0xff
  }
}

function deqRowQ3_K(U16, p, dst, dstOff, cols) {
  var nb = cols >> 8
  var y = dstOff
  var sc = q3kScales
  for (var i = 0; i < nb; i = i + 1) {
    var dAll = fp16Table[U16[(p + 54) | 0]]
    q3kUnpackScales(U16, p, 0)
    var is = 0
    for (var h = 0; h < 2; h = h + 1) {
      // quants: 32 bytes per half (16 words); high-bit mask: 32 bytes shared
      var qp = (p + 16 + (h << 4)) | 0
      for (var s = 0; s < 4; s = s + 1) {
        var shift = s << 1
        var bit = (h << 2) + s
        var dl = dAll * (sc[is] - 32)
        is = is + 1
        for (var k = 0; k < 8; k = k + 1) {
          var qv = U16[(qp + k) | 0]
          var hv = U16[(p + k) | 0]
          var q0 = (qv >>> shift) & 3
          var q1 = (qv >>> (shift + 8)) & 3
          var h0 = 4 - (((hv >>> bit) & 1) << 2)
          var h1 = 4 - (((hv >>> (bit + 8)) & 1) << 2)
          dst[(y + (k << 1)) | 0] = dl * (q0 - h0)
          dst[(y + (k << 1) + 1) | 0] = dl * (q1 - h1)
        }
        y = y + 16
        dl = dAll * (sc[is] - 32)
        is = is + 1
        for (var k = 0; k < 8; k = k + 1) {
          var qv = U16[(qp + 8 + k) | 0]
          var hv = U16[(p + 8 + k) | 0]
          var q0 = (qv >>> shift) & 3
          var q1 = (qv >>> (shift + 8)) & 3
          var h0 = 4 - (((hv >>> bit) & 1) << 2)
          var h1 = 4 - (((hv >>> (bit + 8)) & 1) << 2)
          dst[(y + (k << 1)) | 0] = dl * (q0 - h0)
          dst[(y + (k << 1) + 1) | 0] = dl * (q1 - h1)
        }
        y = y + 16
      }
    }
    p = p + 55
  }
}

// Scale/min pair `is` (0-7) of a Q4_K/Q5_K block: sc | m << 8
function kScaleMin(is, w1, w2, w3) {
  if (is < 4) {
    return ((w1 >> (is << 3)) & 63) | (((w2 >> (is << 3)) & 63) << 8)
  }
  var k = (is - 4) << 3
  var a = (w3 >> k) & 0xff
  var sc = (a & 0xf) | ((((w1 >> k) & 0xff) >> 6) << 4)
  var m = (a >> 4) | ((((w2 >> k) & 0xff) >> 6) << 4)
  return sc | (m << 8)
}

function deqRowQ4_K(I32, p, dst, dstOff, cols) {
  var nb = cols >> 8
  var y = dstOff
  for (var i = 0; i < nb; i = i + 1) {
    var w0 = I32[p]
    var d = fp16Table[w0 & 0xffff]
    var dmin = fp16Table[w0 >>> 16]
    var w1 = I32[(p + 1) | 0]
    var w2 = I32[(p + 2) | 0]
    var w3 = I32[(p + 3) | 0]
    var qp = (p + 4) | 0
    for (var c = 0; c < 4; c = c + 1) {
      var sm = kScaleMin(c << 1, w1, w2, w3)
      var d1 = d * (sm & 0xff)
      var m1 = dmin * (sm >>> 8)
      sm = kScaleMin((c << 1) + 1, w1, w2, w3)
      var d2 = d * (sm & 0xff)
      var m2 = dmin * (sm >>> 8)
      var y1 = (y + 32) | 0
      for (var k = 0; k < 8; k = k + 1) {
        var v = I32[(qp + k) | 0]
        var o = (k << 2) | 0
        dst[(y + o) | 0] = d1 * (v & 0xf) - m1
        dst[(y + o + 1) | 0] = d1 * ((v >>> 8) & 0xf) - m1
        dst[(y + o + 2) | 0] = d1 * ((v >>> 16) & 0xf) - m1
        dst[(y + o + 3) | 0] = d1 * ((v >>> 24) & 0xf) - m1
        dst[(y1 + o) | 0] = d2 * ((v >>> 4) & 0xf) - m2
        dst[(y1 + o + 1) | 0] = d2 * ((v >>> 12) & 0xf) - m2
        dst[(y1 + o + 2) | 0] = d2 * ((v >>> 20) & 0xf) - m2
        dst[(y1 + o + 3) | 0] = d2 * (v >>> 28) - m2
      }
      qp = qp + 8
      y = y + 64
    }
    p = p + 36
  }
}

function deqRowQ5_K(I32, p, dst, dstOff, cols) {
  var nb = cols >> 8
  var y = dstOff
  for (var i = 0; i < nb; i = i + 1) {
    var w0 = I32[p]
    var d = fp16Table[w0 & 0xffff]
    var dmin = fp16Table[w0 >>> 16]
    var w1 = I32[(p + 1) | 0]
    var w2 = I32[(p + 2) | 0]
    var w3 = I32[(p + 3) | 0]
    var hp = (p + 4) | 0
    var qp = (p + 12) | 0
    for (var c = 0; c < 4; c = c + 1) {
      var sm = kScaleMin(c << 1, w1, w2, w3)
      var d1 = d * (sm & 0xff)
      var m1 = dmin * (sm >>> 8)
      sm = kScaleMin((c << 1) + 1, w1, w2, w3)
      var d2 = d * (sm & 0xff)
      var m2 = dmin * (sm >>> 8)
      var b1 = c << 1
      var b2 = b1 + 1
      var y1 = (y + 32) | 0
      for (var k = 0; k < 8; k = k + 1) {
        var v = I32[(qp + k) | 0]
        var hv = I32[(hp + k) | 0]
        var o = (k << 2) | 0
        dst[(y + o) | 0] = d1 * ((v & 0xf) + (((hv >>> b1) & 1) << 4)) - m1
        dst[(y + o + 1) | 0] =
          d1 * (((v >>> 8) & 0xf) + (((hv >>> (b1 + 8)) & 1) << 4)) - m1
        dst[(y + o + 2) | 0] =
          d1 * (((v >>> 16) & 0xf) + (((hv >>> (b1 + 16)) & 1) << 4)) - m1
        dst[(y + o + 3) | 0] =
          d1 * (((v >>> 24) & 0xf) + (((hv >>> (b1 + 24)) & 1) << 4)) - m1
        dst[(y1 + o) | 0] = d2 * (((v >>> 4) & 0xf) + (((hv >>> b2) & 1) << 4)) - m2
        dst[(y1 + o + 1) | 0] =
          d2 * (((v >>> 12) & 0xf) + (((hv >>> (b2 + 8)) & 1) << 4)) - m2
        dst[(y1 + o + 2) | 0] =
          d2 * (((v >>> 20) & 0xf) + (((hv >>> (b2 + 16)) & 1) << 4)) - m2
        dst[(y1 + o + 3) | 0] =
          d2 * ((v >>> 28) + (((hv >>> (b2 + 24)) & 1) << 4)) - m2
      }
      qp = qp + 8
      y = y + 64
    }
    p = p + 44
  }
}

function deqRowQ6_K(U16, p, dst, dstOff, cols) {
  var nb = cols >> 8
  var y = dstOff
  for (var i = 0; i < nb; i = i + 1) {
    var d = fp16Table[U16[(p + 104) | 0]]
    for (var h = 0; h < 2; h = h + 1) {
      // half h: ql bytes 64h.., qh bytes 32h.., scales 8h.. (int8)
      var qlp = (p + (h << 5)) | 0
      var qhp = (p + 64 + (h << 4)) | 0
      var sp = (p + 96 + (h << 2)) | 0
      var sw0 = U16[sp]
      var sw1 = U16[(sp + 1) | 0]
      var sw2 = U16[(sp + 2) | 0]
      var sw3 = U16[(sp + 3) | 0]
      // is = 0 (l 0-15): low bytes of the scale words; is = 1 (l 16-31): high bytes
      var ds0 = d * ((sw0 << 24) >> 24)
      var ds2 = d * ((sw1 << 24) >> 24)
      var ds4 = d * ((sw2 << 24) >> 24)
      var ds6 = d * ((sw3 << 24) >> 24)
      var ds1 = d * ((sw0 << 16) >> 24)
      var ds3 = d * ((sw1 << 16) >> 24)
      var ds5 = d * ((sw2 << 16) >> 24)
      var ds7 = d * ((sw3 << 16) >> 24)
      var yb = (y + (h << 7)) | 0
      for (var k = 0; k < 16; k = k + 1) {
        var a = U16[(qlp + k) | 0]
        var b = U16[(qlp + 16 + k) | 0]
        var hv = U16[(qhp + k) | 0]
        var e0 = ds0
        var e2 = ds2
        var e4 = ds4
        var e6 = ds6
        if (k >= 8) {
          e0 = ds1
          e2 = ds3
          e4 = ds5
          e6 = ds7
        }
        var l = k << 1
        // byte 0 of the words: element l
        var ql1 = a & 0xff
        var ql2 = b & 0xff
        var qh = hv & 0xff
        dst[(yb + l) | 0] = e0 * (((ql1 & 0xf) | ((qh & 3) << 4)) - 32)
        dst[(yb + l + 32) | 0] = e2 * (((ql2 & 0xf) | (((qh >> 2) & 3) << 4)) - 32)
        dst[(yb + l + 64) | 0] = e4 * (((ql1 >> 4) | (((qh >> 4) & 3) << 4)) - 32)
        dst[(yb + l + 96) | 0] = e6 * (((ql2 >> 4) | ((qh >> 6) << 4)) - 32)
        // byte 1 of the words: element l + 1
        ql1 = a >>> 8
        ql2 = b >>> 8
        qh = hv >>> 8
        dst[(yb + l + 1) | 0] = e0 * (((ql1 & 0xf) | ((qh & 3) << 4)) - 32)
        dst[(yb + l + 33) | 0] = e2 * (((ql2 & 0xf) | (((qh >> 2) & 3) << 4)) - 32)
        dst[(yb + l + 65) | 0] = e4 * (((ql1 >> 4) | (((qh >> 4) & 3) << 4)) - 32)
        dst[(yb + l + 97) | 0] = e6 * (((ql2 >> 4) | ((qh >> 6) << 4)) - 32)
      }
    }
    y = y + 256
    p = p + 105
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

// Dequantize count K-quant values at a buffer offset (small tensors)
function deqKQuantAt(type, srcOffset, dst, count) {
  var bytes = getRowSize(count, type)
  var deqFunc = getDeqRowFunc(type)
  if (type === GGML_TYPE.Q3_K || type === GGML_TYPE.Q6_K) {
    deqFunc(new Uint16Array(ggufData, srcOffset, bytes >> 1), 0, dst, 0, count)
  } else {
    deqFunc(new Int32Array(ggufData, srcOffset, bytes >> 2), 0, dst, 0, count)
  }
}

function dequantizeTensor(srcOffset, count, type) {
  ensureFp16Table()
  var dst = new Float32Array(count)
  dequantizeRow(dst, srcOffset, count, type)
  return dst
}

// ----------------------------------------------------------------------------
// Fused quantized vector-matrix multiplication
// These compute dot products directly from quantized weights without full dequantization

// Values per block and bytes per block of the tensor types (index: GGML type)
var BLOCK_VALUES = [1, 1, 32, 32, 0, 0, 32, 32, 32, 0, 256, 256, 256, 256, 256]
var BLOCK_BYTES = [4, 2, 18, 20, 0, 0, 22, 24, 34, 0, 84, 110, 144, 176, 210]

// Get block size for quantization type
function getBlockSize(type) {
  if (type === GGML_TYPE.IQ4_NL) {
    return QK4_NL
  }
  return BLOCK_VALUES[type] || 1
}

// Get bytes per block for quantization type
function getTypeSize(type) {
  if (type === GGML_TYPE.IQ4_NL) {
    return 2 + QK4_NL / 2
  }
  if (type === GGML_TYPE.BF16 || type === 30) {
    return 2
  }
  return BLOCK_BYTES[type] || 0
}

// Get row size in bytes
function getRowSize(nCols, type) {
  var blockSize = getBlockSize(type)
  var typeSize = getTypeSize(type)
  return ((nCols / blockSize) | 0) * typeSize
}

// Dequantize a single row of a quantized tensor into dst
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
    case GGML_TYPE.Q3_K:
    case GGML_TYPE.Q4_K:
    case GGML_TYPE.Q5_K:
    case GGML_TYPE.Q6_K:
      deqKQuantAt(type, srcOffset, dst, nCols)
      break
    case GGML_TYPE.IQ4_NL:
      dequantizeIQ4_NL(srcOffset, dst, 0, nCols)
      break
    default:
      throw new Error("Unsupported quantization type: " + type)
  }
}

// Byte-wise Q8_0 row dot product (rows with an odd number of blocks)
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

// ----------------------------------------------------------------------------
// Block-32 prefill (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL): each row unpacked once to
// int8 + scales, then a 4-row x 3-token integer tile against the Q8_0 x

// Unpack one row: int8 weights w8[wOff..], scales sc[scOff + b], mins mn (*_1)
function unpackRowQ4_0(u8, bo, w8, wOff, sc, scOff, mn, mnOff, nb) {
  var wi = wOff
  for (var b = 0; b < nb; b = b + 1) {
    sc[scOff + b] = fp16Table[u8[bo] | (u8[bo + 1] << 8)]
    for (var j = 0; j < 16; j = j + 1) {
      var q = u8[bo + 2 + j]
      w8[wi + j] = (q & 0x0f) - 8
      w8[wi + j + 16] = (q >> 4) - 8
    }
    bo = bo + 18
    wi = wi + 32
  }
}

function unpackRowQ4_1(u8, bo, w8, wOff, sc, scOff, mn, mnOff, nb) {
  var wi = wOff
  for (var b = 0; b < nb; b = b + 1) {
    sc[scOff + b] = fp16Table[u8[bo] | (u8[bo + 1] << 8)]
    mn[mnOff + b] = fp16Table[u8[bo + 2] | (u8[bo + 3] << 8)]
    for (var j = 0; j < 16; j = j + 1) {
      var q = u8[bo + 4 + j]
      w8[wi + j] = q & 0x0f
      w8[wi + j + 16] = q >> 4
    }
    bo = bo + 20
    wi = wi + 32
  }
}

function unpackRowQ5_0(u8, bo, w8, wOff, sc, scOff, mn, mnOff, nb) {
  var wi = wOff
  for (var b = 0; b < nb; b = b + 1) {
    sc[scOff + b] = fp16Table[u8[bo] | (u8[bo + 1] << 8)]
    var qh = u8[bo + 2] | (u8[bo + 3] << 8) | (u8[bo + 4] << 16) | (u8[bo + 5] << 24)
    for (var j = 0; j < 16; j = j + 1) {
      var q = u8[bo + 6 + j]
      w8[wi + j] = ((q & 0x0f) | (((qh >> j) & 1) << 4)) - 16
      w8[wi + j + 16] = ((q >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16
    }
    bo = bo + 22
    wi = wi + 32
  }
}

function unpackRowQ5_1(u8, bo, w8, wOff, sc, scOff, mn, mnOff, nb) {
  var wi = wOff
  for (var b = 0; b < nb; b = b + 1) {
    sc[scOff + b] = fp16Table[u8[bo] | (u8[bo + 1] << 8)]
    mn[mnOff + b] = fp16Table[u8[bo + 2] | (u8[bo + 3] << 8)]
    var qh = u8[bo + 4] | (u8[bo + 5] << 8) | (u8[bo + 6] << 16) | (u8[bo + 7] << 24)
    for (var j = 0; j < 16; j = j + 1) {
      var q = u8[bo + 8 + j]
      w8[wi + j] = (q & 0x0f) | (((qh >> j) & 1) << 4)
      w8[wi + j + 16] = (q >> 4) | (((qh >> (j + 16)) & 1) << 4)
    }
    bo = bo + 24
    wi = wi + 32
  }
}

function unpackRowIQ4_NL(u8, bo, w8, wOff, sc, scOff, mn, mnOff, nb) {
  var wi = wOff
  var kv = kvalues_iq4nl
  for (var b = 0; b < nb; b = b + 1) {
    sc[scOff + b] = fp16Table[u8[bo] | (u8[bo + 1] << 8)]
    for (var j = 0; j < 16; j = j + 1) {
      var q = u8[bo + 2 + j]
      w8[wi + j] = kv[q & 0x0f]
      w8[wi + j + 16] = kv[q >> 4]
    }
    bo = bo + 18
    wi = wi + 32
  }
}

function getUnpackRowFunc(type) {
  switch (type) {
    case GGML_TYPE.Q4_0:
      return unpackRowQ4_0
    case GGML_TYPE.Q4_1:
      return unpackRowQ4_1
    case GGML_TYPE.Q5_0:
      return unpackRowQ5_0
    case GGML_TYPE.Q5_1:
      return unpackRowQ5_1
    case GGML_TYPE.IQ4_NL:
      return unpackRowIQ4_NL
    default:
      return null
  }
}

// Per-token, per-block integer sums of the quantized x (used by the *_1 formats)
function block32XSums(bQ8i8, batchSize, nb, sc, xsBase) {
  for (var b = 0; b < batchSize; b = b + 1) {
    var xi8 = bQ8i8[b]
    var xo = 2
    var xsOff = xsBase + b * nb
    for (var blk = 0; blk < nb; blk = blk + 1) {
      var t = 0
      for (var k = 0; k < 32; k = k + 1) {
        t = t + xi8[xo + k]
      }
      sc[xsOff + blk] = t
      xo = xo + 34
    }
  }
}

// Integer 4-row x 3-token tile for tokens bt..bt+nTok-1 (a short group repeats
// its last token)
function block32Tile(
  outs,
  bt,
  nTok,
  i,
  nb,
  cols,
  hasMin,
  sc,
  scBase,
  mnBase,
  xsBase
) {
  var bQ8 = state.batchQ8
  var bQ8i8 = state.batchQ8i8
  var w8 = matmulDeqI8
  var b1 = nTok > 1 ? bt + 1 : bt
  var b2 = nTok > 2 ? bt + 2 : b1
  var xa = bQ8i8[bt]
  var xb = bQ8i8[b1]
  var xc = bQ8i8[b2]
  var ua = bQ8[bt]
  var ub = bQ8[b1]
  var uc = bQ8[b2]
  var wOff1 = cols
  var wOff2 = cols + cols
  var wOff3 = wOff2 + cols
  var s00 = 0.0
  var s01 = 0.0
  var s02 = 0.0
  var s10 = 0.0
  var s11 = 0.0
  var s12 = 0.0
  var s20 = 0.0
  var s21 = 0.0
  var s22 = 0.0
  var s30 = 0.0
  var s31 = 0.0
  var s32 = 0.0
  var xo = 2
  var wo = 0
  for (var blk = 0; blk < nb; blk = blk + 1) {
    var i00 = 0
    var i01 = 0
    var i02 = 0
    var i10 = 0
    var i11 = 0
    var i12 = 0
    var i20 = 0
    var i21 = 0
    var i22 = 0
    var i30 = 0
    var i31 = 0
    var i32 = 0
    for (var k = 0; k < 32; k = k + 1) {
      var x0 = xa[(xo + k) | 0]
      var x1 = xb[(xo + k) | 0]
      var x2 = xc[(xo + k) | 0]
      var w0 = w8[(wo + k) | 0]
      var w1 = w8[(wo + wOff1 + k) | 0]
      var w2 = w8[(wo + wOff2 + k) | 0]
      var w3 = w8[(wo + wOff3 + k) | 0]
      i00 = (i00 + w0 * x0) | 0
      i01 = (i01 + w0 * x1) | 0
      i02 = (i02 + w0 * x2) | 0
      i10 = (i10 + w1 * x0) | 0
      i11 = (i11 + w1 * x1) | 0
      i12 = (i12 + w1 * x2) | 0
      i20 = (i20 + w2 * x0) | 0
      i21 = (i21 + w2 * x1) | 0
      i22 = (i22 + w2 * x2) | 0
      i30 = (i30 + w3 * x0) | 0
      i31 = (i31 + w3 * x1) | 0
      i32 = (i32 + w3 * x2) | 0
    }
    var dxa = fp16Table[ua[xo - 2] | (ua[xo - 1] << 8)]
    var dxb = fp16Table[ub[xo - 2] | (ub[xo - 1] << 8)]
    var dxc = fp16Table[uc[xo - 2] | (uc[xo - 1] << 8)]
    var d0 = sc[scBase + blk]
    var d1 = sc[scBase + nb + blk]
    var d2 = sc[scBase + 2 * nb + blk]
    var d3 = sc[scBase + 3 * nb + blk]
    s00 = s00 + d0 * dxa * i00
    s01 = s01 + d0 * dxb * i01
    s02 = s02 + d0 * dxc * i02
    s10 = s10 + d1 * dxa * i10
    s11 = s11 + d1 * dxb * i11
    s12 = s12 + d1 * dxc * i12
    s20 = s20 + d2 * dxa * i20
    s21 = s21 + d2 * dxb * i21
    s22 = s22 + d2 * dxc * i22
    s30 = s30 + d3 * dxa * i30
    s31 = s31 + d3 * dxb * i31
    s32 = s32 + d3 * dxc * i32
    if (hasMin) {
      var m0 = sc[mnBase + blk]
      var m1 = sc[mnBase + nb + blk]
      var m2 = sc[mnBase + 2 * nb + blk]
      var m3 = sc[mnBase + 3 * nb + blk]
      var xsa = sc[xsBase + bt * nb + blk]
      var xsb = sc[xsBase + b1 * nb + blk]
      var xsc = sc[xsBase + b2 * nb + blk]
      s00 = s00 + m0 * dxa * xsa
      s01 = s01 + m0 * dxb * xsb
      s02 = s02 + m0 * dxc * xsc
      s10 = s10 + m1 * dxa * xsa
      s11 = s11 + m1 * dxb * xsb
      s12 = s12 + m1 * dxc * xsc
      s20 = s20 + m2 * dxa * xsa
      s21 = s21 + m2 * dxb * xsb
      s22 = s22 + m2 * dxc * xsc
      s30 = s30 + m3 * dxa * xsa
      s31 = s31 + m3 * dxb * xsb
      s32 = s32 + m3 * dxc * xsc
    }
    xo = xo + 34
    wo = wo + 32
  }
  var oA = outs[bt]
  oA[i] = s00
  oA[i + 1] = s10
  oA[i + 2] = s20
  oA[i + 3] = s30
  if (nTok > 1) {
    var oB = outs[bt + 1]
    oB[i] = s01
    oB[i + 1] = s11
    oB[i + 2] = s21
    oB[i + 3] = s31
  }
  if (nTok > 2) {
    var oC = outs[bt + 2]
    oC[i] = s02
    oC[i + 1] = s12
    oC[i + 2] = s22
    oC[i + 3] = s32
  }
}

// Scratch layout in matmulDeqBuf: int8 weights of the 4 rows, then (doubles
// from cols/2) 4*nb scales, 4*nb mins (Q4_1 / Q5_1), batch*nb x block sums
function matmulBlock32Q8Batch(outs, xs, qw, batchSize) {
  var rows = qw.rows
  var cols = qw.cols
  var rowSize = qw.rowSize
  var base = qw.dataOffset
  var nb = cols >> 5
  var unpack = qw.k.unpackRowFunc
  var hasMin = qw.type === GGML_TYPE.Q4_1 || qw.type === GGML_TYPE.Q5_1
  var u8 = ggufUint8
  var w8 = matmulDeqI8
  var sc = matmulDeqBuf
  var scBase = cols >> 1
  var mnBase = scBase + 4 * nb
  var xsBase = mnBase + 4 * nb
  var bQ8 = state.batchQ8
  var bQ8i8 = state.batchQ8i8

  // Quantize the activations once (Q8_0 blocks: 2-byte scale + 32 int8)
  for (var b = 0; b < batchSize; b = b + 1) {
    quantizeToQ8_0Cache(xs[b], 0, bQ8[b], bQ8i8[b], 0, cols)
  }
  if (hasMin) {
    block32XSums(bQ8i8, batchSize, nb, sc, xsBase)
  }

  for (var i = 0; i < rows; i = i + 4) {
    var ro = base + i * rowSize
    unpack(u8, ro, w8, 0, sc, scBase, sc, mnBase, nb)
    unpack(u8, ro + rowSize, w8, cols, sc, scBase + nb, sc, mnBase + nb, nb)
    unpack(
      u8,
      ro + rowSize + rowSize,
      w8,
      cols + cols,
      sc,
      scBase + 2 * nb,
      sc,
      mnBase + 2 * nb,
      nb
    )
    unpack(
      u8,
      ro + rowSize + rowSize + rowSize,
      w8,
      cols + cols + cols,
      sc,
      scBase + 3 * nb,
      sc,
      mnBase + 3 * nb,
      nb
    )
    for (var bt = 0; bt < batchSize; bt = bt + 3) {
      var nTok = batchSize - bt
      if (nTok > 3) {
        nTok = 3
      }
      block32Tile(outs, bt, nTok, i, nb, cols, hasMin, sc, scBase, mnBase, xsBase)
    }
  }
}

// ----------------------------------------------------------------------------
// Single-token block-32 kernels: rows i..i+3 per call, weights through the
// Uint16 view, x quantized once; same math as the prefill tile

function matmulQ4_0Q8Rows4(out, qw, i) {
  var U16 = qw.view
  var rowWords = qw.rowSize >> 1
  var nb = qw.cols >> 5
  var xq = xQ8Int8Buf
  var xu = xQ8Buf
  var p0 = qw.base + i * rowWords
  var p1 = p0 + rowWords
  var p2 = p1 + rowWords
  var p3 = p2 + rowWords
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xo = 2
  for (var b = 0; b < nb; b = b + 1) {
    var dx = fp16Table[xu[xo - 2] | (xu[xo - 1] << 8)]
    var d0 = fp16Table[U16[p0]]
    var d1 = fp16Table[U16[p1]]
    var d2 = fp16Table[U16[p2]]
    var d3 = fp16Table[U16[p3]]
    var i0 = 0
    var i1 = 0
    var i2 = 0
    var i3 = 0
    var xs = 0
    var xg = xo
    var wk = 1
    // weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
    for (var g = 0; g < 4; g = g + 1) {
      var x0 = xq[xg]
      var y0 = xq[(xg + 16) | 0]
      var x1 = xq[(xg + 1) | 0]
      var y1 = xq[(xg + 17) | 0]
      var x2 = xq[(xg + 2) | 0]
      var y2 = xq[(xg + 18) | 0]
      var x3 = xq[(xg + 3) | 0]
      var y3 = xq[(xg + 19) | 0]
      xs = (xs + x0 + x1 + x2 + x3 + y0 + y1 + y2 + y3) | 0
      var w0 = U16[(p0 + wk) | 0]
      var v0 = U16[(p0 + wk + 1) | 0]
      i0 =
        (i0 +
          (w0 & 0xf) * x0 +
          ((w0 >>> 4) & 0xf) * y0 +
          ((w0 >>> 8) & 0xf) * x1 +
          (w0 >>> 12) * y1 +
          (v0 & 0xf) * x2 +
          ((v0 >>> 4) & 0xf) * y2 +
          ((v0 >>> 8) & 0xf) * x3 +
          (v0 >>> 12) * y3) |
        0
      var w1 = U16[(p1 + wk) | 0]
      var v1 = U16[(p1 + wk + 1) | 0]
      i1 =
        (i1 +
          (w1 & 0xf) * x0 +
          ((w1 >>> 4) & 0xf) * y0 +
          ((w1 >>> 8) & 0xf) * x1 +
          (w1 >>> 12) * y1 +
          (v1 & 0xf) * x2 +
          ((v1 >>> 4) & 0xf) * y2 +
          ((v1 >>> 8) & 0xf) * x3 +
          (v1 >>> 12) * y3) |
        0
      var w2 = U16[(p2 + wk) | 0]
      var v2 = U16[(p2 + wk + 1) | 0]
      i2 =
        (i2 +
          (w2 & 0xf) * x0 +
          ((w2 >>> 4) & 0xf) * y0 +
          ((w2 >>> 8) & 0xf) * x1 +
          (w2 >>> 12) * y1 +
          (v2 & 0xf) * x2 +
          ((v2 >>> 4) & 0xf) * y2 +
          ((v2 >>> 8) & 0xf) * x3 +
          (v2 >>> 12) * y3) |
        0
      var w3 = U16[(p3 + wk) | 0]
      var v3 = U16[(p3 + wk + 1) | 0]
      i3 =
        (i3 +
          (w3 & 0xf) * x0 +
          ((w3 >>> 4) & 0xf) * y0 +
          ((w3 >>> 8) & 0xf) * x1 +
          (w3 >>> 12) * y1 +
          (v3 & 0xf) * x2 +
          ((v3 >>> 4) & 0xf) * y2 +
          ((v3 >>> 8) & 0xf) * x3 +
          (v3 >>> 12) * y3) |
        0
      xg = xg + 4
      wk = wk + 2
    }
    i0 = i0 - xs * 8
    i1 = i1 - xs * 8
    i2 = i2 - xs * 8
    i3 = i3 - xs * 8
    s0 = s0 + d0 * dx * i0
    s1 = s1 + d1 * dx * i1
    s2 = s2 + d2 * dx * i2
    s3 = s3 + d3 * dx * i3
    xo = xo + 34
    p0 = p0 + 9
    p1 = p1 + 9
    p2 = p2 + 9
    p3 = p3 + 9
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

function matmulQ4_1Q8Rows4(out, qw, i) {
  var U16 = qw.view
  var rowWords = qw.rowSize >> 1
  var nb = qw.cols >> 5
  var xq = xQ8Int8Buf
  var xu = xQ8Buf
  var p0 = qw.base + i * rowWords
  var p1 = p0 + rowWords
  var p2 = p1 + rowWords
  var p3 = p2 + rowWords
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xo = 2
  for (var b = 0; b < nb; b = b + 1) {
    var dx = fp16Table[xu[xo - 2] | (xu[xo - 1] << 8)]
    var d0 = fp16Table[U16[p0]]
    var d1 = fp16Table[U16[p1]]
    var d2 = fp16Table[U16[p2]]
    var d3 = fp16Table[U16[p3]]
    var m0 = fp16Table[U16[(p0 + 1) | 0]]
    var m1 = fp16Table[U16[(p1 + 1) | 0]]
    var m2 = fp16Table[U16[(p2 + 1) | 0]]
    var m3 = fp16Table[U16[(p3 + 1) | 0]]
    var i0 = 0
    var i1 = 0
    var i2 = 0
    var i3 = 0
    var xs = 0
    var xg = xo
    var wk = 2
    // weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
    for (var g = 0; g < 4; g = g + 1) {
      var x0 = xq[xg]
      var y0 = xq[(xg + 16) | 0]
      var x1 = xq[(xg + 1) | 0]
      var y1 = xq[(xg + 17) | 0]
      var x2 = xq[(xg + 2) | 0]
      var y2 = xq[(xg + 18) | 0]
      var x3 = xq[(xg + 3) | 0]
      var y3 = xq[(xg + 19) | 0]
      xs = (xs + x0 + x1 + x2 + x3 + y0 + y1 + y2 + y3) | 0
      var w0 = U16[(p0 + wk) | 0]
      var v0 = U16[(p0 + wk + 1) | 0]
      i0 =
        (i0 +
          (w0 & 0xf) * x0 +
          ((w0 >>> 4) & 0xf) * y0 +
          ((w0 >>> 8) & 0xf) * x1 +
          (w0 >>> 12) * y1 +
          (v0 & 0xf) * x2 +
          ((v0 >>> 4) & 0xf) * y2 +
          ((v0 >>> 8) & 0xf) * x3 +
          (v0 >>> 12) * y3) |
        0
      var w1 = U16[(p1 + wk) | 0]
      var v1 = U16[(p1 + wk + 1) | 0]
      i1 =
        (i1 +
          (w1 & 0xf) * x0 +
          ((w1 >>> 4) & 0xf) * y0 +
          ((w1 >>> 8) & 0xf) * x1 +
          (w1 >>> 12) * y1 +
          (v1 & 0xf) * x2 +
          ((v1 >>> 4) & 0xf) * y2 +
          ((v1 >>> 8) & 0xf) * x3 +
          (v1 >>> 12) * y3) |
        0
      var w2 = U16[(p2 + wk) | 0]
      var v2 = U16[(p2 + wk + 1) | 0]
      i2 =
        (i2 +
          (w2 & 0xf) * x0 +
          ((w2 >>> 4) & 0xf) * y0 +
          ((w2 >>> 8) & 0xf) * x1 +
          (w2 >>> 12) * y1 +
          (v2 & 0xf) * x2 +
          ((v2 >>> 4) & 0xf) * y2 +
          ((v2 >>> 8) & 0xf) * x3 +
          (v2 >>> 12) * y3) |
        0
      var w3 = U16[(p3 + wk) | 0]
      var v3 = U16[(p3 + wk + 1) | 0]
      i3 =
        (i3 +
          (w3 & 0xf) * x0 +
          ((w3 >>> 4) & 0xf) * y0 +
          ((w3 >>> 8) & 0xf) * x1 +
          (w3 >>> 12) * y1 +
          (v3 & 0xf) * x2 +
          ((v3 >>> 4) & 0xf) * y2 +
          ((v3 >>> 8) & 0xf) * x3 +
          (v3 >>> 12) * y3) |
        0
      xg = xg + 4
      wk = wk + 2
    }
    s0 = s0 + d0 * dx * i0
    s1 = s1 + d1 * dx * i1
    s2 = s2 + d2 * dx * i2
    s3 = s3 + d3 * dx * i3
    s0 = s0 + m0 * dx * xs
    s1 = s1 + m1 * dx * xs
    s2 = s2 + m2 * dx * xs
    s3 = s3 + m3 * dx * xs
    xo = xo + 34
    p0 = p0 + 10
    p1 = p1 + 10
    p2 = p2 + 10
    p3 = p3 + 10
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

function matmulQ5_0Q8Rows4(out, qw, i) {
  var U16 = qw.view
  var rowWords = qw.rowSize >> 1
  var nb = qw.cols >> 5
  var xq = xQ8Int8Buf
  var xu = xQ8Buf
  var p0 = qw.base + i * rowWords
  var p1 = p0 + rowWords
  var p2 = p1 + rowWords
  var p3 = p2 + rowWords
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xo = 2
  for (var b = 0; b < nb; b = b + 1) {
    var dx = fp16Table[xu[xo - 2] | (xu[xo - 1] << 8)]
    var d0 = fp16Table[U16[p0]]
    var d1 = fp16Table[U16[p1]]
    var d2 = fp16Table[U16[p2]]
    var d3 = fp16Table[U16[p3]]
    var h0 = U16[(p0 + 1) | 0] | (U16[(p0 + 2) | 0] << 16)
    var h1 = U16[(p1 + 1) | 0] | (U16[(p1 + 2) | 0] << 16)
    var h2 = U16[(p2 + 1) | 0] | (U16[(p2 + 2) | 0] << 16)
    var h3 = U16[(p3 + 1) | 0] | (U16[(p3 + 2) | 0] << 16)
    var i0 = 0
    var i1 = 0
    var i2 = 0
    var i3 = 0
    var xs = 0
    var xg = xo
    var wk = 3
    // weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
    for (var g = 0; g < 4; g = g + 1) {
      var x0 = xq[xg]
      var y0 = xq[(xg + 16) | 0]
      var x1 = xq[(xg + 1) | 0]
      var y1 = xq[(xg + 17) | 0]
      var x2 = xq[(xg + 2) | 0]
      var y2 = xq[(xg + 18) | 0]
      var x3 = xq[(xg + 3) | 0]
      var y3 = xq[(xg + 19) | 0]
      xs = (xs + x0 + x1 + x2 + x3 + y0 + y1 + y2 + y3) | 0
      var w0 = U16[(p0 + wk) | 0]
      var v0 = U16[(p0 + wk + 1) | 0]
      var hs0 = h0 >>> (g << 2)
      i0 =
        (i0 +
          ((w0 & 0xf) | ((hs0 << 4) & 0x10)) * x0 +
          (((w0 >>> 4) & 0xf) | ((hs0 >>> 12) & 0x10)) * y0 +
          (((w0 >>> 8) & 0xf) | ((hs0 << 3) & 0x10)) * x1 +
          ((w0 >>> 12) | ((hs0 >>> 13) & 0x10)) * y1 +
          ((v0 & 0xf) | ((hs0 << 2) & 0x10)) * x2 +
          (((v0 >>> 4) & 0xf) | ((hs0 >>> 14) & 0x10)) * y2 +
          (((v0 >>> 8) & 0xf) | ((hs0 << 1) & 0x10)) * x3 +
          ((v0 >>> 12) | ((hs0 >>> 15) & 0x10)) * y3) |
        0
      var w1 = U16[(p1 + wk) | 0]
      var v1 = U16[(p1 + wk + 1) | 0]
      var hs1 = h1 >>> (g << 2)
      i1 =
        (i1 +
          ((w1 & 0xf) | ((hs1 << 4) & 0x10)) * x0 +
          (((w1 >>> 4) & 0xf) | ((hs1 >>> 12) & 0x10)) * y0 +
          (((w1 >>> 8) & 0xf) | ((hs1 << 3) & 0x10)) * x1 +
          ((w1 >>> 12) | ((hs1 >>> 13) & 0x10)) * y1 +
          ((v1 & 0xf) | ((hs1 << 2) & 0x10)) * x2 +
          (((v1 >>> 4) & 0xf) | ((hs1 >>> 14) & 0x10)) * y2 +
          (((v1 >>> 8) & 0xf) | ((hs1 << 1) & 0x10)) * x3 +
          ((v1 >>> 12) | ((hs1 >>> 15) & 0x10)) * y3) |
        0
      var w2 = U16[(p2 + wk) | 0]
      var v2 = U16[(p2 + wk + 1) | 0]
      var hs2 = h2 >>> (g << 2)
      i2 =
        (i2 +
          ((w2 & 0xf) | ((hs2 << 4) & 0x10)) * x0 +
          (((w2 >>> 4) & 0xf) | ((hs2 >>> 12) & 0x10)) * y0 +
          (((w2 >>> 8) & 0xf) | ((hs2 << 3) & 0x10)) * x1 +
          ((w2 >>> 12) | ((hs2 >>> 13) & 0x10)) * y1 +
          ((v2 & 0xf) | ((hs2 << 2) & 0x10)) * x2 +
          (((v2 >>> 4) & 0xf) | ((hs2 >>> 14) & 0x10)) * y2 +
          (((v2 >>> 8) & 0xf) | ((hs2 << 1) & 0x10)) * x3 +
          ((v2 >>> 12) | ((hs2 >>> 15) & 0x10)) * y3) |
        0
      var w3 = U16[(p3 + wk) | 0]
      var v3 = U16[(p3 + wk + 1) | 0]
      var hs3 = h3 >>> (g << 2)
      i3 =
        (i3 +
          ((w3 & 0xf) | ((hs3 << 4) & 0x10)) * x0 +
          (((w3 >>> 4) & 0xf) | ((hs3 >>> 12) & 0x10)) * y0 +
          (((w3 >>> 8) & 0xf) | ((hs3 << 3) & 0x10)) * x1 +
          ((w3 >>> 12) | ((hs3 >>> 13) & 0x10)) * y1 +
          ((v3 & 0xf) | ((hs3 << 2) & 0x10)) * x2 +
          (((v3 >>> 4) & 0xf) | ((hs3 >>> 14) & 0x10)) * y2 +
          (((v3 >>> 8) & 0xf) | ((hs3 << 1) & 0x10)) * x3 +
          ((v3 >>> 12) | ((hs3 >>> 15) & 0x10)) * y3) |
        0
      xg = xg + 4
      wk = wk + 2
    }
    i0 = i0 - xs * 16
    i1 = i1 - xs * 16
    i2 = i2 - xs * 16
    i3 = i3 - xs * 16
    s0 = s0 + d0 * dx * i0
    s1 = s1 + d1 * dx * i1
    s2 = s2 + d2 * dx * i2
    s3 = s3 + d3 * dx * i3
    xo = xo + 34
    p0 = p0 + 11
    p1 = p1 + 11
    p2 = p2 + 11
    p3 = p3 + 11
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

function matmulQ5_1Q8Rows4(out, qw, i) {
  var U16 = qw.view
  var rowWords = qw.rowSize >> 1
  var nb = qw.cols >> 5
  var xq = xQ8Int8Buf
  var xu = xQ8Buf
  var p0 = qw.base + i * rowWords
  var p1 = p0 + rowWords
  var p2 = p1 + rowWords
  var p3 = p2 + rowWords
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xo = 2
  for (var b = 0; b < nb; b = b + 1) {
    var dx = fp16Table[xu[xo - 2] | (xu[xo - 1] << 8)]
    var d0 = fp16Table[U16[p0]]
    var d1 = fp16Table[U16[p1]]
    var d2 = fp16Table[U16[p2]]
    var d3 = fp16Table[U16[p3]]
    var m0 = fp16Table[U16[(p0 + 1) | 0]]
    var m1 = fp16Table[U16[(p1 + 1) | 0]]
    var m2 = fp16Table[U16[(p2 + 1) | 0]]
    var m3 = fp16Table[U16[(p3 + 1) | 0]]
    var h0 = U16[(p0 + 2) | 0] | (U16[(p0 + 3) | 0] << 16)
    var h1 = U16[(p1 + 2) | 0] | (U16[(p1 + 3) | 0] << 16)
    var h2 = U16[(p2 + 2) | 0] | (U16[(p2 + 3) | 0] << 16)
    var h3 = U16[(p3 + 2) | 0] | (U16[(p3 + 3) | 0] << 16)
    var i0 = 0
    var i1 = 0
    var i2 = 0
    var i3 = 0
    var xs = 0
    var xg = xo
    var wk = 4
    // weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
    for (var g = 0; g < 4; g = g + 1) {
      var x0 = xq[xg]
      var y0 = xq[(xg + 16) | 0]
      var x1 = xq[(xg + 1) | 0]
      var y1 = xq[(xg + 17) | 0]
      var x2 = xq[(xg + 2) | 0]
      var y2 = xq[(xg + 18) | 0]
      var x3 = xq[(xg + 3) | 0]
      var y3 = xq[(xg + 19) | 0]
      xs = (xs + x0 + x1 + x2 + x3 + y0 + y1 + y2 + y3) | 0
      var w0 = U16[(p0 + wk) | 0]
      var v0 = U16[(p0 + wk + 1) | 0]
      var hs0 = h0 >>> (g << 2)
      i0 =
        (i0 +
          ((w0 & 0xf) | ((hs0 << 4) & 0x10)) * x0 +
          (((w0 >>> 4) & 0xf) | ((hs0 >>> 12) & 0x10)) * y0 +
          (((w0 >>> 8) & 0xf) | ((hs0 << 3) & 0x10)) * x1 +
          ((w0 >>> 12) | ((hs0 >>> 13) & 0x10)) * y1 +
          ((v0 & 0xf) | ((hs0 << 2) & 0x10)) * x2 +
          (((v0 >>> 4) & 0xf) | ((hs0 >>> 14) & 0x10)) * y2 +
          (((v0 >>> 8) & 0xf) | ((hs0 << 1) & 0x10)) * x3 +
          ((v0 >>> 12) | ((hs0 >>> 15) & 0x10)) * y3) |
        0
      var w1 = U16[(p1 + wk) | 0]
      var v1 = U16[(p1 + wk + 1) | 0]
      var hs1 = h1 >>> (g << 2)
      i1 =
        (i1 +
          ((w1 & 0xf) | ((hs1 << 4) & 0x10)) * x0 +
          (((w1 >>> 4) & 0xf) | ((hs1 >>> 12) & 0x10)) * y0 +
          (((w1 >>> 8) & 0xf) | ((hs1 << 3) & 0x10)) * x1 +
          ((w1 >>> 12) | ((hs1 >>> 13) & 0x10)) * y1 +
          ((v1 & 0xf) | ((hs1 << 2) & 0x10)) * x2 +
          (((v1 >>> 4) & 0xf) | ((hs1 >>> 14) & 0x10)) * y2 +
          (((v1 >>> 8) & 0xf) | ((hs1 << 1) & 0x10)) * x3 +
          ((v1 >>> 12) | ((hs1 >>> 15) & 0x10)) * y3) |
        0
      var w2 = U16[(p2 + wk) | 0]
      var v2 = U16[(p2 + wk + 1) | 0]
      var hs2 = h2 >>> (g << 2)
      i2 =
        (i2 +
          ((w2 & 0xf) | ((hs2 << 4) & 0x10)) * x0 +
          (((w2 >>> 4) & 0xf) | ((hs2 >>> 12) & 0x10)) * y0 +
          (((w2 >>> 8) & 0xf) | ((hs2 << 3) & 0x10)) * x1 +
          ((w2 >>> 12) | ((hs2 >>> 13) & 0x10)) * y1 +
          ((v2 & 0xf) | ((hs2 << 2) & 0x10)) * x2 +
          (((v2 >>> 4) & 0xf) | ((hs2 >>> 14) & 0x10)) * y2 +
          (((v2 >>> 8) & 0xf) | ((hs2 << 1) & 0x10)) * x3 +
          ((v2 >>> 12) | ((hs2 >>> 15) & 0x10)) * y3) |
        0
      var w3 = U16[(p3 + wk) | 0]
      var v3 = U16[(p3 + wk + 1) | 0]
      var hs3 = h3 >>> (g << 2)
      i3 =
        (i3 +
          ((w3 & 0xf) | ((hs3 << 4) & 0x10)) * x0 +
          (((w3 >>> 4) & 0xf) | ((hs3 >>> 12) & 0x10)) * y0 +
          (((w3 >>> 8) & 0xf) | ((hs3 << 3) & 0x10)) * x1 +
          ((w3 >>> 12) | ((hs3 >>> 13) & 0x10)) * y1 +
          ((v3 & 0xf) | ((hs3 << 2) & 0x10)) * x2 +
          (((v3 >>> 4) & 0xf) | ((hs3 >>> 14) & 0x10)) * y2 +
          (((v3 >>> 8) & 0xf) | ((hs3 << 1) & 0x10)) * x3 +
          ((v3 >>> 12) | ((hs3 >>> 15) & 0x10)) * y3) |
        0
      xg = xg + 4
      wk = wk + 2
    }
    s0 = s0 + d0 * dx * i0
    s1 = s1 + d1 * dx * i1
    s2 = s2 + d2 * dx * i2
    s3 = s3 + d3 * dx * i3
    s0 = s0 + m0 * dx * xs
    s1 = s1 + m1 * dx * xs
    s2 = s2 + m2 * dx * xs
    s3 = s3 + m3 * dx * xs
    xo = xo + 34
    p0 = p0 + 12
    p1 = p1 + 12
    p2 = p2 + 12
    p3 = p3 + 12
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

function matmulIQ4_NLQ8Rows4(out, qw, i) {
  var U16 = qw.view
  var rowWords = qw.rowSize >> 1
  var nb = qw.cols >> 5
  var xq = xQ8Int8Buf
  var xu = xQ8Buf
  var kv = kvalues_iq4nl
  var p0 = qw.base + i * rowWords
  var p1 = p0 + rowWords
  var p2 = p1 + rowWords
  var p3 = p2 + rowWords
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xo = 2
  for (var b = 0; b < nb; b = b + 1) {
    var dx = fp16Table[xu[xo - 2] | (xu[xo - 1] << 8)]
    var d0 = fp16Table[U16[p0]]
    var d1 = fp16Table[U16[p1]]
    var d2 = fp16Table[U16[p2]]
    var d3 = fp16Table[U16[p3]]
    var i0 = 0
    var i1 = 0
    var i2 = 0
    var i3 = 0
    var xg = xo
    var wk = 1
    // weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
    for (var g = 0; g < 4; g = g + 1) {
      var x0 = xq[xg]
      var y0 = xq[(xg + 16) | 0]
      var x1 = xq[(xg + 1) | 0]
      var y1 = xq[(xg + 17) | 0]
      var x2 = xq[(xg + 2) | 0]
      var y2 = xq[(xg + 18) | 0]
      var x3 = xq[(xg + 3) | 0]
      var y3 = xq[(xg + 19) | 0]
      var w0 = U16[(p0 + wk) | 0]
      var v0 = U16[(p0 + wk + 1) | 0]
      i0 =
        (i0 +
          kv[w0 & 0xf] * x0 +
          kv[(w0 >>> 4) & 0xf] * y0 +
          kv[(w0 >>> 8) & 0xf] * x1 +
          kv[w0 >>> 12] * y1 +
          kv[v0 & 0xf] * x2 +
          kv[(v0 >>> 4) & 0xf] * y2 +
          kv[(v0 >>> 8) & 0xf] * x3 +
          kv[v0 >>> 12] * y3) |
        0
      var w1 = U16[(p1 + wk) | 0]
      var v1 = U16[(p1 + wk + 1) | 0]
      i1 =
        (i1 +
          kv[w1 & 0xf] * x0 +
          kv[(w1 >>> 4) & 0xf] * y0 +
          kv[(w1 >>> 8) & 0xf] * x1 +
          kv[w1 >>> 12] * y1 +
          kv[v1 & 0xf] * x2 +
          kv[(v1 >>> 4) & 0xf] * y2 +
          kv[(v1 >>> 8) & 0xf] * x3 +
          kv[v1 >>> 12] * y3) |
        0
      var w2 = U16[(p2 + wk) | 0]
      var v2 = U16[(p2 + wk + 1) | 0]
      i2 =
        (i2 +
          kv[w2 & 0xf] * x0 +
          kv[(w2 >>> 4) & 0xf] * y0 +
          kv[(w2 >>> 8) & 0xf] * x1 +
          kv[w2 >>> 12] * y1 +
          kv[v2 & 0xf] * x2 +
          kv[(v2 >>> 4) & 0xf] * y2 +
          kv[(v2 >>> 8) & 0xf] * x3 +
          kv[v2 >>> 12] * y3) |
        0
      var w3 = U16[(p3 + wk) | 0]
      var v3 = U16[(p3 + wk + 1) | 0]
      i3 =
        (i3 +
          kv[w3 & 0xf] * x0 +
          kv[(w3 >>> 4) & 0xf] * y0 +
          kv[(w3 >>> 8) & 0xf] * x1 +
          kv[w3 >>> 12] * y1 +
          kv[v3 & 0xf] * x2 +
          kv[(v3 >>> 4) & 0xf] * y2 +
          kv[(v3 >>> 8) & 0xf] * x3 +
          kv[v3 >>> 12] * y3) |
        0
      xg = xg + 4
      wk = wk + 2
    }
    s0 = s0 + d0 * dx * i0
    s1 = s1 + d1 * dx * i1
    s2 = s2 + d2 * dx * i2
    s3 = s3 + d3 * dx * i3
    xo = xo + 34
    p0 = p0 + 9
    p1 = p1 + 9
    p2 = p2 + 9
    p3 = p3 + 9
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

function getDeqRowFunc(type) {
  switch (type) {
    case GGML_TYPE.Q2_K:
      return deqRowQ2_K
    case GGML_TYPE.Q3_K:
      return deqRowQ3_K
    case GGML_TYPE.Q4_K:
      return deqRowQ4_K
    case GGML_TYPE.Q5_K:
      return deqRowQ5_K
    case GGML_TYPE.Q6_K:
      return deqRowQ6_K
    case GGML_TYPE.F16:
      return deqRowF16
    case GGML_TYPE.BF16:
    case 30:
      ensureBf16Table()
      return deqRowBF16
    case GGML_TYPE.F32:
      return deqRowF32
    default:
      return null
  }
}

// Single-token group kernel: (out, qw, i) block-32 (4 rows), (out, x, qw, i)
// fused (4 rows, 2 for F32); null for Q8_0 and Q2_K
function getRowsFunc(type) {
  switch (type) {
    case GGML_TYPE.Q4_0:
      return matmulQ4_0Q8Rows4
    case GGML_TYPE.Q4_1:
      return matmulQ4_1Q8Rows4
    case GGML_TYPE.Q5_0:
      return matmulQ5_0Q8Rows4
    case GGML_TYPE.Q5_1:
      return matmulQ5_1Q8Rows4
    case GGML_TYPE.IQ4_NL:
      return matmulIQ4_NLQ8Rows4
    case GGML_TYPE.Q4_K:
      return matmulQ4_KRows
    case GGML_TYPE.Q5_K:
      return matmulQ5_KRows
    case GGML_TYPE.Q3_K:
    case GGML_TYPE.Q6_K:
      return matmulK16Rows
    case GGML_TYPE.F16:
    case GGML_TYPE.BF16:
    case 30:
      return matmulF16Rows
    case GGML_TYPE.F32:
      return matmulF32Rows
    default:
      return null
  }
}

function ensureXQ8Buf() {
  if (xQ8Buf === null) {
    var xQ8Buffer = new ArrayBuffer(xQ8Size)
    xQ8Buf = new Uint8Array(xQ8Buffer)
    xQ8Int8Buf = new Int8Array(xQ8Buffer)
  }
}

function matmulQuantized(out, x, qw) {
  var kind = qw.k.kind
  if (kind === 1) {
    matmulQ8_0Local(out, x, qw)
  } else if (kind === 3) {
    // Fused kernel when the type has one, else dequantize-then-dot
    if (qw.k.rowsFunc !== null) {
      matmulRows(out, x, qw)
    } else {
      matmulKQuantLocal(out, x, qw, 0)
    }
  } else if (kind === 2) {
    // Block-32 formats: quantize x to Q8_0 once, then integer dot products
    ensureXQ8Buf()
    quantizeToQ8_0Cache(x, 0, xQ8Buf, xQ8Int8Buf, 0, qw.cols)
    matmulQuantizedPreQ8(out, qw)
  } else {
    var rows = qw.rows
    var cols = qw.cols
    var baseOffset = qw.dataOffset
    var rowSize = qw.rowSize
    for (var i = 0; i < rows; i = i + 1) {
      out[i] = vecDotQ8_0(x, baseOffset + i * rowSize, cols)
    }
  }
}

// Quantized matmul with x already quantized into xQ8Buf/xQ8Int8Buf
function matmulQuantizedPreQ8(out, qw) {
  // 4 rows per call, weights read as Uint16 words, x shared by the 4 rows
  var rows = qw.rows
  var rowsFunc = qw.k.rowsFunc
  for (var i = 0; i < rows; i = i + 4) {
    rowsFunc(out, qw, i)
  }
}

// Batched matmul: several input vectors against the same weight matrix
var PREFILL_BATCH_SIZE = 32

// xs64: Float64 staging of xs for the float tiles (or null); stage: fill it
function matmulQuantizedBatch(outs, xs, qw, batchSize, xs64, stage) {
  var rows = qw.rows
  var cols = qw.cols
  var baseOffset = qw.dataOffset
  var rowSize = qw.rowSize

  var kind = qw.k.kind
  if (kind === 1) {
    // Q8_0 through the Int32 row dequantizer
    matmulDeqBatch(outs, xs, xs64, stage, qw, batchSize, deqRowQ8_0I32, qw.view)
  } else if (kind === 3) {
    // K-quants and F16/BF16/F32
    matmulDeqBatch(outs, xs, xs64, stage, qw, batchSize, qw.k.deqRowFunc, qw.view)
  } else if (kind === 2) {
    // Block-32 formats with Q8 activations: unpack once, integer tile
    matmulBlock32Q8Batch(outs, xs, qw, batchSize)
  } else {
    for (var i = 0; i < rows; i = i + 1) {
      var rowOff = baseOffset + i * rowSize
      for (var b = 0; b < batchSize; b = b + 1) {
        outs[b][i] = vecDotQ8_0(xs[b], rowOff, cols)
      }
    }
  }
}

// Q8_0 matmul, single token: 6 rows per step, one Int32 word (4 weights) per
// row, so every x load serves 6 rows; q = word of row i, the other rows at fixed
// strides; blocks walked in pairs (17 words). The last group starts at rows - 6
// (same values twice). Needs the Int32 view (kind 1).
function matmulQ8_0Local(out, x, qw) {
  var I32 = qw.view
  var rows = qw.rows
  var cols = qw.cols
  var rowWords = qw.rowSize >> 2
  var lastI = rows - 6
  var w1 = rowWords
  var w2 = (w1 + rowWords) | 0
  var w3 = (w2 + rowWords) | 0
  var w4 = (w3 + rowWords) | 0
  var w5 = (w4 + rowWords) | 0
  var q = 0
  for (var i = 0; i < rows; i = i + 6) {
    if (i > lastI) {
      i = lastI
    }
    q = (qw.base + i * rowWords) | 0
    var s0 = 0.0
    var s1 = 0.0
    var s2 = 0.0
    var s3 = 0.0
    var s4 = 0.0
    var s5 = 0.0
    for (var xb = 0; xb < cols; xb = xb + 64) {
      // Even block: word 0 holds the scale (low half) and weights 0-1
      var v0 = I32[q]
      var v1 = I32[(q + w1) | 0]
      var v2 = I32[(q + w2) | 0]
      var v3 = I32[(q + w3) | 0]
      var v4 = I32[(q + w4) | 0]
      var v5 = I32[(q + w5) | 0]
      var e0 = v0 & 0xffff
      var e1 = v1 & 0xffff
      var e2 = v2 & 0xffff
      var e3 = v3 & 0xffff
      var e4 = v4 & 0xffff
      var e5 = v5 & 0xffff
      var xa = x[xb]
      var xd = x[(xb + 1) | 0]
      var a0 = xa * ((v0 << 8) >> 24) + xd * (v0 >> 24)
      var a1 = xa * ((v1 << 8) >> 24) + xd * (v1 >> 24)
      var a2 = xa * ((v2 << 8) >> 24) + xd * (v2 >> 24)
      var a3 = xa * ((v3 << 8) >> 24) + xd * (v3 >> 24)
      var a4 = xa * ((v4 << 8) >> 24) + xd * (v4 >> 24)
      var a5 = xa * ((v5 << 8) >> 24) + xd * (v5 >> 24)
      // Words 1-7: weights 2-29
      var j = (xb + 2) | 0
      var qEnd = (q + 8) | 0
      for (q = (q + 1) | 0; q < qEnd; q = (q + 1) | 0) {
        v0 = I32[q]
        v1 = I32[(q + w1) | 0]
        v2 = I32[(q + w2) | 0]
        v3 = I32[(q + w3) | 0]
        v4 = I32[(q + w4) | 0]
        v5 = I32[(q + w5) | 0]
        xa = x[j]
        var xb1 = x[(j + 1) | 0]
        var xc = x[(j + 2) | 0]
        xd = x[(j + 3) | 0]
        j = (j + 4) | 0
        a0 =
          a0 +
          xa * ((v0 << 24) >> 24) +
          xb1 * ((v0 << 16) >> 24) +
          xc * ((v0 << 8) >> 24) +
          xd * (v0 >> 24)
        a1 =
          a1 +
          xa * ((v1 << 24) >> 24) +
          xb1 * ((v1 << 16) >> 24) +
          xc * ((v1 << 8) >> 24) +
          xd * (v1 >> 24)
        a2 =
          a2 +
          xa * ((v2 << 24) >> 24) +
          xb1 * ((v2 << 16) >> 24) +
          xc * ((v2 << 8) >> 24) +
          xd * (v2 >> 24)
        a3 =
          a3 +
          xa * ((v3 << 24) >> 24) +
          xb1 * ((v3 << 16) >> 24) +
          xc * ((v3 << 8) >> 24) +
          xd * (v3 >> 24)
        a4 =
          a4 +
          xa * ((v4 << 24) >> 24) +
          xb1 * ((v4 << 16) >> 24) +
          xc * ((v4 << 8) >> 24) +
          xd * (v4 >> 24)
        a5 =
          a5 +
          xa * ((v5 << 24) >> 24) +
          xb1 * ((v5 << 16) >> 24) +
          xc * ((v5 << 8) >> 24) +
          xd * (v5 >> 24)
      }
      // Word 8 (q): weights 30-31 (low half); the odd scale is in its high half
      v0 = I32[q]
      v1 = I32[(q + w1) | 0]
      v2 = I32[(q + w2) | 0]
      v3 = I32[(q + w3) | 0]
      v4 = I32[(q + w4) | 0]
      v5 = I32[(q + w5) | 0]
      var o0 = v0 >>> 16
      var o1 = v1 >>> 16
      var o2 = v2 >>> 16
      var o3 = v3 >>> 16
      var o4 = v4 >>> 16
      var o5 = v5 >>> 16
      xa = x[(xb + 30) | 0]
      xd = x[(xb + 31) | 0]
      a0 = a0 + xa * ((v0 << 24) >> 24) + xd * ((v0 << 16) >> 24)
      a1 = a1 + xa * ((v1 << 24) >> 24) + xd * ((v1 << 16) >> 24)
      a2 = a2 + xa * ((v2 << 24) >> 24) + xd * ((v2 << 16) >> 24)
      a3 = a3 + xa * ((v3 << 24) >> 24) + xd * ((v3 << 16) >> 24)
      a4 = a4 + xa * ((v4 << 24) >> 24) + xd * ((v4 << 16) >> 24)
      a5 = a5 + xa * ((v5 << 24) >> 24) + xd * ((v5 << 16) >> 24)
      s0 = s0 + fp16Table[e0] * a0
      s1 = s1 + fp16Table[e1] * a1
      s2 = s2 + fp16Table[e2] * a2
      s3 = s3 + fp16Table[e3] * a3
      s4 = s4 + fp16Table[e4] * a4
      s5 = s5 + fp16Table[e5] * a5
      // Odd block: words 9-16, weights 32-63 (0.0 + p is p)
      a0 = 0.0
      a1 = 0.0
      a2 = 0.0
      a3 = 0.0
      a4 = 0.0
      a5 = 0.0
      j = (xb + 32) | 0
      qEnd = (q + 9) | 0
      for (q = (q + 1) | 0; q < qEnd; q = (q + 1) | 0) {
        v0 = I32[q]
        v1 = I32[(q + w1) | 0]
        v2 = I32[(q + w2) | 0]
        v3 = I32[(q + w3) | 0]
        v4 = I32[(q + w4) | 0]
        v5 = I32[(q + w5) | 0]
        xa = x[j]
        var xb1 = x[(j + 1) | 0]
        var xc = x[(j + 2) | 0]
        xd = x[(j + 3) | 0]
        j = (j + 4) | 0
        a0 =
          a0 +
          xa * ((v0 << 24) >> 24) +
          xb1 * ((v0 << 16) >> 24) +
          xc * ((v0 << 8) >> 24) +
          xd * (v0 >> 24)
        a1 =
          a1 +
          xa * ((v1 << 24) >> 24) +
          xb1 * ((v1 << 16) >> 24) +
          xc * ((v1 << 8) >> 24) +
          xd * (v1 >> 24)
        a2 =
          a2 +
          xa * ((v2 << 24) >> 24) +
          xb1 * ((v2 << 16) >> 24) +
          xc * ((v2 << 8) >> 24) +
          xd * (v2 >> 24)
        a3 =
          a3 +
          xa * ((v3 << 24) >> 24) +
          xb1 * ((v3 << 16) >> 24) +
          xc * ((v3 << 8) >> 24) +
          xd * (v3 >> 24)
        a4 =
          a4 +
          xa * ((v4 << 24) >> 24) +
          xb1 * ((v4 << 16) >> 24) +
          xc * ((v4 << 8) >> 24) +
          xd * (v4 >> 24)
        a5 =
          a5 +
          xa * ((v5 << 24) >> 24) +
          xb1 * ((v5 << 16) >> 24) +
          xc * ((v5 << 8) >> 24) +
          xd * (v5 >> 24)
      }
      s0 = s0 + fp16Table[o0] * a0
      s1 = s1 + fp16Table[o1] * a1
      s2 = s2 + fp16Table[o2] * a2
      s3 = s3 + fp16Table[o3] * a3
      s4 = s4 + fp16Table[o4] * a4
      s5 = s5 + fp16Table[o5] * a5
    }
    out[i] = s0
    out[i + 1] = s1
    out[i + 2] = s2
    out[i + 3] = s3
    out[i + 4] = s4
    out[i + 5] = s5
  }
}

// Dequantize one Q8_0 row (even number of blocks) through the Int32 view
function deqRowQ8_0I32(I32, p, dst, dstOff, cols) {
  var nbp = cols >> 6
  var o = dstOff
  for (var b = 0; b < nbp; b = b + 1) {
    var v0 = I32[p]
    var v8 = I32[p + 8]
    var d0 = fp16Table[v0 & 0xffff]
    var d1 = fp16Table[v8 >>> 16]
    dst[o] = d0 * ((v0 << 8) >> 24)
    dst[o + 1] = d0 * (v0 >> 24)
    for (var k = 1; k < 8; k = k + 1) {
      var v = I32[p + k]
      var j = o + 4 * k - 2
      dst[j] = d0 * ((v << 24) >> 24)
      dst[j + 1] = d0 * ((v << 16) >> 24)
      dst[j + 2] = d0 * ((v << 8) >> 24)
      dst[j + 3] = d0 * (v >> 24)
    }
    dst[o + 30] = d0 * ((v8 << 24) >> 24)
    dst[o + 31] = d0 * ((v8 << 16) >> 24)
    for (var k = 9; k < 17; k = k + 1) {
      var v = I32[p + k]
      var j = o + 4 * k - 4
      dst[j] = d1 * ((v << 24) >> 24)
      dst[j + 1] = d1 * ((v << 16) >> 24)
      dst[j + 2] = d1 * ((v << 8) >> 24)
      dst[j + 3] = d1 * (v >> 24)
    }
    p = p + 17
    o = o + 64
  }
}

// ----------------------------------------------------------------------------
// Fused single-token kernels: rows i.. of a matrix against x from the matrix
// view; weights as in the row dequantizers, rows summed left to right (bit-
// identical); >> not >>> keeps the words int32 (JavaScriptCore)

// F16 / BF16: two weights per Int32 load through the 64K table, 4 rows per call
function matmulF16Rows(out, x, qw, i) {
  var I32 = qw.view
  var half = qw.cols >> 1
  var tab = qw.type === GGML_TYPE.F16 ? fp16Table : bf16Table
  var o0 = (qw.base + i * half) | 0
  var o1 = (o0 + half) | 0
  var o2 = (o1 + half) | 0
  var o3 = (o2 + half) | 0
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  for (var k = 0; k < half; k = k + 1) {
    var a = x[k << 1]
    var c = x[((k << 1) + 1) | 0]
    var v = I32[(o0 + k) | 0]
    s0 = s0 + a * tab[v & 0xffff]
    s0 = s0 + c * tab[v >>> 16]
    v = I32[(o1 + k) | 0]
    s1 = s1 + a * tab[v & 0xffff]
    s1 = s1 + c * tab[v >>> 16]
    v = I32[(o2 + k) | 0]
    s2 = s2 + a * tab[v & 0xffff]
    s2 = s2 + c * tab[v >>> 16]
    v = I32[(o3 + k) | 0]
    s3 = s3 + a * tab[v & 0xffff]
    s3 = s3 + c * tab[v >>> 16]
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

// F32: through the Float32 view.
function matmulF32Rows(out, x, qw, i) {
  var F32 = qw.view
  var cols = qw.cols
  var o0 = (qw.base + i * cols) | 0
  var o1 = (o0 + cols) | 0
  var s0 = 0.0
  var s1 = 0.0
  for (var j = 0; j < cols; j = j + 1) {
    var a = x[j]
    s0 = s0 + a * F32[(o0 + j) | 0]
    s1 = s1 + a * F32[(o1 + j) | 0]
  }
  out[i] = s0
  out[i + 1] = s1
}

// Q4_K: 8 groups of 32 columns per block; group c: nibbles (low: c even) of
// pair c >> 1 with scale/min c. Rows i .. i + 3 per call (row r at rw * r words)
function matmulQ4_KRows(out, x, qw, i) {
  var I32 = qw.view
  var nb = qw.cols >> 8
  var rw = qw.rowSize >> 2
  var rw2 = (rw * 2) | 0
  var rw3 = (rw * 3) | 0
  var p = (qw.base + i * rw) | 0
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xb = 0
  for (var b = 0; b < nb; b = b + 1) {
    // d, dmin and the three scale words of each row's block
    var h
    h = I32[p]
    var d0 = fp16Table[h & 0xffff]
    var n0 = fp16Table[h >>> 16]
    var u0 = I32[(p + 1) | 0]
    var v0 = I32[(p + 2) | 0]
    var w0 = I32[(p + 3) | 0]
    h = I32[(p + rw) | 0]
    var d1 = fp16Table[h & 0xffff]
    var n1 = fp16Table[h >>> 16]
    var u1 = I32[(p + rw + 1) | 0]
    var v1 = I32[(p + rw + 2) | 0]
    var w1 = I32[(p + rw + 3) | 0]
    h = I32[(p + rw2) | 0]
    var d2 = fp16Table[h & 0xffff]
    var n2 = fp16Table[h >>> 16]
    var u2 = I32[(p + rw2 + 1) | 0]
    var v2 = I32[(p + rw2 + 2) | 0]
    var w2 = I32[(p + rw2 + 3) | 0]
    h = I32[(p + rw3) | 0]
    var d3 = fp16Table[h & 0xffff]
    var n3 = fp16Table[h >>> 16]
    var u3 = I32[(p + rw3 + 1) | 0]
    var v3 = I32[(p + rw3 + 2) | 0]
    var w3 = I32[(p + rw3 + 3) | 0]
    for (var c = 0; c < 8; c = c + 1) {
      var sm
      sm = kScaleMin(c, u0, v0, w0)
      var k0 = d0 * (sm & 0xff)
      var m0 = n0 * (sm >>> 8)
      sm = kScaleMin(c, u1, v1, w1)
      var k1 = d1 * (sm & 0xff)
      var m1 = n1 * (sm >>> 8)
      sm = kScaleMin(c, u2, v2, w2)
      var k2 = d2 * (sm & 0xff)
      var m2 = n2 * (sm >>> 8)
      sm = kScaleMin(c, u3, v3, w3)
      var k3 = d3 * (sm & 0xff)
      var m3 = n3 * (sm >>> 8)
      var sh = (c & 1) << 2
      var pq = (p + 4 + ((c >> 1) << 3)) | 0
      for (var k = 0; k < 8; k = k + 1) {
        var j = (xb + (k << 2)) | 0
        var x0 = x[j]
        var x1 = x[(j + 1) | 0]
        var x2 = x[(j + 2) | 0]
        var x3 = x[(j + 3) | 0]
        var pk = (pq + k) | 0
        var q
        q = (I32[pk] >> sh) & 0x0f0f0f0f
        s0 = s0 + x0 * (k0 * (q & 0xff) - m0)
        s0 = s0 + x1 * (k0 * ((q >>> 8) & 0xff) - m0)
        s0 = s0 + x2 * (k0 * ((q >>> 16) & 0xff) - m0)
        s0 = s0 + x3 * (k0 * (q >>> 24) - m0)
        q = (I32[(pk + rw) | 0] >> sh) & 0x0f0f0f0f
        s1 = s1 + x0 * (k1 * (q & 0xff) - m1)
        s1 = s1 + x1 * (k1 * ((q >>> 8) & 0xff) - m1)
        s1 = s1 + x2 * (k1 * ((q >>> 16) & 0xff) - m1)
        s1 = s1 + x3 * (k1 * (q >>> 24) - m1)
        q = (I32[(pk + rw2) | 0] >> sh) & 0x0f0f0f0f
        s2 = s2 + x0 * (k2 * (q & 0xff) - m2)
        s2 = s2 + x1 * (k2 * ((q >>> 8) & 0xff) - m2)
        s2 = s2 + x2 * (k2 * ((q >>> 16) & 0xff) - m2)
        s2 = s2 + x3 * (k2 * (q >>> 24) - m2)
        q = (I32[(pk + rw3) | 0] >> sh) & 0x0f0f0f0f
        s3 = s3 + x0 * (k3 * (q & 0xff) - m3)
        s3 = s3 + x1 * (k3 * ((q >>> 8) & 0xff) - m3)
        s3 = s3 + x2 * (k3 * ((q >>> 16) & 0xff) - m3)
        s3 = s3 + x3 * (k3 * (q >>> 24) - m3)
      }
      xb = xb + 32
    }
    p = (p + 36) | 0
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

// Q5_K: Q4_K plus the fifth bit (bit c of the 32 high-bit bytes) of every quant
function matmulQ5_KRows(out, x, qw, i) {
  var I32 = qw.view
  var nb = qw.cols >> 8
  var rw = qw.rowSize >> 2
  var rw2 = (rw * 2) | 0
  var rw3 = (rw * 3) | 0
  var p = (qw.base + i * rw) | 0
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xb = 0
  for (var b = 0; b < nb; b = b + 1) {
    // d, dmin and the three scale words of each row's block
    var h
    h = I32[p]
    var d0 = fp16Table[h & 0xffff]
    var n0 = fp16Table[h >>> 16]
    var u0 = I32[(p + 1) | 0]
    var v0 = I32[(p + 2) | 0]
    var w0 = I32[(p + 3) | 0]
    h = I32[(p + rw) | 0]
    var d1 = fp16Table[h & 0xffff]
    var n1 = fp16Table[h >>> 16]
    var u1 = I32[(p + rw + 1) | 0]
    var v1 = I32[(p + rw + 2) | 0]
    var w1 = I32[(p + rw + 3) | 0]
    h = I32[(p + rw2) | 0]
    var d2 = fp16Table[h & 0xffff]
    var n2 = fp16Table[h >>> 16]
    var u2 = I32[(p + rw2 + 1) | 0]
    var v2 = I32[(p + rw2 + 2) | 0]
    var w2 = I32[(p + rw2 + 3) | 0]
    h = I32[(p + rw3) | 0]
    var d3 = fp16Table[h & 0xffff]
    var n3 = fp16Table[h >>> 16]
    var u3 = I32[(p + rw3 + 1) | 0]
    var v3 = I32[(p + rw3 + 2) | 0]
    var w3 = I32[(p + rw3 + 3) | 0]
    for (var c = 0; c < 8; c = c + 1) {
      var sm
      sm = kScaleMin(c, u0, v0, w0)
      var k0 = d0 * (sm & 0xff)
      var m0 = n0 * (sm >>> 8)
      sm = kScaleMin(c, u1, v1, w1)
      var k1 = d1 * (sm & 0xff)
      var m1 = n1 * (sm >>> 8)
      sm = kScaleMin(c, u2, v2, w2)
      var k2 = d2 * (sm & 0xff)
      var m2 = n2 * (sm >>> 8)
      sm = kScaleMin(c, u3, v3, w3)
      var k3 = d3 * (sm & 0xff)
      var m3 = n3 * (sm >>> 8)
      var sh = (c & 1) << 2
      var pq = (p + 12 + ((c >> 1) << 3)) | 0
      for (var k = 0; k < 8; k = k + 1) {
        var j = (xb + (k << 2)) | 0
        var x0 = x[j]
        var x1 = x[(j + 1) | 0]
        var x2 = x[(j + 2) | 0]
        var x3 = x[(j + 3) | 0]
        var pk = (pq + k) | 0
        var hk = (p + 4 + k) | 0
        var q
        q = ((I32[pk] >> sh) & 0x0f0f0f0f) | (((I32[hk] >> c) & 0x01010101) << 4)
        s0 = s0 + x0 * (k0 * (q & 0xff) - m0)
        s0 = s0 + x1 * (k0 * ((q >>> 8) & 0xff) - m0)
        s0 = s0 + x2 * (k0 * ((q >>> 16) & 0xff) - m0)
        s0 = s0 + x3 * (k0 * (q >>> 24) - m0)
        q =
          ((I32[(pk + rw) | 0] >> sh) & 0x0f0f0f0f) |
          (((I32[(hk + rw) | 0] >> c) & 0x01010101) << 4)
        s1 = s1 + x0 * (k1 * (q & 0xff) - m1)
        s1 = s1 + x1 * (k1 * ((q >>> 8) & 0xff) - m1)
        s1 = s1 + x2 * (k1 * ((q >>> 16) & 0xff) - m1)
        s1 = s1 + x3 * (k1 * (q >>> 24) - m1)
        q =
          ((I32[(pk + rw2) | 0] >> sh) & 0x0f0f0f0f) |
          (((I32[(hk + rw2) | 0] >> c) & 0x01010101) << 4)
        s2 = s2 + x0 * (k2 * (q & 0xff) - m2)
        s2 = s2 + x1 * (k2 * ((q >>> 8) & 0xff) - m2)
        s2 = s2 + x2 * (k2 * ((q >>> 16) & 0xff) - m2)
        s2 = s2 + x3 * (k2 * (q >>> 24) - m2)
        q =
          ((I32[(pk + rw3) | 0] >> sh) & 0x0f0f0f0f) |
          (((I32[(hk + rw3) | 0] >> c) & 0x01010101) << 4)
        s3 = s3 + x0 * (k3 * (q & 0xff) - m3)
        s3 = s3 + x1 * (k3 * ((q >>> 8) & 0xff) - m3)
        s3 = s3 + x2 * (k3 * ((q >>> 16) & 0xff) - m3)
        s3 = s3 + x3 * (k3 * (q >>> 24) - m3)
      }
      xb = xb + 32
    }
    p = (p + 44) | 0
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

// Q3_K / Q6_K: 16 groups of 16 columns per block (group t: half t >> 3, position
// (t >> 1) & 3, scale t), 4 quants per DataView load (blocks are 2-byte
// aligned), low bits merged with the high bits; rows i .. i + 3 per call
function matmulK16Rows(out, x, qw, i) {
  var U16 = qw.view
  var DV = rowsDataView
  var sc = q3kScales
  var q6 = qw.type === GGML_TYPE.Q6_K
  var blockWords = q6 ? 105 : 55
  var loMask = q6 ? 0x0f0f0f0f : 0x03030303
  var hiMask = q6 ? 0x03030303 : 0x01010101
  var hiUp = q6 ? 4 : 2
  var bias = q6 ? 32 : 4
  var scBias = q6 ? 0 : 32
  var nb = qw.cols >> 8
  var rowWords = qw.rowSize >> 1
  // r0: 16-bit word index of row i's block inside the matrix (what the
  // DataView covers); vb: matrix start in the Uint16 view; rb: row stride in bytes
  var vb = qw.base
  var r0 = (i * rowWords) | 0
  var rb = rowWords << 1
  var rb2 = (rb * 2) | 0
  var rb3 = (rb * 3) | 0
  var s0 = 0.0
  var s1 = 0.0
  var s2 = 0.0
  var s3 = 0.0
  var xb = 0
  for (var blk = 0; blk < nb; blk = blk + 1) {
    var d0 = fp16Table[U16[(vb + r0 + blockWords - 1) | 0]]
    var d1 = fp16Table[U16[(vb + r0 + 1 * rowWords + blockWords - 1) | 0]]
    var d2 = fp16Table[U16[(vb + r0 + 2 * rowWords + blockWords - 1) | 0]]
    var d3 = fp16Table[U16[(vb + r0 + 3 * rowWords + blockWords - 1) | 0]]
    // Scales of the 4 rows into sc[16 * r ..] (one loop, so the compiled code
    // holds one copy of the unpacking)
    var sp = (vb + r0) | 0
    for (var r = 0; r < 64; r = r + 16) {
      if (q6) {
        // 16 int8 scales in the 8 words before d
        for (var t = 0; t < 8; t = t + 1) {
          var w = U16[(sp + 96 + t) | 0]
          sc[(r + (t << 1)) | 0] = w
          sc[(r + (t << 1) + 1) | 0] = w >> 8
        }
      } else {
        q3kUnpackScales(U16, sp, r)
      }
      sp = (sp + rowWords) | 0
    }
    for (var t = 0; t < 16; t = t + 1) {
      var e0 = d0 * (sc[(0 + t) | 0] - scBias)
      var e1 = d1 * (sc[(16 + t) | 0] - scBias)
      var e2 = d2 * (sc[(32 + t) | 0] - scBias)
      var e3 = d3 * (sc[(48 + t) | 0] - scBias)
      var h = t >> 3
      var g = (t >> 1) & 3
      var ss = (t & 1) << 3
      // Word offsets inside the block and bit shifts of the low / high bits
      var lo = (16 + (h << 4) + ss) | 0
      var ho = ss
      var loShift = g << 1
      var hiShift = (h << 2) + g
      if (q6) {
        lo = ((h << 5) + ((g & 1) << 4) + ss) | 0
        ho = (64 + (h << 4) + ss) | 0
        loShift = (g >> 1) << 2
        hiShift = g << 1
      }
      // Byte offsets of row i's low and high words in the DataView
      var lb = ((r0 + lo) << 1) | 0
      var hb = ((r0 + ho) << 1) | 0
      for (var k = 0; k < 16; k = k + 4) {
        var j = (xb + k) | 0
        var x0 = x[j]
        var x1 = x[(j + 1) | 0]
        var x2 = x[(j + 2) | 0]
        var x3 = x[(j + 3) | 0]
        var lk = (lb + k) | 0
        var hk = (hb + k) | 0
        var q
        q =
          ((DV.getInt32(lk | 0, true) >> loShift) & loMask) |
          (((DV.getInt32(hk | 0, true) >> hiShift) & hiMask) << hiUp)
        s0 = s0 + x0 * (e0 * ((q & 0xff) - bias))
        s0 = s0 + x1 * (e0 * (((q >>> 8) & 0xff) - bias))
        s0 = s0 + x2 * (e0 * (((q >>> 16) & 0xff) - bias))
        s0 = s0 + x3 * (e0 * ((q >>> 24) - bias))
        q =
          ((DV.getInt32((lk + rb) | 0, true) >> loShift) & loMask) |
          (((DV.getInt32((hk + rb) | 0, true) >> hiShift) & hiMask) << hiUp)
        s1 = s1 + x0 * (e1 * ((q & 0xff) - bias))
        s1 = s1 + x1 * (e1 * (((q >>> 8) & 0xff) - bias))
        s1 = s1 + x2 * (e1 * (((q >>> 16) & 0xff) - bias))
        s1 = s1 + x3 * (e1 * ((q >>> 24) - bias))
        q =
          ((DV.getInt32((lk + rb2) | 0, true) >> loShift) & loMask) |
          (((DV.getInt32((hk + rb2) | 0, true) >> hiShift) & hiMask) << hiUp)
        s2 = s2 + x0 * (e2 * ((q & 0xff) - bias))
        s2 = s2 + x1 * (e2 * (((q >>> 8) & 0xff) - bias))
        s2 = s2 + x2 * (e2 * (((q >>> 16) & 0xff) - bias))
        s2 = s2 + x3 * (e2 * ((q >>> 24) - bias))
        q =
          ((DV.getInt32((lk + rb3) | 0, true) >> loShift) & loMask) |
          (((DV.getInt32((hk + rb3) | 0, true) >> hiShift) & hiMask) << hiUp)
        s3 = s3 + x0 * (e3 * ((q & 0xff) - bias))
        s3 = s3 + x1 * (e3 * (((q >>> 8) & 0xff) - bias))
        s3 = s3 + x2 * (e3 * (((q >>> 16) & 0xff) - bias))
        s3 = s3 + x3 * (e3 * ((q >>> 24) - bias))
      }
      xb = xb + 16
    }
    r0 = (r0 + blockWords) | 0
  }
  out[i] = s0
  out[i + 1] = s1
  out[i + 2] = s2
  out[i + 3] = s3
}

// DataView of the matrix being multiplied (Q3_K / Q6_K, see matmulRows)
var rowsDataView = null

// Single-token matmul through a fused kernel; leftover rows: generic kernel
function matmulRows(out, x, qw) {
  var rowsFunc = qw.k.rowsFunc
  var step = rowsFunc === matmulF32Rows ? 2 : 4
  var rowsN = qw.rows - (qw.rows % step)
  if (qw.view.BYTES_PER_ELEMENT === 2) {
    // Q3_K / Q6_K: DataView for 4-byte loads, only during the call
    rowsDataView = new DataView(ggufData, qw.dataOffset, qw.rows * qw.rowSize)
  }
  for (var i = 0; i < rowsN; i = i + step) {
    rowsFunc(out, x, qw, i)
  }
  rowsDataView = null
  if (rowsN < qw.rows) {
    matmulKQuantLocal(out, x, qw, rowsN)
  }
}

// K-quant matmul from startRow: 4 rows dequantized into a scratch (matmulDeqBuf
// in prefill, else an idle buffer: logits during the layers, hb for the logits)
function matmulKQuantLocal(out, x, qw, startRow) {
  var rows = qw.rows
  var cols = qw.cols
  var rowSize = qw.rowSize
  var deqFunc = qw.k.deqRowFunc
  var buf = matmulDeqBuf
  var rows4 = startRow + ((rows - startRow) & ~3)
  if (buf === null) {
    buf = out === state.logits ? state.hb64 : state.logits64
    if (buf === null || buf.length < cols) {
      if (state.kqRowScratch === null || state.kqRowScratch.length < cols) {
        state.kqRowScratch = new Float64Array(cols)
      }
      buf = state.kqRowScratch
    }
    if (buf.length < 4 * cols) {
      rows4 = startRow
    }
  }
  var off1 = cols
  var off2 = cols + cols
  var off3 = off2 + cols
  var view = qw.view
  // Row length and position in elements of the view
  var rowEl = rowSize / view.BYTES_PER_ELEMENT
  var base = qw.base
  for (var i = startRow; i < rows4; i = i + 4) {
    var p = base + i * rowEl
    deqFunc(view, p, buf, 0, cols)
    deqFunc(view, p + rowEl, buf, off1, cols)
    deqFunc(view, p + rowEl + rowEl, buf, off2, cols)
    deqFunc(view, p + rowEl + rowEl + rowEl, buf, off3, cols)
    var s0 = 0.0
    var s1 = 0.0
    var s2 = 0.0
    var s3 = 0.0
    for (var j = 0; j < cols; j = j + 1) {
      var a = x[j]
      s0 = s0 + a * buf[j]
      s1 = s1 + a * buf[(off1 + j) | 0]
      s2 = s2 + a * buf[(off2 + j) | 0]
      s3 = s3 + a * buf[(off3 + j) | 0]
    }
    out[i] = s0
    out[i + 1] = s1
    out[i + 2] = s2
    out[i + 3] = s3
  }
  for (var i = rows4; i < rows; i = i + 1) {
    deqFunc(view, base + i * rowEl, buf, 0, cols)
    var s = 0.0
    for (var j = 0; j < cols; j = j + 1) {
      s = s + x[j] * buf[j]
    }
    out[i] = s
  }
}

// Prefill matmul of the formats with a row dequantizer (Q8_0 Int32, K-quants,
// F16 / BF16 / F32): 4 rows into matmulDeqBuf, then a 4-row x 3-token tile (12
// accumulators; 1-2 tokens left: 2 lanes), same column order as the single-
// token kernels. With stage, xs64 is filled first, last token first and each
// backwards: the w2 staging lies over hb itself (slot b covers tokens 2b, 2b+1)
function deqTile(outs, xs, bt, nTok, i, cols) {
  var buf0 = matmulDeqRows[0]
  var buf1 = matmulDeqRows[1]
  var buf2 = matmulDeqRows[2]
  var buf3 = matmulDeqRows[3]
  var xA = xs[bt]
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
  if (nTok === 3) {
    var xB = xs[bt + 1]
    var xC = xs[bt + 2]
    for (var j = 0; j < cols; j = j + 1) {
      var a = xA[j]
      var e = xB[j]
      var q = xC[j]
      var w0 = buf0[j]
      var w1 = buf1[j]
      var w2 = buf2[j]
      var w3 = buf3[j]
      s0 = s0 + a * w0
      t0 = t0 + e * w0
      u0 = u0 + q * w0
      s1 = s1 + a * w1
      t1 = t1 + e * w1
      u1 = u1 + q * w1
      s2 = s2 + a * w2
      t2 = t2 + e * w2
      u2 = u2 + q * w2
      s3 = s3 + a * w3
      t3 = t3 + e * w3
      u3 = u3 + q * w3
    }
    var oC = outs[bt + 2]
    oC[i] = u0
    oC[i + 1] = u1
    oC[i + 2] = u2
    oC[i + 3] = u3
  } else {
    var xB = xs[nTok > 1 ? bt + 1 : bt]
    for (var j = 0; j < cols; j = j + 1) {
      var a = xA[j]
      var e = xB[j]
      var w0 = buf0[j]
      var w1 = buf1[j]
      var w2 = buf2[j]
      var w3 = buf3[j]
      s0 = s0 + a * w0
      t0 = t0 + e * w0
      s1 = s1 + a * w1
      t1 = t1 + e * w1
      s2 = s2 + a * w2
      t2 = t2 + e * w2
      s3 = s3 + a * w3
      t3 = t3 + e * w3
    }
  }
  var oA = outs[bt]
  oA[i] = s0
  oA[i + 1] = s1
  oA[i + 2] = s2
  oA[i + 3] = s3
  if (nTok > 1) {
    var oB = outs[bt + 1]
    oB[i] = t0
    oB[i + 1] = t1
    oB[i + 2] = t2
    oB[i + 3] = t3
  }
}

function matmulDeqBatch(outs, xs, xs64, stage, qw, batchSize, deqFunc, view) {
  var rows = qw.rows
  var cols = qw.cols
  if (xs64 !== null) {
    if (stage) {
      for (var b = batchSize - 1; b >= 0; b = b - 1) {
        var d = xs64[b]
        var x = xs[b]
        for (var j = cols - 1; j >= 0; j = j - 1) {
          d[j] = x[j]
        }
      }
    }
    xs = xs64
  }
  var rows4 = rows & ~3
  var buf0 = matmulDeqRows[0]
  var buf1 = matmulDeqRows[1]
  var buf2 = matmulDeqRows[2]
  var buf3 = matmulDeqRows[3]
  // Row length and position in elements of the view
  var rowEl = qw.rowSize / view.BYTES_PER_ELEMENT
  var base = qw.base
  for (var i = 0; i < rows4; i = i + 4) {
    var p = base + i * rowEl
    deqFunc(view, p, buf0, 0, cols)
    deqFunc(view, p + rowEl, buf1, 0, cols)
    deqFunc(view, p + rowEl + rowEl, buf2, 0, cols)
    deqFunc(view, p + rowEl + rowEl + rowEl, buf3, 0, cols)
    for (var bt = 0; bt < batchSize; bt = bt + 3) {
      var nTok = batchSize - bt
      if (nTok > 3) {
        nTok = 3
      }
      deqTile(outs, xs, bt, nTok, i, cols)
    }
  }
  // Remaining 1-3 rows
  for (var i = rows4; i < rows; i = i + 1) {
    deqFunc(view, base + i * rowEl, buf0, 0, cols)
    for (var bt = 0; bt < batchSize; bt = bt + 1) {
      var xArr = xs[bt]
      var s = 0.0
      for (var j = 0; j < cols; j = j + 1) {
        s = s + xArr[j] * buf0[j]
      }
      outs[bt][i] = s
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

// RMS norm of x[off..off+size) into out (w from 0); Gemma weights already
// carry the +1. Also the per-head QK norms
function rmsnorm(out, x, off, w, size, eps, invSize) {
  invSize = invSize || 1.0 / size
  var ss = 0.0
  var end = off + size
  for (var i = off; i < end; i = i + 1) {
    var v = x[i]
    ss = ss + v * v
  }
  ss = 1.0 / Math.sqrt(ss * invSize + eps)
  var wi = 0
  for (var i = off; i < end; i = i + 1, wi = wi + 1) {
    out[i] = w[wi] * ss * x[i]
  }
} // Fused embedding scale + RMS norm for Gemma first layer (scales x in place)
function rmsnormGemmaFusedScale(out, x, w, size, eps, invSize, scale) {
  var ss = 0.0
  for (var i = 0; i < size; i = i + 1) {
    var xv = x[i] * scale
    x[i] = xv
    ss = ss + xv * xv
  }
  ss = 1.0 / Math.sqrt(ss * invSize + eps)
  for (var i = 0; i < size; i = i + 1) {
    out[i] = w[i] * ss * x[i]
  }
}
function accum(a, b, size) {
  for (var i = 0; i < size; i = i + 1) {
    a[i] = a[i] + b[i]
  }
}
// ----------------------------------------------------------------------------
// GGUF parsing

// Metadata keys unused by the engine (skipped without decoding)
var SKIP_METADATA_KEYS = {
  "tokenizer.ggml.merges": true,
  "tokenizer.ggml.pre": true,
  "tokenizer.ggml.scores": true,
  "tokenizer.ggml.token_type": true,
  "tokenizer.ggml.add_bos_token": true,
  "tokenizer.ggml.add_eos_token": true,
  "tokenizer.ggml.add_space_prefix": true,
  "tokenizer.ggml.padding_token_id": true,
  "tokenizer.ggml.unknown_token_id": true,
  "tokenizer.chat_template": true,
  "general.name": true,
  "general.description": true,
  "general.author": true,
  "general.license": true,
  "general.url": true,
  "general.version": true,
  "general.type": true,
  "general.tags": true,
  "general.languages": true,
  "general.datasets": true,
  "general.finetune": true,
  "general.source.url": true,
  "general.source.huggingface.repository": true,
  "general.quantization_version": true,
  "general.file_type": true,
  "general.base_model.count": true,
  "general.organization": true,
  "general.basename": true,
  "general.size_label": true,
}

// Skip n GGUF strings, return the longest; 4 per iteration keeps
// JavaScriptCore's top tier out of the load
function skipStrings(n) {
  var dv = dataView
  var p = offset
  var slen = 0
  var maxLen = 0
  var n4 = n & ~3
  for (var i = 0; i < n4; i = i + 4) {
    slen = dv.getUint32(p, true)
    maxLen = slen > maxLen ? slen : maxLen
    p = p + 8 + slen
    slen = dv.getUint32(p, true)
    maxLen = slen > maxLen ? slen : maxLen
    p = p + 8 + slen
    slen = dv.getUint32(p, true)
    maxLen = slen > maxLen ? slen : maxLen
    p = p + 8 + slen
    slen = dv.getUint32(p, true)
    maxLen = slen > maxLen ? slen : maxLen
    p = p + 8 + slen
  }
  for (var i = n4; i < n; i = i + 1) {
    slen = dv.getUint32(p, true)
    maxLen = slen > maxLen ? slen : maxLen
    p = p + 8 + slen
  }
  offset = p
  return maxLen
}

// Advance offset past a value without building any JS objects
function skipGGUFValue(type) {
  var arrType
  var arrLen
  var slen
  switch (type) {
    case GGUF_TYPE.UINT8:
    case GGUF_TYPE.INT8:
    case GGUF_TYPE.BOOL:
      offset = offset + 1
      break
    case GGUF_TYPE.UINT16:
    case GGUF_TYPE.INT16:
      offset = offset + 2
      break
    case GGUF_TYPE.UINT32:
    case GGUF_TYPE.INT32:
    case GGUF_TYPE.FLOAT32:
      offset = offset + 4
      break
    case GGUF_TYPE.UINT64:
    case GGUF_TYPE.INT64:
    case GGUF_TYPE.FLOAT64:
      offset = offset + 8
      break
    case GGUF_TYPE.STRING:
      // Read low 4 bytes of uint64 length (high 4 bytes are always 0 for strings)
      slen =
        ggufUint8[offset] |
        (ggufUint8[offset + 1] << 8) |
        (ggufUint8[offset + 2] << 16) |
        (ggufUint8[offset + 3] << 24)
      offset = offset + 8 + slen
      break
    case GGUF_TYPE.ARRAY:
      arrType = readUint32()
      arrLen = readUint64()
      if (
        arrType === GGUF_TYPE.UINT8 ||
        arrType === GGUF_TYPE.INT8 ||
        arrType === GGUF_TYPE.BOOL
      ) {
        offset = offset + arrLen
      } else if (arrType === GGUF_TYPE.UINT16 || arrType === GGUF_TYPE.INT16) {
        offset = offset + arrLen * 2
      } else if (
        arrType === GGUF_TYPE.UINT32 ||
        arrType === GGUF_TYPE.INT32 ||
        arrType === GGUF_TYPE.FLOAT32
      ) {
        offset = offset + arrLen * 4
      } else if (
        arrType === GGUF_TYPE.UINT64 ||
        arrType === GGUF_TYPE.INT64 ||
        arrType === GGUF_TYPE.FLOAT64
      ) {
        offset = offset + arrLen * 8
      } else if (arrType === GGUF_TYPE.STRING) {
        skipStrings(arrLen)
      }
      break
  }
}

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
    if (SKIP_METADATA_KEYS[key]) {
      skipGGUFValue(valueType)
    } else {
      metadata[key] = readGGUFValue(valueType)
    }
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
      nElements = nElements * dims[d]
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

// The token strings stay in the file: note where the array starts and its
// longest string (per-token tables: ensureVocabTables)
function readStringRefs(arrLen) {
  var start = offset
  var maxLen = skipStrings(arrLen)
  return {
    __stringRefs: true,
    vocabOffset: start,
    vocabCount: arrLen,
    vocabMaxLen: maxLen,
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
      if (arrType === GGUF_TYPE.STRING) {
        return readStringRefs(arrLen)
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
  vocabHashHead = null
  vocabHashNext = null
  genScratch = null

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

  var vocabRefs = meta["tokenizer.ggml.tokens"]
  var vocabSize = 0
  var vocabOffset = 0
  var vocabMaxLen = 0
  if (vocabRefs && vocabRefs.__stringRefs) {
    vocabSize = vocabRefs.vocabCount
    vocabOffset = vocabRefs.vocabOffset
    vocabMaxLen = vocabRefs.vocabMaxLen
  }

  if (vocabSize > 0) {
    config.vocabSize = vocabSize
  }

  tokenizer = {
    // Token strings start / longest one (per-token tables: ensureVocabTables)
    vocabOffset: vocabOffset,
    vocabMaxLen: vocabMaxLen,
    vocabLengths: null,
    vocabSparseCum: null,
    vocabSize: vocabSize,
    bosToken: meta["tokenizer.ggml.bos_token_id"] || 1,
    eosToken: meta["tokenizer.ggml.eos_token_id"] || 2,
    eotToken: -1,
    // Ids of the special tokens looked up so far (findSpecialToken)
    specialTokens: {},
  }

  postMessage({ type: "progress", message: "Loading weights..." })

  weights = loadWeights(gguf)
  state = createRunState(config, weights.hasBlock32)

  postMessage({ type: "progress", message: "Model loaded!" })

  return {
    config: config,
    tokenizer: tokenizer,
  }
}

// Kernels record of the Q8_0 matrices read through the Int32 view
var Q8_LOCAL_KERNELS = {
  kind: 1,
  deqRowFunc: null,
  unpackRowFunc: null,
  rowsFunc: null,
}

function loadWeights(gguf) {
  var tensors = gguf.tensors
  var baseOffset = gguf.tensorDataOffset
  var w = {}
  w.hasKQuant = false
  // Some matrix is block-32 (the prefill keeps a Q8_0 copy of every input)
  w.hasBlock32 = false
  // Some layer matrix goes through the Q8_0 Int32 kernels (see generate)
  w.q8Layers = false

  // Load tensor as dequantized float (for small tensors like norms and embeddings)
  function loadTensorFloat(name) {
    var t = tensors[name]
    if (!t) {
      return null
    }
    var off = baseOffset + t.offset
    // Aligned F32 tensor (every norm): zero-copy view
    if (t.type === GGML_TYPE.F32 && (off & 3) === 0) {
      return new Float32Array(ggufData, off, t.nElements)
    }
    return dequantizeTensor(off, t.nElements, t.type)
  }

  // Matrix views (0 Int32Array, 1 Uint16Array, 2 Float32Array): one per kind
  // over the whole file, `base` = matrix start; >2 GB files: one per matrix
  var sharedViews = [null, null, null]
  function matrixView(rec, off, totalBytes, kind) {
    var Ctor = kind === 0 ? Int32Array : kind === 1 ? Uint16Array : Float32Array
    var elemBytes = Ctor.BYTES_PER_ELEMENT
    if (ggufData.byteLength / elemBytes >= 2147483648) {
      return new Ctor(ggufData, off, totalBytes / elemBytes)
    }
    rec.base = (off / elemBytes) | 0
    if (sharedViews[kind] === null) {
      sharedViews[kind] = new Ctor(
        ggufData,
        0,
        Math.floor(ggufData.byteLength / elemBytes)
      )
    }
    return sharedViews[kind]
  }

  // Kernels per tensor type; kind: 1 Q8_0 (Int32 view), 2 block-32, 3 row
  // dequantizer, 0 byte-wise fallback
  var typeKernels = []

  // Load tensor keeping it quantized (for large weight matrices)
  // Returns a QuantizedTensor object with pre-computed rowSize and kernels
  function loadTensorQuantized(name, rows, cols) {
    var t = tensors[name]
    if (!t) {
      return null
    }
    var type = t.type
    var kernels = typeKernels[type]
    if (!kernels) {
      kernels = {
        kind: 0,
        deqRowFunc: getDeqRowFunc(type),
        unpackRowFunc: getUnpackRowFunc(type),
        rowsFunc: getRowsFunc(type),
      }
      if (kernels.deqRowFunc !== null) {
        kernels.kind = 3
      } else if (kernels.unpackRowFunc !== null) {
        kernels.kind = 2
      }
      typeKernels[type] = kernels
    }
    var rs = getRowSize(cols, type)
    var off = baseOffset + t.offset
    var totalBytes = rows * rs
    var result = {
      dataOffset: off,
      type: type,
      rows: rows,
      cols: cols,
      rowSize: rs,
      // Kernels of the type and the view with the matrix position in it
      k: kernels,
      view: null,
      base: 0,
    }
    if (type === GGML_TYPE.Q8_0) {
      // Rows with an odd number of blocks keep the byte-wise kernel
      if (((off | rs) & 3) === 0) {
        result.view = matrixView(result, off, totalBytes, 0)
        result.k = Q8_LOCAL_KERNELS
      }
      return result
    }
    // The other formats are read as 16/32-bit words (GGUF aligns tensor data
    // to 32 bytes: this only rejects a corrupt file)
    var block32 = kernels.kind === 2
    var halfFloat = type === GGML_TYPE.F16 || type === GGML_TYPE.BF16 || type === 30
    if (
      (off & 3) !== 0 ||
      (block32 ? (rows & 3) !== 0 : kernels.kind !== 3) ||
      (halfFloat && (cols & 1) !== 0)
    ) {
      throw new Error("Unsupported tensor layout: " + name)
    }
    if (block32) {
      w.hasBlock32 = true
      result.view = matrixView(result, off, totalBytes, 1)
    } else {
      w.hasKQuant = true
      var viewKind = 0
      if (type === GGML_TYPE.Q3_K || type === GGML_TYPE.Q6_K) {
        viewKind = 1
      } else if (type === GGML_TYPE.F32) {
        viewKind = 2
      }
      result.view = matrixView(result, off, totalBytes, viewKind)
    }
    return result
  }

  function loadLayerTensorFloat(layer, suffix) {
    var name = "blk." + layer + "." + suffix
    return loadTensorFloat(name)
  }

  function loadLayerTensorQuantized(layer, suffix, rows, cols) {
    var name = "blk." + layer + "." + suffix
    var result = loadTensorQuantized(name, rows, cols)
    if (result !== null && result.k.kind === 1) {
      w.q8Layers = true
    }
    return result
  }

  var headSize = config.headDim
  var kvDim = config.nKvHeads * headSize
  var qDim = config.nHeads * headSize

  postMessage({ type: "progress", message: "Loading embeddings..." })
  // Token embeddings - keep quantized to save memory, dequantize on-demand
  w.tokenEmbedding = loadTensorQuantized(
    "token_embd.weight",
    config.vocabSize,
    config.dim
  )

  // Use per-layer arrays
  w.layers = []

  // Tensors of one layer
  function loadLayer(l) {
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

    return layer
  }

  for (var l = 0; l < config.nLayers; l = l + 1) {
    postMessage({
      type: "progress",
      message: "Loading layer " + (l + 1) + "/" + config.nLayers + "...",
    })

    w.layers.push(loadLayer(l))
  }

  // Final norm - small, dequantize
  w.rmsFinalWeight = loadTensorFloat("output_norm.weight")

  // Output projection - may be tied to embeddings or separate
  if (tensors["output.weight"]) {
    w.wcls = loadTensorQuantized("output.weight", config.vocabSize, config.dim)
  } else {
    // Tied embeddings: the classifier is the embedding matrix itself
    w.wcls = w.tokenEmbedding
  }

  return w
}

// RoPE frequencies of one head: 1 / theta^(2i / headSize)
function ropeFrequencies(theta, headSize) {
  var ropeSize = headSize / 2
  var freqs = new Float32Array(ropeSize)
  for (var i = 0; i < ropeSize; i = i + 1) {
    freqs[i] = 1.0 / Math.pow(theta, (i * 2) / headSize)
  }
  return freqs
}

function createRunState(p, hasBlock32) {
  var headSize = p.headDim
  var kvDim = p.nKvHeads * headSize
  var qDim = p.nHeads * headSize
  var maxDim = Math.max(p.dim, qDim)

  // RoPE frequencies (ropeSize floats); cos/sin filled per pass (fillRopeBuffers)
  var ropeSize = headSize / 2
  var ropeFreqs = ropeFrequencies(p.ropeTheta, headSize)
  var ropeSwaFreqs = ropeFreqs
  if (p.isGemma) {
    ropeSwaFreqs = ropeFrequencies(
      p.ropeThetaSwa > 0 ? p.ropeThetaSwa : 10000.0,
      headSize
    )
  }

  // Q8_0 KV cache, [layer][kv_head][position][head_data], grown on demand
  // (ensureKvCapacity) up to seqLen
  var headBytesQ8 = (headSize >> 5) * Q8_0_BLOCK_SIZE
  var initialCap = 0
  var headSeqBytes = 0
  var kvCacheLayerBytes = 0
  var kvCacheTotalBytes = 0

  // Create ArrayBuffer for KV caches with both Uint8 and Int8 views
  var keyCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)
  var valueCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)

  // Size of the Q8_0 scratch for x in matmulQuantized (allocated lazily)
  var maxCols = Math.max(p.dim, qDim, p.hiddenDim)
  xQ8Size = (maxCols >> 5) * 34
  xQ8Buf = null
  xQ8Int8Buf = null

  // generate() scratch: the largest of the hash table, the batch buffers and
  // the logits
  var hashBuckets = 256
  while (hashBuckets * 4 < p.vocabSize) {
    hashBuckets = hashBuckets << 1
  }
  // Per-token Q8_0 copy of the input, only for block-32 matrices
  var batchQ8Stride = hasBlock32 ? (xQ8Size + 7) & ~7 : 0
  var batchTokenBytes =
    (p.dim + maxDim + p.dim + qDim + kvDim + kvDim + p.hiddenDim + p.hiddenDim) * 4 +
    batchQ8Stride
  var scratchBytes = Math.max(
    (hashBuckets + p.vocabSize) * 4,
    maxCols * 32 + PREFILL_BATCH_SIZE * batchTokenBytes,
    p.vocabSize * 4
  )

  // Q8_0 buffer for quantized Q heads (for Q8 attention scoring)
  var qQ8TotalBytes = p.nHeads * headBytesQ8
  var qQ8Buffer = new ArrayBuffer(qQ8TotalBytes)

  // xb2 lives in hb and q, k, v in hb2 (never live with the FFN activations)
  var hbBuf = new Float32Array(p.hiddenDim)
  var hb2Buf = new Float32Array(p.hiddenDim)
  var xb2Buf = new Float32Array(p.dim)
  var qBuf = new Float32Array(qDim)
  var kBuf = new Float32Array(kvDim)
  var vBuf = new Float32Array(kvDim)
  if (p.dim <= p.hiddenDim) {
    xb2Buf = new Float32Array(hbBuf.buffer, 0, p.dim)
  }
  if (qDim + kvDim + kvDim <= p.hiddenDim) {
    qBuf = new Float32Array(hb2Buf.buffer, 0, qDim)
    kBuf = new Float32Array(hb2Buf.buffer, qDim * 4, kvDim)
    vBuf = new Float32Array(hb2Buf.buffer, (qDim + kvDim) * 4, kvDim)
  }

  var runState = {
    x: new Float32Array(p.dim),
    xb: new Float32Array(maxDim),
    xb2: xb2Buf,
    hb: hbBuf,
    hb2: hb2Buf,
    q: qBuf,
    k: kBuf,
    v: vBuf,
    // Attention scores, nHeads x kvCapacity (ensureKvCapacity)
    att: null,
    // Allocated on the first forward pass that needs it (ensureLogits)
    logits: null,
    logits64: null,
    // Float64 view over hb: idle scratch for a K-quant output matmul
    hb64: null,
    // Fallback one-row K-quant scratch, only if no idle buffer is wide enough.
    kqRowScratch: null,
    // Sizes for lazy batch-buffer allocation (ensureBatchBuffers).
    _batchDim: p.dim,
    _batchMaxDim: maxDim,
    _batchQDim: qDim,
    _batchKvDim: kvDim,
    _batchHiddenDim: p.hiddenDim,
    _batchXQ8Size: xQ8Size,
    _batchQ8Stride: batchQ8Stride,
    _batchMatmulDeqCols: maxCols,
    _batchBuffersReady: false,
    scratchBytes: scratchBytes,
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
    // RoPE scratch (fillRopeBuffers), [batchIdx * ropeSize + i], sized by
    // setRopeSlots(); non-Gemma: SwaAll === All
    ropeCosAll: null,
    ropeSinAll: null,
    ropeCosSwaAll: null,
    ropeSinSwaAll: null,
    ropeFreqs: ropeFreqs,
    ropeSwaFreqs: ropeSwaFreqs,
    ropeSize: ropeSize,
    // Cached constants to avoid recomputation in transformer
    headSize: headSize,
    kvDim: kvDim,
    qDim: qDim,
    kvMul: p.nHeads / p.nKvHeads,
    // Layer size in bytes of the Q8_0 cache
    kvCacheLayerSize: kvCacheLayerBytes,
    kvCapacity: initialCap,
    // Token of every cached position; generate() keeps the matching prefix
    kvTokens: new Int32Array(0),
    kvCount: 0,
    // How many of the leading positions were written by the batched prefill
    kvBatch: 0,
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
    headBytesQ8: headBytesQ8,
    // Batch buffers of the prefill (ensureBatchBuffers / freeBatchBuffers)
    batchX: null,
    batchXb: null,
    batchXb2: null,
    batchQ: null,
    batchK: null,
    batchV: null,
    batchHb: null,
    batchHb2: null,
    batchQ8: null,
    batchQ8i8: null,
    // Float64 staging views of the matmul inputs (ensureBatchBuffers)
    batchStageB: null,
    batchStageC: null,
  }
  runState.hb64 = new Float64Array(hbBuf.buffer, 0, p.hiddenDim >> 1)
  setRopeSlots(runState, 1)
  return runState
}

// (Re)allocate the RoPE cos/sin scratch for `slots` positions per forward pass
function setRopeSlots(s, slots) {
  var n = slots * s.ropeSize
  s.ropeCosAll = new Float32Array(n)
  s.ropeSinAll = new Float32Array(n)
  if (s.isGemma) {
    s.ropeCosSwaAll = new Float32Array(n)
    s.ropeSinSwaAll = new Float32Array(n)
  } else {
    s.ropeCosSwaAll = s.ropeCosAll
    s.ropeSinSwaAll = s.ropeSinAll
  }
}

// ----------------------------------------------------------------------------
// Transformer forward pass

// n views of len elements, stride bytes apart from off (per-token buffers)
function tokenViews(Ctor, buf, off, stride, len, n) {
  var views = new Array(n)
  for (var b = 0; b < n; b = b + 1) {
    views[b] = new Ctor(buf, off + b * stride, len)
  }
  return views
}

// Allocate the prefill batch buffers (2.5 MB) on first prefill
function ensureBatchBuffers(s) {
  if (s._batchBuffersReady) {
    return
  }
  var n = PREFILL_BATCH_SIZE
  var dim = s._batchDim
  var maxDim = s._batchMaxDim
  var qDim = s._batchQDim
  var kvDim = s._batchKvDim
  var hiddenDim = s._batchHiddenDim
  var deqCols = s._batchMatmulDeqCols
  ensureGenScratch(s)
  var buf = genScratch
  // matmulDeqBuf: the first 32 * deqCols bytes, read by the batch kernels
  matmulDeqBuf = new Float64Array(buf, 0, 4 * deqCols)
  matmulDeqI8 = new Int8Array(buf, 0, 32 * deqCols)
  matmulDeqRows = tokenViews(Float64Array, buf, 0, deqCols * 8, deqCols, 4)
  // One region per kind of buffer, tokens one after another; the regions idle
  // during a matmul hold the Float64 staging of its input: B over xb2/q/k/v
  // (w1, w3), C over hb and hb2 (QKV, wo, w2)
  var off = deqCols * 32
  s.batchX = tokenViews(Float32Array, buf, off, dim * 4, dim, n)
  off = off + n * dim * 4
  s.batchXb = tokenViews(Float32Array, buf, off, maxDim * 4, maxDim, n)
  off = off + n * maxDim * 4
  var offB = off
  s.batchXb2 = tokenViews(Float32Array, buf, off, dim * 4, dim, n)
  off = off + n * dim * 4
  s.batchQ = tokenViews(Float32Array, buf, off, qDim * 4, qDim, n)
  off = off + n * qDim * 4
  s.batchK = tokenViews(Float32Array, buf, off, kvDim * 4, kvDim, n)
  off = off + n * kvDim * 4
  s.batchV = tokenViews(Float32Array, buf, off, kvDim * 4, kvDim, n)
  off = off + n * kvDim * 4
  var offC = off
  s.batchHb = tokenViews(Float32Array, buf, off, hiddenDim * 4, hiddenDim, n)
  off = off + n * hiddenDim * 4
  s.batchHb2 = tokenViews(Float32Array, buf, off, hiddenDim * 4, hiddenDim, n)
  off = off + n * hiddenDim * 4
  if (maxDim <= hiddenDim && dim + qDim + 2 * kvDim >= 2 * maxDim) {
    s.batchStageB = tokenViews(Float64Array, buf, offB, maxDim * 8, maxDim, n)
    s.batchStageC = tokenViews(Float64Array, buf, offC, hiddenDim * 8, hiddenDim, n)
  }
  var q8Stride = s._batchQ8Stride
  if (q8Stride > 0) {
    var xQ8Size = s._batchXQ8Size
    s.batchQ8 = tokenViews(Uint8Array, buf, off, q8Stride, xQ8Size, n)
    s.batchQ8i8 = tokenViews(Int8Array, buf, off, q8Stride, xQ8Size, n)
  }
  setRopeSlots(s, PREFILL_BATCH_SIZE)
  s._batchBuffersReady = true
}

// Release the prefill batch buffers once prefill is over
function freeBatchBuffers(s) {
  if (!s._batchBuffersReady) {
    return
  }
  s.batchX = null
  s.batchXb = null
  s.batchXb2 = null
  s.batchQ = null
  s.batchK = null
  s.batchV = null
  s.batchHb = null
  s.batchHb2 = null
  s.batchQ8 = null
  s.batchQ8i8 = null
  s.batchStageB = null
  s.batchStageC = null
  matmulDeqBuf = null
  matmulDeqRows = null
  matmulDeqI8 = null
  setRopeSlots(s, 1)
  s._batchBuffersReady = false
}

// Allocate the logits buffer on the first forward pass that needs it
function ensureLogits(s) {
  if (s.logits === null) {
    ensureGenScratch(s)
    s.logits = new Float32Array(genScratch, 0, s.vocabSize)
    // Float64 view over the same bytes: K-quant scratch while the layers run
    s.logits64 = new Float64Array(genScratch, 0, s.vocabSize >> 1)
  }
}

function ensureGenScratch(s) {
  if (genScratch === null) {
    genScratch = new ArrayBuffer(s.scratchBytes)
  }
}

// Grow the KV cache (doubling, capped at seqLen), copying the positions written
function ensureKvCapacity(s, needed) {
  var cap = s.kvCapacity
  if (needed <= cap) {
    return
  }
  var maxCap = s.seqLen
  // From cap, or 1 on the first pass, doubling up to needed
  var newCap = cap > 0 ? cap : 1
  while (newCap < needed) {
    newCap = newCap * 2
  }
  if (newCap > maxCap) {
    newCap = maxCap
  }

  var nLayers = s.nLayers
  var nKvHeads = s.nKvHeads
  var headBytesQ8 = s.headBytesQ8
  var oldHeadSeq = cap * headBytesQ8
  var newHeadSeq = newCap * headBytesQ8
  var newLayerBytes = nKvHeads * newHeadSeq
  var newTotal = nLayers * newLayerBytes

  var newKeyBuf = new ArrayBuffer(newTotal)
  var newValBuf = new ArrayBuffer(newTotal)
  var newKey = new Uint8Array(newKeyBuf)
  var newVal = new Uint8Array(newValBuf)
  var oldKey = s.keyCache
  var oldVal = s.valueCache
  var oldLayerBytes = nKvHeads * oldHeadSeq
  for (var l = 0; l < nLayers; l = l + 1) {
    var oldLayerBase = l * oldLayerBytes
    var newLayerBase = l * newLayerBytes
    for (var h = 0; h < nKvHeads; h = h + 1) {
      var oldHeadBase = oldLayerBase + h * oldHeadSeq
      var newHeadBase = newLayerBase + h * newHeadSeq
      newKey.set(oldKey.subarray(oldHeadBase, oldHeadBase + oldHeadSeq), newHeadBase)
      newVal.set(oldVal.subarray(oldHeadBase, oldHeadBase + oldHeadSeq), newHeadBase)
    }
  }

  s.keyCache = newKey
  s.valueCache = newVal
  s.keyCacheInt8 = new Int8Array(newKeyBuf)
  s.valueCacheInt8 = new Int8Array(newValBuf)
  s.headSeqBytes = newHeadSeq
  s.kvCacheLayerSize = newLayerBytes
  s.kvCapacity = newCap
  var newTokens = new Int32Array(newCap)
  newTokens.set(s.kvTokens)
  s.kvTokens = newTokens
  // Scores of one position against every cached position (plain scratch)
  s.att = new Float32Array(s.nHeads * newCap)
}

// Fill the RoPE cos/sin scratch for batchSize positions from startPos
function fillRopeBuffers(s, startPos, batchSize) {
  var ropeSize = s.ropeSize
  var freqs = s.ropeFreqs
  var cos = s.ropeCosAll
  var sin = s.ropeSinAll
  for (var b = 0; b < batchSize; b = b + 1) {
    var pos = startPos + b
    var base = b * ropeSize
    for (var i = 0; i < ropeSize; i = i + 1) {
      var v = pos * freqs[i]
      cos[base + i] = Math.cos(v)
      sin[base + i] = Math.sin(v)
    }
  }
  if (s.ropeCosSwaAll !== cos) {
    var swaFreqs = s.ropeSwaFreqs
    var swaCos = s.ropeCosSwaAll
    var swaSin = s.ropeSinSwaAll
    for (var b = 0; b < batchSize; b = b + 1) {
      var pos = startPos + b
      var base = b * ropeSize
      for (var i = 0; i < ropeSize; i = i + 1) {
        var v = pos * swaFreqs[i]
        swaCos[base + i] = Math.cos(v)
        swaSin[base + i] = Math.sin(v)
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Per-token element loops (small functions keep the transformer code small)

// SwiGLU gate: hb = silu(hb) * hb2 with silu(x) = 0.5 * x * (1 + tanh(x / 2))
function siluGate(hbArr, hb2Arr, n) {
  for (var i = 0; i < n; i = i + 1) {
    var val = hbArr[i]
    hbArr[i] = 0.5 * val * (1.0 + fastTanh(0.5 * val)) * hb2Arr[i]
  }
}
// GeGLU gate: hb = gelu(hb) * hb2 (tanh approximation)
function geluGate(hbArr, hb2Arr, n) {
  var GELU_A = 0.7978845608
  var GELU_B = 0.035677408137
  for (var i = 0; i < n; i = i + 1) {
    var xv = hbArr[i]
    hbArr[i] =
      0.5 * xv * (1.0 + fastTanh(xv * (GELU_A + GELU_B * xv * xv))) * hb2Arr[i]
  }
}
// Llama RoPE on consecutive pairs; Q also gets the attention scale folded in.
function ropeLlama(
  qArr,
  kArr,
  qDim,
  kvDim,
  half,
  ropeCos,
  ropeSin,
  ropeBase,
  attnScale
) {
  for (var i = 0; i < kvDim; i = i + 2) {
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
}
// Gemma (NEOX) RoPE on pairs (i, i + half) of every head, times scale (1.0 for K)
function ropeNeox(arr, nHeads, headSize, half, ropeCos, ropeSin, ropeBase, scale) {
  for (var h = 0; h < nHeads; h = h + 1) {
    var idx = h * headSize
    for (var i = 0; i < half; i = i + 1) {
      var fcr = ropeCos[ropeBase + i]
      var fci = ropeSin[ropeBase + i]
      var v0 = arr[idx + i]
      var v1 = arr[idx + i + half]
      arr[idx + i] = (v0 * fcr - v1 * fci) * scale
      arr[idx + i + half] = (v0 * fci + v1 * fcr) * scale
    }
  }
}

// GQA attention with Q8 scoring for one layer and position: the Q heads of a
// KV group against the cache, softmax over [startT, pos], values into xbArr
function attendQ8(s, loff, pos, startT, xbArr) {
  var qDim = s.qDim
  for (var i = 0; i < qDim; i = i + 1) {
    xbArr[i] = 0
  }
  var nKvHeads = s.nKvHeads
  var kvMul = s.kvMul
  var headSeqBytes = s.headSeqBytes
  var headBytesQ8 = s.headBytesQ8
  var headSize = s.headSize
  var attStride = s.kvCapacity
  var sAtt = s.att
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8
  var keyCache = s.keyCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCache = s.valueCache
  var valueCacheInt8 = s.valueCacheInt8
  for (var kvH = 0; kvH < nKvHeads; kvH = kvH + 1) {
    var kBase = loff + kvH * headSeqBytes

    // Score all Q heads in this GQA group against all K positions
    for (var t = startT; t <= pos; t = t + 1) {
      var kOff = kBase + t * headBytesQ8
      for (var mh = 0; mh < kvMul; mh = mh + 1) {
        var h = kvH * kvMul + mh
        sAtt[h * attStride + t] = dotQ8_0_Q8_0Cache(
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
      var attOffset = h * attStride

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
}

// Two or three single-token matmuls over one input (block-32: x quantized once)
function matmulSameInput(x, cols, o1, q1, o2, q2, o3, q3) {
  if (q1.k.kind === 2 && q2.k.kind === 2 && (q3 === null || q3.k.kind === 2)) {
    ensureXQ8Buf()
    quantizeToQ8_0Cache(x, 0, xQ8Buf, xQ8Int8Buf, 0, cols)
    matmulQuantizedPreQ8(o1, q1)
    matmulQuantizedPreQ8(o2, q2)
    if (q3 !== null) {
      matmulQuantizedPreQ8(o3, q3)
    }
  } else {
    matmulQuantized(o1, x, q1)
    matmulQuantized(o2, x, q2)
    if (q3 !== null) {
      matmulQuantized(o3, x, q3)
    }
  }
}

// Gemma final logit soft-capping
function softcapLogits(logits, n, cap) {
  for (var i = 0; i < n; i = i + 1) {
    logits[i] = cap * fastTanh(logits[i] / cap)
  }
}

// Transformer forward pass, Llama and Gemma (Gemma adds QK norms, NEOX RoPE,
// sliding window, GELU, post-norms). transformer: one token at pos, single-
// token kernels, final norm and logits
function transformer(token, pos, computeLogits) {
  var w = weights
  var s = state
  if (w.hasKQuant) {
    // K-quant matmuls borrow the logits buffer as scratch during generation
    ensureLogits(s)
  }
  if (pos >= s.kvCapacity) {
    ensureKvCapacity(s, pos + 1)
  }
  var isGemma = s.isGemma
  var dim = s.dim
  var hiddenDim = s.hiddenDim
  var eps = s.rmsNormEps
  var headSize = s.headSize
  var qDim = s.qDim
  var nHeads = s.nHeads
  var nKvHeads = s.nKvHeads
  var invHeadSize = s.invHeadSize
  var attnScale = s.attnScale
  var half = headSize >> 1
  // Llama norms keep their 1e-5
  var normEps = isGemma ? eps : 1e-5
  var nLayers = s.nLayers
  var invDim = s.invDim

  var xArr = s.x
  var xbArr = s.xb
  var xb2Arr = s.xb2
  var qArr = s.q
  var kArr = s.k
  var vArr = s.v
  var hbArr = s.hb
  var hb2Arr = s.hb2

  // Embedding (Gemma: scaling fused into the first rmsnorm)
  embedToken(xArr, token)

  // Fill the RoPE scratch buffers for this position (slot 0)
  fillRopeBuffers(s, pos, 1)

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l]

    if (isGemma && l === 0) {
      // First layer: fused embed scale + rmsnorm
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
      rmsnorm(xbArr, xArr, 0, lw.rmsAttWeight, dim, normEps, invDim)
    }

    // QKV matmuls
    matmulSameInput(xbArr, dim, qArr, lw.wq, kArr, lw.wk, vArr, lw.wv)

    // QK norms, RoPE, KV cache write, attention
    var loff = l * s.kvCacheLayerSize
    var isSwaLayer =
      isGemma && s.swaPattern > 0 && l % s.swaPattern < s.swaPattern - 1
    var ropeCos = isSwaLayer ? s.ropeCosSwaAll : s.ropeCosAll
    var ropeSin = isSwaLayer ? s.ropeSinSwaAll : s.ropeSinAll

    if (isGemma) {
      if (lw.attnQNorm && lw.attnKNorm) {
        for (var h = 0; h < nHeads; h = h + 1) {
          rmsnorm(qArr, qArr, h * headSize, lw.attnQNorm, headSize, eps, invHeadSize)
        }
        for (var h = 0; h < nKvHeads; h = h + 1) {
          rmsnorm(kArr, kArr, h * headSize, lw.attnKNorm, headSize, eps, invHeadSize)
        }
      }
      // NEOX RoPE with fused attnScale on Q
      ropeNeox(qArr, nHeads, headSize, half, ropeCos, ropeSin, 0, attnScale)
      ropeNeox(kArr, nKvHeads, headSize, half, ropeCos, ropeSin, 0, 1.0)
    } else {
      // RoPE with fused attnScale on Q
      ropeLlama(qArr, kArr, qDim, s.kvDim, half, ropeCos, ropeSin, 0, attnScale)
    }

    // Per-head KV cache write
    for (var h = 0; h < nKvHeads; h = h + 1) {
      var headOff = loff + h * s.headSeqBytes + pos * s.headBytesQ8
      quantizeToQ8_0Cache(
        kArr,
        h * headSize,
        s.keyCache,
        s.keyCacheInt8,
        headOff,
        headSize
      )
      quantizeToQ8_0Cache(
        vArr,
        h * headSize,
        s.valueCache,
        s.valueCacheInt8,
        headOff,
        headSize
      )
    }

    // Quantize all Q heads to Q8_0
    quantizeToQ8_0Cache(qArr, 0, s.qQ8, s.qQ8i8, 0, qDim)

    // SWA window enforcement
    var startT =
      isSwaLayer && config.swaWindow > 0
        ? Math.max(0, pos - config.swaWindow + 1)
        : 0

    attendQ8(s, loff, pos, startT, xbArr)

    // Attention output
    matmulQuantized(xb2Arr, xbArr, lw.wo)
    if (isGemma && lw.attnPostNorm) {
      rmsnorm(xb2Arr, xb2Arr, 0, lw.attnPostNorm, dim, eps, invDim)
    }
    accum(xArr, xb2Arr, dim)

    // FFN
    rmsnorm(xbArr, xArr, 0, lw.rmsFfnWeight, dim, normEps, invDim)
    matmulSameInput(xbArr, dim, hbArr, lw.w1, hb2Arr, lw.w3, null, null)
    if (isGemma) {
      geluGate(hbArr, hb2Arr, hiddenDim)
    } else {
      siluGate(hbArr, hb2Arr, hiddenDim)
    }
    matmulQuantized(xbArr, hbArr, lw.w2)
    if (isGemma && lw.ffnPostNorm) {
      rmsnorm(xbArr, xbArr, 0, lw.ffnPostNorm, dim, eps, invDim)
    }
    accum(xArr, xbArr, dim)
  }

  // Final norm
  rmsnorm(xArr, xArr, 0, w.rmsFinalWeight, dim, normEps, invDim)

  // Classifier into logits
  if (computeLogits !== false) {
    ensureLogits(s)
    matmulQuantized(s.logits, xArr, w.wcls)
    if (isGemma && config.finalLogitSoftcapping > 0) {
      softcapLogits(s.logits, s.vocabSize, config.finalLogitSoftcapping)
    }
  }

  // Note which token this pass left in the KV cache
  s.kvTokens[pos] = token
  s.kvCount = pos + 1
  if (s.kvBatch > pos) {
    s.kvBatch = pos
  }
}

// transformerPrefill: tokens allTokens[startPos .. startPos + n - 1] in one
// batch; every matrix is read once, only the KV cache is produced
function transformerPrefill(allTokens, startPos, n) {
  var w = weights
  var s = state
  if (startPos + n > s.kvCapacity) {
    ensureKvCapacity(s, startPos + n)
  }
  ensureBatchBuffers(s)
  var isGemma = s.isGemma
  var dim = s.dim
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var eps = s.rmsNormEps
  var headSize = s.headSize
  var nHeads = s.nHeads
  var nKvHeads = s.nKvHeads
  var invHeadSize = s.invHeadSize
  var attnScale = s.attnScale
  var half = headSize >> 1
  var kvDim = s.kvDim
  var headBytesQ8 = s.headBytesQ8
  var headSeqBytes = s.headSeqBytes
  var keyCache = s.keyCache
  var valueCache = s.valueCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCacheInt8 = s.valueCacheInt8
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8
  // Llama norms keep their 1e-5
  var normEps = isGemma ? eps : 1e-5
  var nLayers = s.nLayers
  var invDim = s.invDim

  // One buffer of each kind per token
  var bX = s.batchX
  var bXb = s.batchXb
  var bXb2 = s.batchXb2
  var bQ = s.batchQ
  var bK = s.batchK
  var bV = s.batchV
  var bHb = s.batchHb
  var bHb2 = s.batchHb2
  // Float64 staging views: C for QKV, wo and w2, B for w1 and w3
  var stB = s.batchStageB
  var stC = s.batchStageC

  // Embedding (Gemma: scaling fused into the first rmsnorm)
  for (var b = 0; b < n; b = b + 1) {
    embedToken(bX[b], allTokens[startPos + b])
  }

  // Fill the RoPE scratch buffers for the positions of this pass
  fillRopeBuffers(s, startPos, n)

  for (var l = 0; l < nLayers; l = l + 1) {
    var lw = w.layers[l]
    // The last layer only has to leave K and V in the cache
    var kvOnly = l === nLayers - 1

    for (var b = 0; b < n; b = b + 1) {
      if (isGemma && l === 0) {
        // First layer: fused embed scale + rmsnorm
        rmsnormGemmaFusedScale(
          bXb[b],
          bX[b],
          lw.rmsAttWeight,
          dim,
          eps,
          invDim,
          s.embedScale
        )
      } else {
        rmsnorm(bXb[b], bX[b], 0, lw.rmsAttWeight, dim, normEps, invDim)
      }
    }

    // QKV matmuls; the first float tile (odd kind) fills the staging
    var tq = !kvOnly && (lw.wq.k.kind & 1) !== 0
    var tk = (lw.wk.k.kind & 1) !== 0
    if (!kvOnly) {
      matmulQuantizedBatch(bQ, bXb, lw.wq, n, stC, tq)
    }
    matmulQuantizedBatch(bK, bXb, lw.wk, n, stC, tk && !tq)
    matmulQuantizedBatch(
      bV,
      bXb,
      lw.wv,
      n,
      stC,
      (lw.wv.k.kind & 1) !== 0 && !tq && !tk
    )

    // QK norms, RoPE, KV cache write, attention of every token
    var loff = l * s.kvCacheLayerSize
    var isSwaLayer =
      isGemma && s.swaPattern > 0 && l % s.swaPattern < s.swaPattern - 1
    var ropeCos = isSwaLayer ? s.ropeCosSwaAll : s.ropeCosAll
    var ropeSin = isSwaLayer ? s.ropeSinSwaAll : s.ropeSinAll

    for (var b = 0; b < n; b = b + 1) {
      var pos = startPos + b
      var qArr = bQ[b]
      var kArr = bK[b]
      var vArr = bV[b]
      var xbArr = bXb[b]
      var ropeBase = b * s.ropeSize

      if (isGemma) {
        if (lw.attnQNorm && lw.attnKNorm) {
          if (!kvOnly) {
            for (var h = 0; h < nHeads; h = h + 1) {
              rmsnorm(
                qArr,
                qArr,
                h * headSize,
                lw.attnQNorm,
                headSize,
                eps,
                invHeadSize
              )
            }
          }
          for (var h = 0; h < nKvHeads; h = h + 1) {
            rmsnorm(
              kArr,
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
        if (!kvOnly) {
          ropeNeox(
            qArr,
            nHeads,
            headSize,
            half,
            ropeCos,
            ropeSin,
            ropeBase,
            attnScale
          )
        }
        ropeNeox(kArr, nKvHeads, headSize, half, ropeCos, ropeSin, ropeBase, 1.0)
      } else {
        // RoPE with fused attnScale on Q
        ropeLlama(
          qArr,
          kArr,
          qDim,
          kvDim,
          half,
          ropeCos,
          ropeSin,
          ropeBase,
          attnScale
        )
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

      if (kvOnly) {
        continue
      }

      // Quantize all Q heads to Q8_0
      quantizeToQ8_0Cache(qArr, 0, qQ8, qQ8i8, 0, qDim)

      // SWA window enforcement
      var startT =
        isSwaLayer && config.swaWindow > 0
          ? Math.max(0, pos - config.swaWindow + 1)
          : 0

      attendQ8(s, loff, pos, startT, xbArr)
    }

    if (kvOnly) {
      break
    }

    // Attention output
    matmulQuantizedBatch(bXb2, bXb, lw.wo, n, stC, (lw.wo.k.kind & 1) !== 0)
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma && lw.attnPostNorm) {
        rmsnorm(bXb2[b], bXb2[b], 0, lw.attnPostNorm, dim, eps, invDim)
      }
      accum(bX[b], bXb2[b], dim)
    }

    // FFN
    for (var b = 0; b < n; b = b + 1) {
      rmsnorm(bXb[b], bX[b], 0, lw.rmsFfnWeight, dim, normEps, invDim)
    }
    var t1 = (lw.w1.k.kind & 1) !== 0
    matmulQuantizedBatch(bHb, bXb, lw.w1, n, stB, t1)
    matmulQuantizedBatch(bHb2, bXb, lw.w3, n, stB, (lw.w3.k.kind & 1) !== 0 && !t1)
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma) {
        geluGate(bHb[b], bHb2[b], hiddenDim)
      } else {
        siluGate(bHb[b], bHb2[b], hiddenDim)
      }
    }
    // hb is consumed by its staging (w2 is its only reader)
    matmulQuantizedBatch(bXb, bHb, lw.w2, n, stC, (lw.w2.k.kind & 1) !== 0)
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma && lw.ffnPostNorm) {
        rmsnorm(bXb[b], bXb[b], 0, lw.ffnPostNorm, dim, eps, invDim)
      }
      accum(bX[b], bXb[b], dim)
    }
  }

  // Note which tokens this pass left in the KV cache
  for (var b = 0; b < n; b = b + 1) {
    s.kvTokens[startPos + b] = allTokens[startPos + b]
  }
  s.kvCount = startPos + n
  if (s.kvBatch >= startPos) {
    s.kvBatch = startPos + n
  }
}

// Dequantize one token's embedding row into dst
function embedToken(dst, token) {
  var emb = weights.tokenEmbedding
  if (emb.k.kind === 3) {
    emb.k.deqRowFunc(
      emb.view,
      emb.base + (token * emb.rowSize) / emb.view.BYTES_PER_ELEMENT,
      dst,
      0,
      emb.cols
    )
  } else {
    dequantizeRow(dst, emb.dataOffset + token * emb.rowSize, emb.cols, emb.type)
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

// Vocab strings stay in the GGUF buffer; offset of token i: its checkpoint plus
// the lengths of the tokens before it in the bucket
function vocabOffsetOf(i) {
  var bucket = i >> 8
  var base = tokenizer.vocabSparseCum[bucket]
  var lengths = tokenizer.vocabLengths
  var bucketStart = bucket << 8
  for (var k = bucketStart; k < i; k = k + 1) {
    base = base + lengths[k] + 8
  }
  return base
}

function vocabString(i) {
  var n = tokenizer.vocabSize
  if (i < 0 || i >= n) {
    return ""
  }
  ensureVocabTables()
  var len = tokenizer.vocabLengths[i]
  if (len === 0) {
    return ""
  }
  var off = vocabOffsetOf(i)
  return decodeUTF8(ggufUint8.subarray(off, off + len))
}

// Tokenizer lookup: chained hash table over the token bytes, built on the first
// bpeEncode() of a generate() call and released after the prompt (head[hash &
// mask] = highest id of the chain, next[id] = the next lower one, -1 ends)
var vocabHashHead = null
var vocabHashNext = null

// Encode a JS string to UTF-8 bytes (buffer bound: 3 bytes per code unit)
function encodeStringToUTF8(str) {
  var len = str.length
  var bytes = new Uint8Array(len * 3 + 1)
  var bi = 0
  for (var i = 0; i < len; i = i + 1) {
    var c = str.charCodeAt(i)
    if (c < 0x80) {
      bytes[bi] = c
      bi = bi + 1
    } else if (c < 0x800) {
      bytes[bi] = 0xc0 | (c >> 6)
      bytes[bi + 1] = 0x80 | (c & 0x3f)
      bi = bi + 2
    } else if (c >= 0xd800 && c <= 0xdbff && i + 1 < len) {
      var low = str.charCodeAt(i + 1)
      if (low >= 0xdc00 && low <= 0xdfff) {
        var cp = 0x10000 + ((c - 0xd800) << 10) + (low - 0xdc00)
        bytes[bi] = 0xf0 | (cp >> 18)
        bytes[bi + 1] = 0x80 | ((cp >> 12) & 0x3f)
        bytes[bi + 2] = 0x80 | ((cp >> 6) & 0x3f)
        bytes[bi + 3] = 0x80 | (cp & 0x3f)
        bi = bi + 4
        i = i + 1
      } else {
        bytes[bi] = 0xe0 | (c >> 12)
        bytes[bi + 1] = 0x80 | ((c >> 6) & 0x3f)
        bytes[bi + 2] = 0x80 | (c & 0x3f)
        bi = bi + 3
      }
    } else {
      bytes[bi] = 0xe0 | (c >> 12)
      bytes[bi + 1] = 0x80 | ((c >> 6) & 0x3f)
      bytes[bi + 2] = 0x80 | (c & 0x3f)
      bi = bi + 3
    }
  }
  return bytes.subarray(0, bi)
}

function buildVocabHash() {
  if (vocabHashHead) {
    return
  }
  var vocabLen = tokenizer.vocabSize
  var lengths = tokenizer.vocabLengths
  var sparseCum = tokenizer.vocabSparseCum
  var u8 = ggufUint8
  // Power-of-two bucket count, about 4 tokens per chain
  var buckets = 256
  while (buckets * 4 < vocabLen) {
    buckets = buckets << 1
  }
  var mask = buckets - 1
  ensureGenScratch(state)
  var head = new Int32Array(genScratch, 0, buckets)
  var next = new Int32Array(genScratch, buckets * 4, vocabLen)
  for (var b = 0; b < buckets; b = b + 1) {
    head[b] = -1
  }
  // Byte offset of token i, re-anchored at every checkpoint
  var off = 0
  for (var i = 0; i < vocabLen; i = i + 1) {
    if ((i & 255) === 0) {
      off = sparseCum[i >> 8]
    }
    var li = lengths[i]
    if (li > 0) {
      var h = 0
      for (var k = 0; k < li; k = k + 1) {
        h = ((h << 5) - h + u8[off + k]) | 0
      }
      next[i] = head[h & mask]
      head[h & mask] = i
    }
    off = off + li + 8
  }
  vocabHashHead = head
  vocabHashNext = next
}

// First use of the vocabulary: per-token byte lengths, one offset checkpoint
// every 256 tokens, the end-of-turn token
function ensureVocabTables() {
  var t = tokenizer
  if (t.vocabLengths !== null) {
    return
  }
  var n = t.vocabSize
  var lengths = t.vocabMaxLen < 256 ? new Uint8Array(n) : new Uint16Array(n)
  var sparseCum = new Uint32Array(((n - 1) >> 8) + 1)
  var dv = dataView
  var p = t.vocabOffset
  for (var i = 0; i < n; i = i + 1) {
    var slen = dv.getUint32(p, true)
    p = p + 8
    if ((i & 255) === 0) {
      // Checkpoint: where the bytes of token i start
      sparseCum[i >> 8] = p
    }
    lengths[i] = slen
    p = p + slen
  }
  t.vocabLengths = lengths
  t.vocabSparseCum = sparseCum
  if (config.isGemma) {
    var endTurn = findSpecialToken("<end_of_turn>")
    t.eotToken = endTurn < 0 ? 107 : endTurn
  } else {
    var eot = findSpecialToken("<|eot_id|>")
    t.eotToken = eot < 0 ? 128009 : eot
  }
}

// Id of a special token given its text (linear scan, cached per text)
function findSpecialToken(tokenStr) {
  var cache = tokenizer.specialTokens
  if (cache[tokenStr] !== undefined) {
    return cache[tokenStr]
  }
  var target = encodeStringToUTF8(tokenStr)
  var tLen = target.length
  var u8 = ggufUint8
  var dv = dataView
  var n = tokenizer.vocabSize
  var p = tokenizer.vocabOffset
  var id = -1
  for (var i = 0; i < n && id < 0; i = i + 1) {
    var li = dv.getUint32(p, true)
    p = p + 8
    if (li === tLen) {
      var k = 0
      while (k < tLen && u8[p + k] === target[k]) {
        k = k + 1
      }
      if (k === tLen) {
        id = i
      }
    }
    p = p + li
  }
  cache[tokenStr] = id
  return id
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
  var needPrefix = true // Add \u2581 before first alphanumeric char

  for (var i = 0; i < text.length; i = i + 1) {
    var c = text.charAt(i)
    var code = text.charCodeAt(i)

    if (c === " ") {
      // Space -> \u2581 (U+2581)
      parts.push("\u2581")
      needPrefix = false // \u2581 already added for the space
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

// Streaming UTF-8 decoder for Llama tokens (created per inference run)

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
  if (token < 0 || token >= tokenizer.vocabSize) {
    return new Uint8Array(0)
  }
  var piece = vocabString(token)
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
  if (token < 0 || token >= tokenizer.vocabSize) {
    return ""
  }
  var piece = vocabString(token)
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

  // For Gemma (SentencePiece), just replace \u2581 with space
  if (config.isGemma) {
    return piece.replace(/\u2581/g, " ")
  }

  // For Llama (tiktoken), decode UTF-8 bytes to string
  var bytes = tokenToBytes(token)
  return decodeUTF8(bytes)
}

// Longest token that is a prefix of bytes[pos .. pos + limit), or -1
function vocabLongestMatch(bytes, pos, limit) {
  var head = vocabHashHead
  var next = vocabHashNext
  var mask = head.length - 1
  var lengths = tokenizer.vocabLengths
  var sparseCum = tokenizer.vocabSparseCum
  var u8 = ggufUint8
  var best = -1
  var h = 0
  for (var n = 1; n <= limit; n = n + 1) {
    h = ((h << 5) - h + bytes[pos + n - 1]) | 0
    var id = head[h & mask]
    while (id !== -1) {
      if (lengths[id] === n) {
        // Offset of token id (vocabOffsetOf)
        var off = sparseCum[id >> 8]
        for (var j = id & ~255; j < id; j = j + 1) {
          off = off + lengths[j] + 8
        }
        var k = 0
        while (k < n && u8[off + k] === bytes[pos + k]) {
          k = k + 1
        }
        if (k === n) {
          best = id
          break
        }
      }
      id = next[id]
    }
  }
  return best
}

function bpeEncode(text) {
  buildVocabHash()

  // Convert text based on tokenizer type
  var encodedText
  if (config.isGemma) {
    encodedText = textToSentencePiece(text)
  } else {
    encodedText = textToTiktoken(text)
  }

  var tokens = []

  // Convert encodedText to UTF-8 bytes - the vocabulary is matched by bytes.
  var bytes = encodeStringToUTF8(encodedText)
  var byteLen = bytes.length
  var pos = 0

  var maxLen = tokenizer.vocabMaxLen
  var lengths = tokenizer.vocabLengths

  while (pos < byteLen) {
    var limit = byteLen - pos
    if (limit > maxLen) {
      limit = maxLen
    }
    var bestId = vocabLongestMatch(bytes, pos, limit)
    if (bestId !== -1) {
      tokens.push(bestId)
      pos = pos + lengths[bestId]
    } else {
      // No prefix match - skip this byte
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
    var role = chatHistory[m].type
    var content = getMessageContent(chatHistory[m])

    // Llama uses "assistant" instead of "model"
    var llamaRole = role === "model" ? "assistant" : role

    tokens.push(startHeader)
    var roleTokens = bpeEncode(llamaRole)
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

// Returns the text of a chat history message: model messages keep it in a
// "response" array and the rest of them in "text"
function getMessageContent(message) {
  if (message.type === "model") {
    return message.response.join("")
  }

  return message.text
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
    var role = chatHistory[m].type
    var content = getMessageContent(chatHistory[m])

    tokens.push(startTurn)

    var roleText = role + "\n"
    // Merge system prompt into first user message
    if (!systemUsed && role === "user" && sysPrompt && sysPrompt.length > 0) {
      roleText = role + "\n" + sysPrompt + "\n\n"
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
  ensureFp16Table()
  ensureVocabTables()

  // The system prompt is read from the "system" message of the chat history.
  // Without one, the system prompt given when loading the model is used
  var sysPrompt = systemPrompt
  var messages = []
  for (var m = 0; m < chatHistory.length; m = m + 1) {
    if (chatHistory[m].type === "system") {
      sysPrompt = chatHistory[m].text
    } else {
      messages.push(chatHistory[m])
    }
  }

  var promptTokens
  if (config.isGemma) {
    promptTokens = encodeGemma3Chat(messages, sysPrompt)
  } else {
    promptTokens = encodeLlama3Chat(messages, sysPrompt)
  }

  // Release the vocabulary lookup table (only needed to encode the prompt)
  vocabHashHead = null
  vocabHashNext = null

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

  // Streaming UTF-8 decoder for Llama (sequences cut by token boundaries)
  var streamDecoder = config.isGemma ? null : createStreamingUTF8Decoder()

  var effectiveMaxTokens = maxTokens
  if (effectiveMaxTokens <= 0 || effectiveMaxTokens > config.seqLen) {
    effectiveMaxTokens = config.seqLen
  }

  // Reuse the KV cache of the previous call: positions whose tokens match the
  // start of this prompt stay (the last prompt token always runs again). With
  // Q8_0 layer matrices only prefilled positions are kept: the single-token
  // kernel sums in another order
  var kvTokens = state.kvTokens
  var reuse = state.kvCount
  if (weights.q8Layers && reuse > state.kvBatch) {
    reuse = state.kvBatch
  }
  if (reuse > state.kvCapacity) {
    reuse = state.kvCapacity
  }
  if (reuse > numPromptTokens - 1) {
    reuse = numPromptTokens - 1
  }
  while (pos < reuse && kvTokens[pos] === promptTokens[pos]) {
    pos = pos + 1
  }
  state.kvCount = pos
  if (state.kvBatch > pos) {
    state.kvBatch = pos
  }
  token = promptTokens[pos]

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

  // Prefill is over - single-token generation doesn't touch the batch
  // buffers, so release ~2.5 MB for the generation phase.
  freeBatchBuffers(state)

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
        decoded = streamDecoder.decode(bytes, true)
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

  // A prompt longer than the context ran past the KV cache: nothing is trusted
  if (numPromptTokens - 1 > config.seqLen) {
    state.kvCount = 0
    state.kvBatch = 0
  }

  // Release the scratch of this call (logits included)
  state.logits = null
  state.logits64 = null
  genScratch = null

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
          console.log("The model parameter is required and must be an ArrayBuffer.")
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
    console.log(err)

    // a reply that fails is ended anyway, so the chat doesn't wait for it forever: a
    // space (in case it failed before sending any text) and then the end of the reply
    if (data.type === "generate") {
      postMessage({ type: "token", token: " " })
      postMessage({ type: "complete", output: " " })
    }
  }
}

// Web Worker mode
if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = function (e) {
    llama3pure(e.data)
  }
}

if (typeof module !== "undefined") {
  module.exports = llama3pure
}
