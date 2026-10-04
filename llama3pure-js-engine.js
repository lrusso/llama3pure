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

function createStreamingUTF8Decoder() {
  var pendingBytes = []
  return {
    decode: function decode(bytes, stream) {
      var i
      var j
      var b0
      var seqLen
      var cp
      var combined
      if (!bytes) {
        bytes = new Uint8Array(0)
      }
      if (pendingBytes.length > 0) {
        combined = new Uint8Array(pendingBytes.length + bytes.length)
        for (i = 0; i < pendingBytes.length; i++) {
          combined[i] = pendingBytes[i]
        }
        for (j = 0; j < bytes.length; j++) {
          combined[pendingBytes.length + j] = bytes[j]
        }
        pendingBytes = []
      } else {
        combined = bytes
      }
      var str = ""
      var len = combined.length
      i = 0
      while (i < len) {
        b0 = combined[i]
        if (b0 < 0x80) {
          seqLen = 1
        } else if ((b0 & 0xe0) === 0xc0) {
          seqLen = 2
        } else if ((b0 & 0xf0) === 0xe0) {
          seqLen = 3
        } else if ((b0 & 0xf8) === 0xf0) {
          seqLen = 4
        } else {
          i = i + 1
          continue
        }
        if (i + seqLen > len) {
          if (stream) {
            for (j = i; j < len; j++) {
              pendingBytes.push(combined[j])
            }
          }
          break
        }
        if (seqLen === 1) {
          str = str + String.fromCharCode(b0)
        } else if (seqLen === 2) {
          str =
            str + String.fromCharCode(((b0 & 0x1f) << 6) | (combined[i + 1] & 0x3f))
        } else if (seqLen === 3) {
          str =
            str +
            String.fromCharCode(
              ((b0 & 0x0f) << 12) |
                ((combined[i + 1] & 0x3f) << 6) |
                (combined[i + 2] & 0x3f)
            )
        } else {
          cp =
            ((b0 & 0x07) << 18) |
            ((combined[i + 1] & 0x3f) << 12) |
            ((combined[i + 2] & 0x3f) << 6) |
            (combined[i + 3] & 0x3f)
          cp = cp - 0x10000
          str = str + String.fromCharCode(0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff))
        }
        i = i + seqLen
      }
      return str
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
// Scratch of one generate() call: holds the tokenizer hash table while the
// prompt is encoded, then the prefill batch buffers, then the logits (they
// are never needed at the same time), and is released when the call ends
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

// QuantizedTensor: { dataOffset, type, rows, cols, ... }, read on the fly by
// the matmuls

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

// FP16 to FP32 lookup table (256 KB), filled on the first generate() call
// (ensureFp16Table); until then its pages are never touched
var fp16Table = new Float32Array(65536)
var fp16TableReady = false

function ensureFp16Table() {
  if (fp16TableReady) {
    return
  }
  fp16TableReady = true
  var t = fp16Table
  // Exponent e, mantissa m: (m + 1024) * 2^(e - 25), denormals (e = 0) m * 2^-24;
  // e = 31 is +-Infinity (m = 0) or NaN. The upper half of the table is the
  // negative sign bit.
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

// out[outOffset...] += weight * (Q8_0 cached vector at cacheOffset), `count`
// elements (multiple of 32)
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

// Row dequantizers for 4-byte-aligned F16 / BF16 / F32 matrices, same signature
// as the K-quant ones (view: Int32Array, two values per word, or Float32Array)
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
// Row dequantizers for the K-quant formats (matmuls and embedding rows).
// Arguments: the view the matrix is read through (Int32Array for
// Q2_K/Q4_K/Q5_K, Uint16Array for Q3_K/Q6_K), p = index of the row's first
// element in it, dst[dstOff...] gets cols values.

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

// 12 scale bytes of the Q3_K block at word p -> 16 signed 6-bit scales at
// q3kScales[base...] (same unpacking as llama.cpp).
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

// 6-bit scale/min pair number `is` (0-7) of a Q4_K/Q5_K block from its three
// scale words; returns sc in the low 8 bits and m in the high 8 bits.
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

// Dequantize `count` K-quant values at an absolute buffer offset through a
// temporary view (small tensors and non-embedding rows)
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

// Byte-wise dot product of one Q8_0 row, for the matrices the Int32 kernels
// cannot read (odd number of blocks per row)
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
// Block-32 formats with Q8_0 activations (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL),
// prefill: each row is unpacked once into int8 weights and per-block scales
// (and mins), then a 4-row x 3-token integer tile runs against the Q8_0
// activations; exact integer block sums, scale products in block order.

// Unpack one row into w8[wOff...] (int8 weights, 32 per block), sc[scOff + b]
// (fp16 scale as float) and, for the *_1 formats, mn[mnOff + b] (fp16 min).
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

// Integer 4-row x 3-token tile over one group of 4 unpacked rows (w8: int8
// weights; sc: per-block scales, mins and x block sums) for tokens
// bt..bt+nTok-1; a short group reuses its last token for the missing lanes.
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

// Scratch layout inside matmulDeqBuf (32 * maxCols bytes) for the integer tile:
//   bytes   [0, 4 * cols)                int8 weights of the 4 rows (matmulDeqI8)
//   doubles [cols/2, +4*nb)              per-row, per-block scales
//   doubles [cols/2 + 4*nb, +4*nb)       per-row, per-block mins (Q4_1 / Q5_1)
//   doubles [cols/2 + 8*nb, +batch*nb)   per-token, per-block sums of the Q8 x
function matmulBlock32Q8Batch(outs, xs, qw, batchSize) {
  var rows = qw.rows
  var cols = qw.cols
  var rowSize = qw.rowSize
  var base = qw.dataOffset
  var nb = cols >> 5
  var unpack = qw.unpackRowFunc
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
// Single-token kernels for the block-32 formats with Q8_0 activations: rows
// i..i+3 per call (row loop in matmulQuantizedPreQ8), weights read through the
// Uint16 view, the quantized x once per block for the 4 rows; same math as the
// prefill tile (the Q4_0/Q5_0 offset is folded out of the inner sum).

function matmulQ4_0Q8Rows4(out, qw, i) {
  var U16 = qw.localU16
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
    // 4 quant bytes per step: weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
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
  var U16 = qw.localU16
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
    // 4 quant bytes per step: weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
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
  var U16 = qw.localU16
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
    // 4 quant bytes per step: weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
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
  var U16 = qw.localU16
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
    // 4 quant bytes per step: weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
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
  var U16 = qw.localU16
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
    // 4 quant bytes per step: weights 4g..4g+3 (low nibbles), 4g+16..4g+19 (high)
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

// Single-token kernel that computes a group of rows per call: (out, qw, i)
// for the block-32 formats (4 rows, Q8_0 activations), (out, x, qw, i) for
// the fused ones (2 rows); null for Q8_0 and Q2_K.
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
  if (qw.localI32 !== null) {
    matmulQ8_0Local(out, x, qw)
  } else if (qw.deqRowFunc) {
    // Fused kernel when the type has one, else dequantize-then-dot
    if (qw.rowsFunc !== null) {
      matmulRows(out, x, qw)
    } else {
      matmulKQuantLocal(out, x, qw, 0)
    }
  } else if (qw.rowsFunc !== null) {
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
  var rowsFunc = qw.rowsFunc
  for (var i = 0; i < rows; i = i + 4) {
    rowsFunc(out, qw, i)
  }
}

// Batched matmul: several input vectors against the same weight matrix
var PREFILL_BATCH_SIZE = 32

function matmulQuantizedBatch(outs, xs, qw, batchSize) {
  var rows = qw.rows
  var cols = qw.cols
  var baseOffset = qw.dataOffset
  var rowSize = qw.rowSize

  if (qw.localI32 !== null) {
    matmulQ8_0LocalBatch(outs, xs, qw, batchSize)
  } else if (qw.deqRowFunc) {
    // K-quants and F16/BF16/F32: dequantize 4 rows at a time, 4-row x 3-token tile
    matmulKQuantLocalBatch(outs, xs, qw, batchSize)
  } else if (qw.unpackRowFunc) {
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

// Q8_0 matmul, single token: 4 rows at a time, 4 weights per Int32 load
// (sign-extracted with a shift pair). Rows are walked in pairs of 34-byte
// blocks (17 words: scale + weights 0-1, words 1-7 weights 2-29, word 8 weights
// 30-31 + the next scale, words 9-16 the second block). Same summation order as
// the byte kernel. The second 16-column group of a block sits in a
// one-iteration loop on purpose (keeps V8 from hoisting every load and
// spilling). Requires qw.localI32.
function matmulQ8_0Local(out, x, qw) {
  var I32 = qw.localI32
  var base = qw.base
  var rows = qw.rows
  var cols = qw.cols
  var rowWords = qw.rowSize >> 2
  var nbp = cols >> 6
  var rows4 = rows & ~3
  for (var i = 0; i < rows4; i = i + 4) {
    var p0 = base + i * rowWords
    var p1 = p0 + rowWords
    var p2 = p1 + rowWords
    var p3 = p2 + rowWords
    var s0 = 0.0
    var s1 = 0.0
    var s2 = 0.0
    var s3 = 0.0
    var xb = 0
    for (var b = 0; b < nbp; b = b + 1) {
      var d0_0 = fp16Table[I32[p0] & 0xffff]
      var d0_1 = fp16Table[I32[p1] & 0xffff]
      var d0_2 = fp16Table[I32[p2] & 0xffff]
      var d0_3 = fp16Table[I32[p3] & 0xffff]
      // Even block of the pair (weights 0-31), columns 0-15
      var xi = xb
      var w0 = p0
      var w1 = p1
      var w2 = p2
      var w3 = p3
      var x0 = x[xi]
      var x1 = x[(xi + 1) | 0]
      var x2 = x[(xi + 2) | 0]
      var x3 = x[(xi + 3) | 0]
      var x4 = x[(xi + 4) | 0]
      var x5 = x[(xi + 5) | 0]
      var x6 = x[(xi + 6) | 0]
      var x7 = x[(xi + 7) | 0]
      var x8 = x[(xi + 8) | 0]
      var x9 = x[(xi + 9) | 0]
      var x10 = x[(xi + 10) | 0]
      var x11 = x[(xi + 11) | 0]
      var x12 = x[(xi + 12) | 0]
      var x13 = x[(xi + 13) | 0]
      var x14 = x[(xi + 14) | 0]
      var x15 = x[(xi + 15) | 0]
      var v0_0 = I32[w0]
      var v0_1 = I32[(w0 + 1) | 0]
      var v0_2 = I32[(w0 + 2) | 0]
      var v0_3 = I32[(w0 + 3) | 0]
      var v0_4 = I32[(w0 + 4) | 0]
      var v1_0 = I32[w1]
      var v1_1 = I32[(w1 + 1) | 0]
      var v1_2 = I32[(w1 + 2) | 0]
      var v1_3 = I32[(w1 + 3) | 0]
      var v1_4 = I32[(w1 + 4) | 0]
      var v2_0 = I32[w2]
      var v2_1 = I32[(w2 + 1) | 0]
      var v2_2 = I32[(w2 + 2) | 0]
      var v2_3 = I32[(w2 + 3) | 0]
      var v2_4 = I32[(w2 + 4) | 0]
      var v3_0 = I32[w3]
      var v3_1 = I32[(w3 + 1) | 0]
      var v3_2 = I32[(w3 + 2) | 0]
      var v3_3 = I32[(w3 + 3) | 0]
      var v3_4 = I32[(w3 + 4) | 0]
      var a0 =
        x0 * ((v0_0 << 8) >> 24) +
        x1 * (v0_0 >> 24) +
        x2 * ((v0_1 << 24) >> 24) +
        x3 * ((v0_1 << 16) >> 24) +
        x4 * ((v0_1 << 8) >> 24) +
        x5 * (v0_1 >> 24) +
        x6 * ((v0_2 << 24) >> 24) +
        x7 * ((v0_2 << 16) >> 24) +
        x8 * ((v0_2 << 8) >> 24) +
        x9 * (v0_2 >> 24) +
        x10 * ((v0_3 << 24) >> 24) +
        x11 * ((v0_3 << 16) >> 24) +
        x12 * ((v0_3 << 8) >> 24) +
        x13 * (v0_3 >> 24) +
        x14 * ((v0_4 << 24) >> 24) +
        x15 * ((v0_4 << 16) >> 24)
      var a1 =
        x0 * ((v1_0 << 8) >> 24) +
        x1 * (v1_0 >> 24) +
        x2 * ((v1_1 << 24) >> 24) +
        x3 * ((v1_1 << 16) >> 24) +
        x4 * ((v1_1 << 8) >> 24) +
        x5 * (v1_1 >> 24) +
        x6 * ((v1_2 << 24) >> 24) +
        x7 * ((v1_2 << 16) >> 24) +
        x8 * ((v1_2 << 8) >> 24) +
        x9 * (v1_2 >> 24) +
        x10 * ((v1_3 << 24) >> 24) +
        x11 * ((v1_3 << 16) >> 24) +
        x12 * ((v1_3 << 8) >> 24) +
        x13 * (v1_3 >> 24) +
        x14 * ((v1_4 << 24) >> 24) +
        x15 * ((v1_4 << 16) >> 24)
      var a2 =
        x0 * ((v2_0 << 8) >> 24) +
        x1 * (v2_0 >> 24) +
        x2 * ((v2_1 << 24) >> 24) +
        x3 * ((v2_1 << 16) >> 24) +
        x4 * ((v2_1 << 8) >> 24) +
        x5 * (v2_1 >> 24) +
        x6 * ((v2_2 << 24) >> 24) +
        x7 * ((v2_2 << 16) >> 24) +
        x8 * ((v2_2 << 8) >> 24) +
        x9 * (v2_2 >> 24) +
        x10 * ((v2_3 << 24) >> 24) +
        x11 * ((v2_3 << 16) >> 24) +
        x12 * ((v2_3 << 8) >> 24) +
        x13 * (v2_3 >> 24) +
        x14 * ((v2_4 << 24) >> 24) +
        x15 * ((v2_4 << 16) >> 24)
      var a3 =
        x0 * ((v3_0 << 8) >> 24) +
        x1 * (v3_0 >> 24) +
        x2 * ((v3_1 << 24) >> 24) +
        x3 * ((v3_1 << 16) >> 24) +
        x4 * ((v3_1 << 8) >> 24) +
        x5 * (v3_1 >> 24) +
        x6 * ((v3_2 << 24) >> 24) +
        x7 * ((v3_2 << 16) >> 24) +
        x8 * ((v3_2 << 8) >> 24) +
        x9 * (v3_2 >> 24) +
        x10 * ((v3_3 << 24) >> 24) +
        x11 * ((v3_3 << 16) >> 24) +
        x12 * ((v3_3 << 8) >> 24) +
        x13 * (v3_3 >> 24) +
        x14 * ((v3_4 << 24) >> 24) +
        x15 * ((v3_4 << 16) >> 24)
      // Columns 16-31 (one-iteration loop: bounds the basic block for V8)
      for (var g = 1; g < 2; g = g + 1) {
        xi = xi + 16
        w0 = w0 + 4
        w1 = w1 + 4
        w2 = w2 + 4
        w3 = w3 + 4
        var x0 = x[xi]
        var x1 = x[(xi + 1) | 0]
        var x2 = x[(xi + 2) | 0]
        var x3 = x[(xi + 3) | 0]
        var x4 = x[(xi + 4) | 0]
        var x5 = x[(xi + 5) | 0]
        var x6 = x[(xi + 6) | 0]
        var x7 = x[(xi + 7) | 0]
        var x8 = x[(xi + 8) | 0]
        var x9 = x[(xi + 9) | 0]
        var x10 = x[(xi + 10) | 0]
        var x11 = x[(xi + 11) | 0]
        var x12 = x[(xi + 12) | 0]
        var x13 = x[(xi + 13) | 0]
        var x14 = x[(xi + 14) | 0]
        var x15 = x[(xi + 15) | 0]
        var v0_0 = I32[w0]
        var v0_1 = I32[(w0 + 1) | 0]
        var v0_2 = I32[(w0 + 2) | 0]
        var v0_3 = I32[(w0 + 3) | 0]
        var v0_4 = I32[(w0 + 4) | 0]
        var v1_0 = I32[w1]
        var v1_1 = I32[(w1 + 1) | 0]
        var v1_2 = I32[(w1 + 2) | 0]
        var v1_3 = I32[(w1 + 3) | 0]
        var v1_4 = I32[(w1 + 4) | 0]
        var v2_0 = I32[w2]
        var v2_1 = I32[(w2 + 1) | 0]
        var v2_2 = I32[(w2 + 2) | 0]
        var v2_3 = I32[(w2 + 3) | 0]
        var v2_4 = I32[(w2 + 4) | 0]
        var v3_0 = I32[w3]
        var v3_1 = I32[(w3 + 1) | 0]
        var v3_2 = I32[(w3 + 2) | 0]
        var v3_3 = I32[(w3 + 3) | 0]
        var v3_4 = I32[(w3 + 4) | 0]
        a0 =
          a0 +
          x0 * ((v0_0 << 8) >> 24) +
          x1 * (v0_0 >> 24) +
          x2 * ((v0_1 << 24) >> 24) +
          x3 * ((v0_1 << 16) >> 24) +
          x4 * ((v0_1 << 8) >> 24) +
          x5 * (v0_1 >> 24) +
          x6 * ((v0_2 << 24) >> 24) +
          x7 * ((v0_2 << 16) >> 24) +
          x8 * ((v0_2 << 8) >> 24) +
          x9 * (v0_2 >> 24) +
          x10 * ((v0_3 << 24) >> 24) +
          x11 * ((v0_3 << 16) >> 24) +
          x12 * ((v0_3 << 8) >> 24) +
          x13 * (v0_3 >> 24) +
          x14 * ((v0_4 << 24) >> 24) +
          x15 * ((v0_4 << 16) >> 24)
        a1 =
          a1 +
          x0 * ((v1_0 << 8) >> 24) +
          x1 * (v1_0 >> 24) +
          x2 * ((v1_1 << 24) >> 24) +
          x3 * ((v1_1 << 16) >> 24) +
          x4 * ((v1_1 << 8) >> 24) +
          x5 * (v1_1 >> 24) +
          x6 * ((v1_2 << 24) >> 24) +
          x7 * ((v1_2 << 16) >> 24) +
          x8 * ((v1_2 << 8) >> 24) +
          x9 * (v1_2 >> 24) +
          x10 * ((v1_3 << 24) >> 24) +
          x11 * ((v1_3 << 16) >> 24) +
          x12 * ((v1_3 << 8) >> 24) +
          x13 * (v1_3 >> 24) +
          x14 * ((v1_4 << 24) >> 24) +
          x15 * ((v1_4 << 16) >> 24)
        a2 =
          a2 +
          x0 * ((v2_0 << 8) >> 24) +
          x1 * (v2_0 >> 24) +
          x2 * ((v2_1 << 24) >> 24) +
          x3 * ((v2_1 << 16) >> 24) +
          x4 * ((v2_1 << 8) >> 24) +
          x5 * (v2_1 >> 24) +
          x6 * ((v2_2 << 24) >> 24) +
          x7 * ((v2_2 << 16) >> 24) +
          x8 * ((v2_2 << 8) >> 24) +
          x9 * (v2_2 >> 24) +
          x10 * ((v2_3 << 24) >> 24) +
          x11 * ((v2_3 << 16) >> 24) +
          x12 * ((v2_3 << 8) >> 24) +
          x13 * (v2_3 >> 24) +
          x14 * ((v2_4 << 24) >> 24) +
          x15 * ((v2_4 << 16) >> 24)
        a3 =
          a3 +
          x0 * ((v3_0 << 8) >> 24) +
          x1 * (v3_0 >> 24) +
          x2 * ((v3_1 << 24) >> 24) +
          x3 * ((v3_1 << 16) >> 24) +
          x4 * ((v3_1 << 8) >> 24) +
          x5 * (v3_1 >> 24) +
          x6 * ((v3_2 << 24) >> 24) +
          x7 * ((v3_2 << 16) >> 24) +
          x8 * ((v3_2 << 8) >> 24) +
          x9 * (v3_2 >> 24) +
          x10 * ((v3_3 << 24) >> 24) +
          x11 * ((v3_3 << 16) >> 24) +
          x12 * ((v3_3 << 8) >> 24) +
          x13 * (v3_3 >> 24) +
          x14 * ((v3_4 << 24) >> 24) +
          x15 * ((v3_4 << 16) >> 24)
      }
      s0 = s0 + d0_0 * a0
      s1 = s1 + d0_1 * a1
      s2 = s2 + d0_2 * a2
      s3 = s3 + d0_3 * a3
      // Odd block of the pair (weights 32-63): scale in the high half of word 8
      var d1_0 = fp16Table[I32[(p0 + 8) | 0] >>> 16]
      var d1_1 = fp16Table[I32[(p1 + 8) | 0] >>> 16]
      var d1_2 = fp16Table[I32[(p2 + 8) | 0] >>> 16]
      var d1_3 = fp16Table[I32[(p3 + 8) | 0] >>> 16]
      xi = xb + 32
      w0 = p0 + 9
      w1 = p1 + 9
      w2 = p2 + 9
      w3 = p3 + 9
      var x0 = x[xi]
      var x1 = x[(xi + 1) | 0]
      var x2 = x[(xi + 2) | 0]
      var x3 = x[(xi + 3) | 0]
      var x4 = x[(xi + 4) | 0]
      var x5 = x[(xi + 5) | 0]
      var x6 = x[(xi + 6) | 0]
      var x7 = x[(xi + 7) | 0]
      var x8 = x[(xi + 8) | 0]
      var x9 = x[(xi + 9) | 0]
      var x10 = x[(xi + 10) | 0]
      var x11 = x[(xi + 11) | 0]
      var x12 = x[(xi + 12) | 0]
      var x13 = x[(xi + 13) | 0]
      var x14 = x[(xi + 14) | 0]
      var x15 = x[(xi + 15) | 0]
      var v0_0 = I32[w0]
      var v0_1 = I32[(w0 + 1) | 0]
      var v0_2 = I32[(w0 + 2) | 0]
      var v0_3 = I32[(w0 + 3) | 0]
      var v1_0 = I32[w1]
      var v1_1 = I32[(w1 + 1) | 0]
      var v1_2 = I32[(w1 + 2) | 0]
      var v1_3 = I32[(w1 + 3) | 0]
      var v2_0 = I32[w2]
      var v2_1 = I32[(w2 + 1) | 0]
      var v2_2 = I32[(w2 + 2) | 0]
      var v2_3 = I32[(w2 + 3) | 0]
      var v3_0 = I32[w3]
      var v3_1 = I32[(w3 + 1) | 0]
      var v3_2 = I32[(w3 + 2) | 0]
      var v3_3 = I32[(w3 + 3) | 0]
      a0 =
        x0 * ((v0_0 << 24) >> 24) +
        x1 * ((v0_0 << 16) >> 24) +
        x2 * ((v0_0 << 8) >> 24) +
        x3 * (v0_0 >> 24) +
        x4 * ((v0_1 << 24) >> 24) +
        x5 * ((v0_1 << 16) >> 24) +
        x6 * ((v0_1 << 8) >> 24) +
        x7 * (v0_1 >> 24) +
        x8 * ((v0_2 << 24) >> 24) +
        x9 * ((v0_2 << 16) >> 24) +
        x10 * ((v0_2 << 8) >> 24) +
        x11 * (v0_2 >> 24) +
        x12 * ((v0_3 << 24) >> 24) +
        x13 * ((v0_3 << 16) >> 24) +
        x14 * ((v0_3 << 8) >> 24) +
        x15 * (v0_3 >> 24)
      a1 =
        x0 * ((v1_0 << 24) >> 24) +
        x1 * ((v1_0 << 16) >> 24) +
        x2 * ((v1_0 << 8) >> 24) +
        x3 * (v1_0 >> 24) +
        x4 * ((v1_1 << 24) >> 24) +
        x5 * ((v1_1 << 16) >> 24) +
        x6 * ((v1_1 << 8) >> 24) +
        x7 * (v1_1 >> 24) +
        x8 * ((v1_2 << 24) >> 24) +
        x9 * ((v1_2 << 16) >> 24) +
        x10 * ((v1_2 << 8) >> 24) +
        x11 * (v1_2 >> 24) +
        x12 * ((v1_3 << 24) >> 24) +
        x13 * ((v1_3 << 16) >> 24) +
        x14 * ((v1_3 << 8) >> 24) +
        x15 * (v1_3 >> 24)
      a2 =
        x0 * ((v2_0 << 24) >> 24) +
        x1 * ((v2_0 << 16) >> 24) +
        x2 * ((v2_0 << 8) >> 24) +
        x3 * (v2_0 >> 24) +
        x4 * ((v2_1 << 24) >> 24) +
        x5 * ((v2_1 << 16) >> 24) +
        x6 * ((v2_1 << 8) >> 24) +
        x7 * (v2_1 >> 24) +
        x8 * ((v2_2 << 24) >> 24) +
        x9 * ((v2_2 << 16) >> 24) +
        x10 * ((v2_2 << 8) >> 24) +
        x11 * (v2_2 >> 24) +
        x12 * ((v2_3 << 24) >> 24) +
        x13 * ((v2_3 << 16) >> 24) +
        x14 * ((v2_3 << 8) >> 24) +
        x15 * (v2_3 >> 24)
      a3 =
        x0 * ((v3_0 << 24) >> 24) +
        x1 * ((v3_0 << 16) >> 24) +
        x2 * ((v3_0 << 8) >> 24) +
        x3 * (v3_0 >> 24) +
        x4 * ((v3_1 << 24) >> 24) +
        x5 * ((v3_1 << 16) >> 24) +
        x6 * ((v3_1 << 8) >> 24) +
        x7 * (v3_1 >> 24) +
        x8 * ((v3_2 << 24) >> 24) +
        x9 * ((v3_2 << 16) >> 24) +
        x10 * ((v3_2 << 8) >> 24) +
        x11 * (v3_2 >> 24) +
        x12 * ((v3_3 << 24) >> 24) +
        x13 * ((v3_3 << 16) >> 24) +
        x14 * ((v3_3 << 8) >> 24) +
        x15 * (v3_3 >> 24)
      for (var g = 1; g < 2; g = g + 1) {
        xi = xi + 16
        w0 = w0 + 4
        w1 = w1 + 4
        w2 = w2 + 4
        w3 = w3 + 4
        var x0 = x[xi]
        var x1 = x[(xi + 1) | 0]
        var x2 = x[(xi + 2) | 0]
        var x3 = x[(xi + 3) | 0]
        var x4 = x[(xi + 4) | 0]
        var x5 = x[(xi + 5) | 0]
        var x6 = x[(xi + 6) | 0]
        var x7 = x[(xi + 7) | 0]
        var x8 = x[(xi + 8) | 0]
        var x9 = x[(xi + 9) | 0]
        var x10 = x[(xi + 10) | 0]
        var x11 = x[(xi + 11) | 0]
        var x12 = x[(xi + 12) | 0]
        var x13 = x[(xi + 13) | 0]
        var x14 = x[(xi + 14) | 0]
        var x15 = x[(xi + 15) | 0]
        var v0_0 = I32[w0]
        var v0_1 = I32[(w0 + 1) | 0]
        var v0_2 = I32[(w0 + 2) | 0]
        var v0_3 = I32[(w0 + 3) | 0]
        var v1_0 = I32[w1]
        var v1_1 = I32[(w1 + 1) | 0]
        var v1_2 = I32[(w1 + 2) | 0]
        var v1_3 = I32[(w1 + 3) | 0]
        var v2_0 = I32[w2]
        var v2_1 = I32[(w2 + 1) | 0]
        var v2_2 = I32[(w2 + 2) | 0]
        var v2_3 = I32[(w2 + 3) | 0]
        var v3_0 = I32[w3]
        var v3_1 = I32[(w3 + 1) | 0]
        var v3_2 = I32[(w3 + 2) | 0]
        var v3_3 = I32[(w3 + 3) | 0]
        a0 =
          a0 +
          x0 * ((v0_0 << 24) >> 24) +
          x1 * ((v0_0 << 16) >> 24) +
          x2 * ((v0_0 << 8) >> 24) +
          x3 * (v0_0 >> 24) +
          x4 * ((v0_1 << 24) >> 24) +
          x5 * ((v0_1 << 16) >> 24) +
          x6 * ((v0_1 << 8) >> 24) +
          x7 * (v0_1 >> 24) +
          x8 * ((v0_2 << 24) >> 24) +
          x9 * ((v0_2 << 16) >> 24) +
          x10 * ((v0_2 << 8) >> 24) +
          x11 * (v0_2 >> 24) +
          x12 * ((v0_3 << 24) >> 24) +
          x13 * ((v0_3 << 16) >> 24) +
          x14 * ((v0_3 << 8) >> 24) +
          x15 * (v0_3 >> 24)
        a1 =
          a1 +
          x0 * ((v1_0 << 24) >> 24) +
          x1 * ((v1_0 << 16) >> 24) +
          x2 * ((v1_0 << 8) >> 24) +
          x3 * (v1_0 >> 24) +
          x4 * ((v1_1 << 24) >> 24) +
          x5 * ((v1_1 << 16) >> 24) +
          x6 * ((v1_1 << 8) >> 24) +
          x7 * (v1_1 >> 24) +
          x8 * ((v1_2 << 24) >> 24) +
          x9 * ((v1_2 << 16) >> 24) +
          x10 * ((v1_2 << 8) >> 24) +
          x11 * (v1_2 >> 24) +
          x12 * ((v1_3 << 24) >> 24) +
          x13 * ((v1_3 << 16) >> 24) +
          x14 * ((v1_3 << 8) >> 24) +
          x15 * (v1_3 >> 24)
        a2 =
          a2 +
          x0 * ((v2_0 << 24) >> 24) +
          x1 * ((v2_0 << 16) >> 24) +
          x2 * ((v2_0 << 8) >> 24) +
          x3 * (v2_0 >> 24) +
          x4 * ((v2_1 << 24) >> 24) +
          x5 * ((v2_1 << 16) >> 24) +
          x6 * ((v2_1 << 8) >> 24) +
          x7 * (v2_1 >> 24) +
          x8 * ((v2_2 << 24) >> 24) +
          x9 * ((v2_2 << 16) >> 24) +
          x10 * ((v2_2 << 8) >> 24) +
          x11 * (v2_2 >> 24) +
          x12 * ((v2_3 << 24) >> 24) +
          x13 * ((v2_3 << 16) >> 24) +
          x14 * ((v2_3 << 8) >> 24) +
          x15 * (v2_3 >> 24)
        a3 =
          a3 +
          x0 * ((v3_0 << 24) >> 24) +
          x1 * ((v3_0 << 16) >> 24) +
          x2 * ((v3_0 << 8) >> 24) +
          x3 * (v3_0 >> 24) +
          x4 * ((v3_1 << 24) >> 24) +
          x5 * ((v3_1 << 16) >> 24) +
          x6 * ((v3_1 << 8) >> 24) +
          x7 * (v3_1 >> 24) +
          x8 * ((v3_2 << 24) >> 24) +
          x9 * ((v3_2 << 16) >> 24) +
          x10 * ((v3_2 << 8) >> 24) +
          x11 * (v3_2 >> 24) +
          x12 * ((v3_3 << 24) >> 24) +
          x13 * ((v3_3 << 16) >> 24) +
          x14 * ((v3_3 << 8) >> 24) +
          x15 * (v3_3 >> 24)
      }
      s0 = s0 + d1_0 * a0
      s1 = s1 + d1_1 * a1
      s2 = s2 + d1_2 * a2
      s3 = s3 + d1_3 * a3
      p0 = p0 + 17
      p1 = p1 + 17
      p2 = p2 + 17
      p3 = p3 + 17
      xb = xb + 64
    }
    out[i] = s0
    out[i + 1] = s1
    out[i + 2] = s2
    out[i + 3] = s3
  }
  for (var i = rows4; i < rows; i = i + 1) {
    out[i] = dotRowQ8_0I32(x, I32, base + i * rowWords, nbp)
  }
}

// One row of a Q8_0 matrix against x, same layout and summation order as
// matmulQ8_0Local (row remainder)
function dotRowQ8_0I32(x, I32, p, nbp) {
  var s = 0.0
  var xb = 0
  for (var b = 0; b < nbp; b = b + 1) {
    var v0 = I32[p]
    var d0 = fp16Table[v0 & 0xffff]
    var a = x[xb] * ((v0 << 8) >> 24) + x[xb + 1] * (v0 >> 24)
    for (var k = 1; k < 8; k = k + 1) {
      var v = I32[p + k]
      var j = xb + 4 * k - 2
      a =
        a +
        x[j] * ((v << 24) >> 24) +
        x[j + 1] * ((v << 16) >> 24) +
        x[j + 2] * ((v << 8) >> 24) +
        x[j + 3] * (v >> 24)
    }
    var v8 = I32[p + 8]
    a = a + x[xb + 30] * ((v8 << 24) >> 24) + x[xb + 31] * ((v8 << 16) >> 24)
    s = s + d0 * a
    var d1 = fp16Table[v8 >>> 16]
    var v9 = I32[p + 9]
    a =
      x[xb + 32] * ((v9 << 24) >> 24) +
      x[xb + 33] * ((v9 << 16) >> 24) +
      x[xb + 34] * ((v9 << 8) >> 24) +
      x[xb + 35] * (v9 >> 24)
    for (var k = 10; k < 17; k = k + 1) {
      var v = I32[p + k]
      var j = xb + 4 * k - 4
      a =
        a +
        x[j] * ((v << 24) >> 24) +
        x[j + 1] * ((v << 16) >> 24) +
        x[j + 2] * ((v << 8) >> 24) +
        x[j + 3] * (v >> 24)
    }
    s = s + d1 * a
    p = p + 17
    xb = xb + 64
  }
  return s
}

// Dequantize one Q8_0 row (even number of blocks) into dst[dstOff...] through
// the Int32 block-pair layout, bit-identical to the byte-wise dequantizer
function deqRowQ8_0I32(I32, p, dst, dstOff, nbp) {
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

// Q8_0 batch matmul (prefill): dequantize 4 rows into the row views of
// matmulDeqBuf, then a 4-row x 3-token tile, one column per step (12
// accumulators, the largest shape V8 keeps in registers); same column order as
// the byte kernel.
function matmulQ8_0LocalBatch(outs, xs, qw, batchSize) {
  var I32 = qw.localI32
  var base = qw.base
  var rows = qw.rows
  var cols = qw.cols
  var rowWords = qw.rowSize >> 2
  var nbp = cols >> 6
  var rows4 = rows & ~3
  var batch3 = batchSize - (batchSize % 3)
  var buf0 = matmulDeqRows[0]
  var buf1 = matmulDeqRows[1]
  var buf2 = matmulDeqRows[2]
  var buf3 = matmulDeqRows[3]
  for (var i = 0; i < rows4; i = i + 4) {
    var p = base + i * rowWords
    deqRowQ8_0I32(I32, p, buf0, 0, nbp)
    deqRowQ8_0I32(I32, p + rowWords, buf1, 0, nbp)
    deqRowQ8_0I32(I32, p + rowWords + rowWords, buf2, 0, nbp)
    deqRowQ8_0I32(I32, p + rowWords + rowWords + rowWords, buf3, 0, nbp)
    for (var bt = 0; bt < batch3; bt = bt + 3) {
      var xA = xs[bt]
      var xB = xs[bt + 1]
      var xC = xs[bt + 2]
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
      var oA = outs[bt]
      var oB = outs[bt + 1]
      var oC = outs[bt + 2]
      oA[i] = s0
      oA[i + 1] = s1
      oA[i + 2] = s2
      oA[i + 3] = s3
      oB[i] = t0
      oB[i + 1] = t1
      oB[i + 2] = t2
      oB[i + 3] = t3
      oC[i] = u0
      oC[i + 1] = u1
      oC[i + 2] = u2
      oC[i + 3] = u3
    }
    // Remaining 2 tokens of the batch: 4-row x 2-token tile
    if (batchSize - batch3 === 2) {
      var xA = xs[batch3]
      var xB = xs[batch3 + 1]
      var s0 = 0.0
      var s1 = 0.0
      var s2 = 0.0
      var s3 = 0.0
      var t0 = 0.0
      var t1 = 0.0
      var t2 = 0.0
      var t3 = 0.0
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
      var oA = outs[batch3]
      var oB = outs[batch3 + 1]
      oA[i] = s0
      oA[i + 1] = s1
      oA[i + 2] = s2
      oA[i + 3] = s3
      oB[i] = t0
      oB[i + 1] = t1
      oB[i + 2] = t2
      oB[i + 3] = t3
    } else if (batchSize - batch3 === 1) {
      // Remaining single token
      var xArr = xs[batch3]
      var s0 = 0.0
      var s1 = 0.0
      var s2 = 0.0
      var s3 = 0.0
      for (var j = 0; j < cols; j = j + 1) {
        var a = xArr[j]
        s0 = s0 + a * buf0[j]
        s1 = s1 + a * buf1[j]
        s2 = s2 + a * buf2[j]
        s3 = s3 + a * buf3[j]
      }
      var oArr = outs[batch3]
      oArr[i] = s0
      oArr[i + 1] = s1
      oArr[i + 2] = s2
      oArr[i + 3] = s3
    }
  }
  // Remaining 1-3 rows
  for (var i = rows4; i < rows; i = i + 1) {
    deqRowQ8_0I32(I32, base + i * rowWords, buf0, 0, nbp)
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
// Fused single-token kernels: rows i .. i + 1 of a matrix against x, straight
// from the matrix view. Each weight is computed as in its row dequantizer and
// each row is summed left to right, so the results are bit-identical. Words are
// shifted with >> (not >>>) so they stay int32 (JavaScriptCore keeps the loop
// in integer registers).

// F16 / BF16: two weights per Int32 load, converted through the 64K-entry
// lookup table.
function matmulF16Rows(out, x, qw, i) {
  var I32 = qw.deqView
  var half = qw.cols >> 1
  var tab = qw.type === GGML_TYPE.F16 ? fp16Table : bf16Table
  var o0 = (qw.base + i * half) | 0
  var o1 = (o0 + half) | 0
  var s0 = 0.0
  var s1 = 0.0
  for (var k = 0; k < half; k = k + 1) {
    var a = x[k << 1]
    var c = x[((k << 1) + 1) | 0]
    var v = I32[(o0 + k) | 0]
    s0 = s0 + a * tab[v & 0xffff]
    s0 = s0 + c * tab[v >>> 16]
    v = I32[(o1 + k) | 0]
    s1 = s1 + a * tab[v & 0xffff]
    s1 = s1 + c * tab[v >>> 16]
  }
  out[i] = s0
  out[i + 1] = s1
}

// F32: through the Float32 view.
function matmulF32Rows(out, x, qw, i) {
  var F32 = qw.deqView
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

// Q4_K: 8 groups of 32 columns per block. Group c reads the 32 quant bytes
// of pair c >> 1 (low nibbles when c is even, high nibbles when odd) with
// scale/min number c.
function matmulQ4_KRows(out, x, qw, i) {
  var I32 = qw.deqView
  var nb = qw.cols >> 8
  var rowWords = qw.rowSize >> 2
  var p0 = (qw.base + i * rowWords) | 0
  var p1 = (p0 + rowWords) | 0
  var s0 = 0.0
  var s1 = 0.0
  var xb = 0
  for (var b = 0; b < nb; b = b + 1) {
    var hA = I32[p0]
    var dA = fp16Table[hA & 0xffff]
    var nA = fp16Table[hA >>> 16]
    var uA = I32[(p0 + 1) | 0]
    var vA = I32[(p0 + 2) | 0]
    var wA = I32[(p0 + 3) | 0]
    var hB = I32[p1]
    var dB = fp16Table[hB & 0xffff]
    var nB = fp16Table[hB >>> 16]
    var uB = I32[(p1 + 1) | 0]
    var vB = I32[(p1 + 2) | 0]
    var wB = I32[(p1 + 3) | 0]
    for (var c = 0; c < 8; c = c + 1) {
      var sm
      sm = kScaleMin(c, uA, vA, wA)
      var kA = dA * (sm & 0xff)
      var mA = nA * (sm >>> 8)
      sm = kScaleMin(c, uB, vB, wB)
      var kB = dB * (sm & 0xff)
      var mB = nB * (sm >>> 8)
      var sh = (c & 1) << 2
      var qo = (4 + ((c >> 1) << 3)) | 0
      for (var k = 0; k < 8; k = k + 1) {
        var j = (xb + (k << 2)) | 0
        var x0 = x[j]
        var x1 = x[(j + 1) | 0]
        var x2 = x[(j + 2) | 0]
        var x3 = x[(j + 3) | 0]
        var q
        q = I32[(p0 + qo + k) | 0] >> sh
        s0 = s0 + x0 * (kA * (q & 0xf) - mA)
        s0 = s0 + x1 * (kA * ((q >>> 8) & 0xf) - mA)
        s0 = s0 + x2 * (kA * ((q >>> 16) & 0xf) - mA)
        s0 = s0 + x3 * (kA * ((q >>> 24) & 0xf) - mA)
        q = I32[(p1 + qo + k) | 0] >> sh
        s1 = s1 + x0 * (kB * (q & 0xf) - mB)
        s1 = s1 + x1 * (kB * ((q >>> 8) & 0xf) - mB)
        s1 = s1 + x2 * (kB * ((q >>> 16) & 0xf) - mB)
        s1 = s1 + x3 * (kB * ((q >>> 24) & 0xf) - mB)
      }
      xb = xb + 32
    }
    p0 = p0 + 36
    p1 = p1 + 36
  }
  out[i] = s0
  out[i + 1] = s1
}

// Q5_K: like Q4_K plus the fifth bit of every quant (bit c of the 32 high-bit
// bytes), merged into the four quants of a word before they are extracted.
function matmulQ5_KRows(out, x, qw, i) {
  var I32 = qw.deqView
  var nb = qw.cols >> 8
  var rowWords = qw.rowSize >> 2
  var p0 = (qw.base + i * rowWords) | 0
  var p1 = (p0 + rowWords) | 0
  var s0 = 0.0
  var s1 = 0.0
  var xb = 0
  for (var b = 0; b < nb; b = b + 1) {
    var hA = I32[p0]
    var dA = fp16Table[hA & 0xffff]
    var nA = fp16Table[hA >>> 16]
    var uA = I32[(p0 + 1) | 0]
    var vA = I32[(p0 + 2) | 0]
    var wA = I32[(p0 + 3) | 0]
    var hB = I32[p1]
    var dB = fp16Table[hB & 0xffff]
    var nB = fp16Table[hB >>> 16]
    var uB = I32[(p1 + 1) | 0]
    var vB = I32[(p1 + 2) | 0]
    var wB = I32[(p1 + 3) | 0]
    for (var c = 0; c < 8; c = c + 1) {
      var sm
      sm = kScaleMin(c, uA, vA, wA)
      var kA = dA * (sm & 0xff)
      var mA = nA * (sm >>> 8)
      sm = kScaleMin(c, uB, vB, wB)
      var kB = dB * (sm & 0xff)
      var mB = nB * (sm >>> 8)
      var sh = (c & 1) << 2
      var qo = (12 + ((c >> 1) << 3)) | 0
      for (var k = 0; k < 8; k = k + 1) {
        var j = (xb + (k << 2)) | 0
        var x0 = x[j]
        var x1 = x[(j + 1) | 0]
        var x2 = x[(j + 2) | 0]
        var x3 = x[(j + 3) | 0]
        var q
        q =
          ((I32[(p0 + qo + k) | 0] >> sh) & 0x0f0f0f0f) |
          (((I32[(p0 + 4 + k) | 0] >> c) & 0x01010101) << 4)
        s0 = s0 + x0 * (kA * (q & 0xff) - mA)
        s0 = s0 + x1 * (kA * ((q >>> 8) & 0xff) - mA)
        s0 = s0 + x2 * (kA * ((q >>> 16) & 0xff) - mA)
        s0 = s0 + x3 * (kA * (q >>> 24) - mA)
        q =
          ((I32[(p1 + qo + k) | 0] >> sh) & 0x0f0f0f0f) |
          (((I32[(p1 + 4 + k) | 0] >> c) & 0x01010101) << 4)
        s1 = s1 + x0 * (kB * (q & 0xff) - mB)
        s1 = s1 + x1 * (kB * ((q >>> 8) & 0xff) - mB)
        s1 = s1 + x2 * (kB * ((q >>> 16) & 0xff) - mB)
        s1 = s1 + x3 * (kB * (q >>> 24) - mB)
      }
      xb = xb + 32
    }
    p0 = p0 + 44
    p1 = p1 + 44
  }
  out[i] = s0
  out[i + 1] = s1
}

// Q3_K / Q6_K: 16 groups of 16 columns per block (group t: half h = t >> 3,
// position g = (t >> 1) & 3, scale number t), 4 quants per DataView load (the
// blocks are only 2-byte aligned): low bits merged with high bits (Q6_K: ql
// nibbles + 2 qh bits; Q3_K: 2 qs bits + 1 mask bit).
function matmulK16Rows(out, x, qw, i) {
  var U16 = qw.deqView
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
  // r: 16-bit word index of each row's block inside the matrix (what the
  // DataView covers); vb: matrix start in the Uint16 view
  var vb = qw.base
  var r0 = i * rowWords
  var r1 = r0 + rowWords
  var s0 = 0.0
  var s1 = 0.0
  var xb = 0
  for (var b = 0; b < nb; b = b + 1) {
    var dA = fp16Table[U16[(vb + r0 + blockWords - 1) | 0]]
    var dB = fp16Table[U16[(vb + r1 + blockWords - 1) | 0]]
    if (q6) {
      // 16 int8 scales in the 8 words before d
      for (var t = 0; t < 8; t = t + 1) {
        var w
        w = U16[(vb + r0 + 96 + t) | 0]
        sc[(0 << 4) + (t << 1)] = w
        sc[(0 << 4) + (t << 1) + 1] = w >> 8
        w = U16[(vb + r1 + 96 + t) | 0]
        sc[(1 << 4) + (t << 1)] = w
        sc[(1 << 4) + (t << 1) + 1] = w >> 8
      }
    } else {
      q3kUnpackScales(U16, (vb + r0) | 0, 0 << 4)
      q3kUnpackScales(U16, (vb + r1) | 0, 1 << 4)
    }
    for (var t = 0; t < 16; t = t + 1) {
      var eA = dA * (sc[(0 << 4) + t] - scBias)
      var eB = dB * (sc[(1 << 4) + t] - scBias)
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
      for (var k = 0; k < 8; k = k + 2) {
        var j = (xb + (k << 1)) | 0
        var x0 = x[j]
        var x1 = x[(j + 1) | 0]
        var x2 = x[(j + 2) | 0]
        var x3 = x[(j + 3) | 0]
        var q
        q =
          ((DV.getInt32(((r0 + lo + k) << 1) | 0, true) >> loShift) & loMask) |
          (((DV.getInt32(((r0 + ho + k) << 1) | 0, true) >> hiShift) & hiMask) <<
            hiUp)
        s0 = s0 + x0 * (eA * ((q & 0xff) - bias))
        s0 = s0 + x1 * (eA * (((q >>> 8) & 0xff) - bias))
        s0 = s0 + x2 * (eA * (((q >>> 16) & 0xff) - bias))
        s0 = s0 + x3 * (eA * ((q >>> 24) - bias))
        q =
          ((DV.getInt32(((r1 + lo + k) << 1) | 0, true) >> loShift) & loMask) |
          (((DV.getInt32(((r1 + ho + k) << 1) | 0, true) >> hiShift) & hiMask) <<
            hiUp)
        s1 = s1 + x0 * (eB * ((q & 0xff) - bias))
        s1 = s1 + x1 * (eB * (((q >>> 8) & 0xff) - bias))
        s1 = s1 + x2 * (eB * (((q >>> 16) & 0xff) - bias))
        s1 = s1 + x3 * (eB * ((q >>> 24) - bias))
      }
      xb = xb + 16
    }
    r0 = r0 + blockWords
    r1 = r1 + blockWords
  }
  out[i] = s0
  out[i + 1] = s1
}

// DataView of the matrix being multiplied (Q3_K / Q6_K, see matmulRows)
var rowsDataView = null

// Single-token matmul through a fused kernel, 2 rows per call; the rows left
// over go through the generic kernel below.
function matmulRows(out, x, qw) {
  var rowsFunc = qw.rowsFunc
  var rowsN = qw.rows & ~1
  if (qw.deqView.BYTES_PER_ELEMENT === 2) {
    // Q3_K / Q6_K kernels read 4 bytes at a time through a DataView that only
    // lives during the call
    rowsDataView = new DataView(ggufData, qw.dataOffset, qw.rows * qw.rowSize)
  }
  for (var i = 0; i < rowsN; i = i + 2) {
    rowsFunc(out, x, qw, i)
  }
  rowsDataView = null
  if (rowsN < qw.rows) {
    matmulKQuantLocal(out, x, qw, rowsN)
  }
}

// K-quant matmul: dequantize 4 rows into a scratch buffer, then flat dot
// product, from startRow on. Scratch: matmulDeqBuf during prefill, else an idle
// buffer (logits while running the layers, hb while computing the logits),
// fewer rows per pass if it is small.
function matmulKQuantLocal(out, x, qw, startRow) {
  var rows = qw.rows
  var cols = qw.cols
  var rowSize = qw.rowSize
  var deqFunc = qw.deqRowFunc
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
  var view = qw.deqView
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

// K-quant / F16 / BF16 / F32 batch matmul (prefill): dequantize 4 rows into the
// row views of matmulDeqBuf, then a 4-row x 3-token tile per group of tokens (a
// short last group reuses its last token for the missing lanes); every sum runs
// over the columns left to right.
function kQuantTile(outs, xs, bt, nTok, i, cols) {
  var buf0 = matmulDeqRows[0]
  var buf1 = matmulDeqRows[1]
  var buf2 = matmulDeqRows[2]
  var buf3 = matmulDeqRows[3]
  var xA = xs[bt]
  var xB = xs[nTok > 1 ? bt + 1 : bt]
  var xC = xs[nTok > 2 ? bt + 2 : bt]
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
  if (nTok > 2) {
    var oC = outs[bt + 2]
    oC[i] = u0
    oC[i + 1] = u1
    oC[i + 2] = u2
    oC[i + 3] = u3
  }
}

function matmulKQuantLocalBatch(outs, xs, qw, batchSize) {
  var rows = qw.rows
  var cols = qw.cols
  var rowSize = qw.rowSize
  var deqFunc = qw.deqRowFunc
  var rows4 = rows & ~3
  var buf0 = matmulDeqRows[0]
  var buf1 = matmulDeqRows[1]
  var buf2 = matmulDeqRows[2]
  var buf3 = matmulDeqRows[3]
  var view = qw.deqView
  // Row length and position in elements of the view
  var rowEl = rowSize / view.BYTES_PER_ELEMENT
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
      kQuantTile(outs, xs, bt, nTok, i, cols)
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

// Advance offset past n GGUF strings without decoding them and return the
// longest one; 4 strings per iteration keeps JavaScriptCore from bringing up
// its top compiler tier during the load.
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

// The token strings of the vocabulary stay in the file: this only notes where
// the array starts and its longest string (the per-token tables are built by
// ensureVocabTables on the first use)
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
    // Where the token strings start in the file and the longest one; the
    // per-token tables and the end-of-turn token come from ensureVocabTables()
    // on the first generate() call
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
  w.hasKQuant = false
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

  // View a matrix is read through (kind 0: Int32Array, 1: Uint16Array, 2:
  // Float32Array): one shared view per kind over the whole file, the matrix
  // position in elements in `base`; files too big for int32 indices get a view
  // per matrix.
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

  // Kernels of the tensor types, looked up once per type
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
        deqRowFunc: getDeqRowFunc(type),
        unpackRowFunc: getUnpackRowFunc(type),
        rowsFunc: getRowsFunc(type),
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
      deqRowFunc: kernels.deqRowFunc,
      unpackRowFunc: kernels.unpackRowFunc,
      rowsFunc: kernels.rowsFunc,
      // Views the matrix is read through (Int32: Q8_0 kernels, Uint16: block-32
      // kernels, deqView: row dequantizers) and its position in them
      localI32: null,
      localU16: null,
      deqView: null,
      base: 0,
    }
    if (type === GGML_TYPE.Q8_0) {
      // Rows with an odd number of blocks keep the byte-wise kernel
      if (((off | rs) & 3) === 0) {
        result.localI32 = matrixView(result, off, totalBytes, 0)
      }
      return result
    }
    // The other formats are read through 16/32-bit words, the block-32 ones
    // 4 rows at a time. GGUF aligns tensor data to 32 bytes, so this only
    // rejects a corrupt file or a shape no model has.
    var block32 = result.unpackRowFunc !== null
    var halfFloat = type === GGML_TYPE.F16 || type === GGML_TYPE.BF16 || type === 30
    if (
      (off & 3) !== 0 ||
      (block32 ? (rows & 3) !== 0 : result.deqRowFunc === null) ||
      (halfFloat && (cols & 1) !== 0)
    ) {
      throw new Error("Unsupported tensor layout: " + name)
    }
    if (block32) {
      result.localU16 = matrixView(result, off, totalBytes, 1)
    } else {
      w.hasKQuant = true
      var kind = 0
      if (type === GGML_TYPE.Q3_K || type === GGML_TYPE.Q6_K) {
        kind = 1
      } else if (type === GGML_TYPE.F32) {
        kind = 2
      }
      result.deqView = matrixView(result, off, totalBytes, kind)
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
    if (result !== null && result.localI32 !== null) {
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

  // Tensors of one layer (kept out of the loop below so the loop stays small)
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

function createRunState(p) {
  var headSize = p.headDim
  var kvDim = p.nKvHeads * headSize
  var qDim = p.nHeads * headSize
  var maxDim = Math.max(p.dim, qDim)

  // RoPE frequencies (ropeSize floats each); cos/sin are filled per forward
  // pass by fillRopeBuffers()
  var ropeSize = headSize / 2
  var ropeFreqs = ropeFrequencies(p.ropeTheta, headSize)
  var ropeSwaFreqs = ropeFreqs
  if (p.isGemma) {
    ropeSwaFreqs = ropeFrequencies(
      p.ropeThetaSwa > 0 ? p.ropeThetaSwa : 10000.0,
      headSize
    )
  }

  // Q8_0 KV cache: 34 bytes per 32 floats, layout
  // [layer][kv_head][position][head_data]. It starts empty and grows on demand
  // (ensureKvCapacity) up to seqLen.
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

  // Bytes of the generate() scratch: the largest of the hash table
  // (buildVocabHash), the prefill batch buffers (ensureBatchBuffers) and
  // the logits
  var hashBuckets = 256
  while (hashBuckets * 4 < p.vocabSize) {
    hashBuckets = hashBuckets << 1
  }
  var batchTokenBytes =
    (p.dim + maxDim + p.dim + qDim + kvDim + kvDim + p.hiddenDim + p.hiddenDim) * 4 +
    ((xQ8Size + 7) & ~7)
  var scratchBytes = Math.max(
    (hashBuckets + p.vocabSize) * 4,
    maxCols * 32 + PREFILL_BATCH_SIZE * batchTokenBytes,
    p.vocabSize * 4
  )

  // Q8_0 buffer for quantized Q heads (for Q8 attention scoring)
  var qQ8TotalBytes = p.nHeads * headBytesQ8
  var qQ8Buffer = new ArrayBuffer(qQ8TotalBytes)

  // Prefill batch buffers: allocated on first prefill (ensureBatchBuffers),
  // these arrays are the placeholders
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

  var hbBuf = new Float32Array(p.hiddenDim)

  var runState = {
    x: new Float32Array(p.dim),
    xb: new Float32Array(maxDim),
    xb2: new Float32Array(p.dim),
    hb: hbBuf,
    hb2: new Float32Array(p.hiddenDim),
    q: new Float32Array(qDim),
    k: new Float32Array(kvDim),
    v: new Float32Array(kvDim),
    // Attention scores, nHeads x kvCapacity: (re)allocated by
    // ensureKvCapacity together with the KV cache.
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
    // RoPE scratch filled by fillRopeBuffers(), [batchIdx * ropeSize + i];
    // sized by setRopeSlots() (1 position, PREFILL_BATCH_SIZE during prefill).
    // Non-Gemma: SwaAll === All.
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
    // Layer size in bytes of the Q8_0 cache (nKvHeads * kvCapacity *
    // headBytesQ8)
    kvCacheLayerSize: kvCacheLayerBytes,
    kvCapacity: initialCap,
    // Token of every cached position (kvTokens[0 .. kvCount - 1]); generate()
    // keeps the positions whose tokens match the start of the next prompt.
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
  var xQ8Size = s._batchXQ8Size
  var deqCols = s._batchMatmulDeqCols
  ensureGenScratch(s)
  var buf = genScratch
  // matmulDeqBuf is a global read by the batch-matmul kernels: the first
  // 32 * deqCols bytes of the scratch, then one block per token
  matmulDeqBuf = new Float64Array(buf, 0, 4 * deqCols)
  matmulDeqI8 = new Int8Array(buf, 0, 32 * deqCols)
  matmulDeqRows = [
    new Float64Array(buf, 0, deqCols),
    new Float64Array(buf, deqCols * 8, deqCols),
    new Float64Array(buf, deqCols * 16, deqCols),
    new Float64Array(buf, deqCols * 24, deqCols),
  ]
  var off = deqCols * 32
  for (var b = 0; b < n; b = b + 1) {
    s.batchX[b] = new Float32Array(buf, off, dim)
    off = off + dim * 4
    s.batchXb[b] = new Float32Array(buf, off, maxDim)
    off = off + maxDim * 4
    s.batchXb2[b] = new Float32Array(buf, off, dim)
    off = off + dim * 4
    s.batchQ[b] = new Float32Array(buf, off, qDim)
    off = off + qDim * 4
    s.batchK[b] = new Float32Array(buf, off, kvDim)
    off = off + kvDim * 4
    s.batchV[b] = new Float32Array(buf, off, kvDim)
    off = off + kvDim * 4
    s.batchHb[b] = new Float32Array(buf, off, hiddenDim)
    off = off + hiddenDim * 4
    s.batchHb2[b] = new Float32Array(buf, off, hiddenDim)
    off = off + hiddenDim * 4
    s.batchQ8[b] = new Uint8Array(buf, off, xQ8Size)
    s.batchQ8i8[b] = new Int8Array(buf, off, xQ8Size)
    off = off + ((xQ8Size + 7) & ~7)
  }
  setRopeSlots(s, PREFILL_BATCH_SIZE)
  s._batchBuffersReady = true
}

// Release the prefill batch buffers once prefill is over (single-token
// generation doesn't use them)
function freeBatchBuffers(s) {
  if (!s._batchBuffersReady) {
    return
  }
  var n = PREFILL_BATCH_SIZE
  for (var b = 0; b < n; b = b + 1) {
    s.batchX[b] = null
    s.batchXb[b] = null
    s.batchXb2[b] = null
    s.batchQ[b] = null
    s.batchK[b] = null
    s.batchV[b] = null
    s.batchHb[b] = null
    s.batchHb2[b] = null
    s.batchQ8[b] = null
    s.batchQ8i8[b] = null
  }
  matmulDeqBuf = null
  matmulDeqRows = null
  matmulDeqI8 = null
  setRopeSlots(s, 1)
  s._batchBuffersReady = false
}

// Allocate the vocabSize-sized logits buffer on the first forward pass that
// needs it
function ensureLogits(s) {
  if (s.logits === null) {
    ensureGenScratch(s)
    s.logits = new Float32Array(genScratch, 0, s.vocabSize)
    // Float64 view over the same bytes: idle scratch for K-quant matmuls while
    // the layers run
    s.logits64 = new Float64Array(genScratch, 0, s.vocabSize >> 1)
  }
}

function ensureGenScratch(s) {
  if (genScratch === null) {
    genScratch = new ArrayBuffer(s.scratchBytes)
  }
}

// Grow the KV cache (doubling, capped at seqLen) when the next forward pass
// needs positions past its capacity; the positions written so far are copied to
// their place in the new buffer.
function ensureKvCapacity(s, needed) {
  var cap = s.kvCapacity
  if (needed <= cap) {
    return
  }
  var maxCap = s.seqLen
  // Start at cap (common case after the first allocation) or 1 so the
  // doubling loop can reach `needed` from zero on the first forward pass.
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
  // Scores of one position against every cached position (plain scratch,
  // nothing to carry over)
  s.att = new Float32Array(s.nHeads * newCap)
}

// Fill the RoPE cos/sin scratch for batchSize positions starting at startPos,
// indexed [b * ropeSize + i] (b: batch slot)
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
// Per-token element loops of the transformer. They live in small functions so
// the compiled code of the big transformer functions stays small.

// SwiGLU gate: hb = silu(hb) * hb2 with silu(x) = 0.5 * x * (1 + tanh(x / 2))
function siluGate(hbArr, hb2Arr, n) {
  var n4 = n & ~3
  for (var i = 0; i < n4; i = i + 4) {
    var v0 = hbArr[i]
    var v1 = hbArr[i + 1]
    var v2 = hbArr[i + 2]
    var v3 = hbArr[i + 3]
    hbArr[i] = 0.5 * v0 * (1.0 + fastTanh(0.5 * v0)) * hb2Arr[i]
    hbArr[i + 1] = 0.5 * v1 * (1.0 + fastTanh(0.5 * v1)) * hb2Arr[i + 1]
    hbArr[i + 2] = 0.5 * v2 * (1.0 + fastTanh(0.5 * v2)) * hb2Arr[i + 2]
    hbArr[i + 3] = 0.5 * v3 * (1.0 + fastTanh(0.5 * v3)) * hb2Arr[i + 3]
  }
  for (var i = n4; i < n; i = i + 1) {
    var val = hbArr[i]
    hbArr[i] = 0.5 * val * (1.0 + fastTanh(0.5 * val)) * hb2Arr[i]
  }
}

// GeGLU gate: hb = gelu(hb) * hb2 (tanh approximation)
function geluGate(hbArr, hb2Arr, n) {
  var n4 = n & ~3
  var GELU_A = 0.7978845608
  var GELU_B = 0.035677408137
  for (var i = 0; i < n4; i = i + 4) {
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
  for (var i = n4; i < n; i = i + 1) {
    var x = hbArr[i]
    hbArr[i] = 0.5 * x * (1.0 + fastTanh(x * (GELU_A + GELU_B * x * x))) * hb2Arr[i]
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
}

// Gemma (NEOX) RoPE on pairs (i, i + half) of every head, times scale
// (attention scale for Q, 1.0 for K: multiplying by 1.0 is exact).
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

function zeroFloats(arr, n) {
  for (var i = 0; i < n; i = i + 1) {
    arr[i] = 0
  }
}

// GQA-batched attention with Q8 scoring for one layer and position: every Q
// head of a KV group against the cached keys, softmax per head over [startT,
// pos], the Q8 values accumulated into xbArr.
function attendQ8(s, loff, pos, startT, xbArr) {
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

// Single-token matmuls of two or three matrices over the same input: when all
// of them take Q8_0 activations, x is quantized once for all.
function matmulSameInput(x, cols, o1, q1, o2, q2, o3, q3) {
  if (q1.unpackRowFunc && q2.unpackRowFunc && (q3 === null || q3.unpackRowFunc)) {
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

// Transformer forward pass, Llama and Gemma (per-head KV layout, Q8 attention,
// GQA batching; Gemma adds QK norms, NEOX RoPE, sliding-window attention, GELU
// and post-norms), in two shapes.
// transformer: one token at position pos (generation and the last prompt
// token), through the single-token kernels, plus the final norm and the logits.
function transformer(token, pos, computeLogits) {
  var w = weights
  var s = state
  if (w.hasKQuant) {
    // K-quant matmuls borrow the logits buffer as scratch during generation
    ensureLogits(s)
  }
  ensureKvCapacity(s, pos + 1)
  var isGemma = s.isGemma
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
  var attnScale = s.attnScale
  var half = headSize >> 1

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

    if (!isGemma) {
      rmsnorm(xbArr, xArr, lw.rmsAttWeight, dim, invDim)
    } else if (l === 0) {
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
      rmsnormGemma(xbArr, xArr, lw.rmsAttWeight, dim, eps, invDim)
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

    zeroFloats(xbArr, qDim)

    // SWA window enforcement
    var startT =
      isSwaLayer && config.swaWindow > 0
        ? Math.max(0, pos - config.swaWindow + 1)
        : 0

    attendQ8(s, loff, pos, startT, xbArr)

    // Attention output
    matmulQuantized(xb2Arr, xbArr, lw.wo)
    if (isGemma && lw.attnPostNorm) {
      rmsnormGemma(xb2Arr, xb2Arr, lw.attnPostNorm, dim, eps, invDim)
    }
    accum(xArr, xb2Arr, dim)

    // FFN
    if (isGemma) {
      rmsnormGemma(xbArr, xArr, lw.rmsFfnWeight, dim, eps, invDim)
    } else {
      rmsnorm(xbArr, xArr, lw.rmsFfnWeight, dim, invDim)
    }
    matmulSameInput(xbArr, dim, hbArr, lw.w1, hb2Arr, lw.w3, null, null)
    if (isGemma) {
      geluGate(hbArr, hb2Arr, hiddenDim)
    } else {
      siluGate(hbArr, hb2Arr, hiddenDim)
    }
    matmulQuantized(xbArr, hbArr, lw.w2)
    if (isGemma && lw.ffnPostNorm) {
      rmsnormGemma(xbArr, xbArr, lw.ffnPostNorm, dim, eps, invDim)
    }
    accum(xArr, xbArr, dim)
  }

  // Final norm
  if (isGemma) {
    rmsnormGemma(xArr, xArr, w.rmsFinalWeight, dim, eps, invDim)
  } else {
    rmsnorm(xArr, xArr, w.rmsFinalWeight, dim, invDim)
  }

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

// transformerPrefill: the prompt tokens allTokens[startPos .. startPos + n - 1]
// in one batch. Every weight matrix is read once for the whole batch and only
// the KV cache is produced.
function transformerPrefill(allTokens, startPos, n) {
  var w = weights
  var s = state
  ensureKvCapacity(s, startPos + n)
  ensureBatchBuffers(s)
  var isGemma = s.isGemma
  var dim = s.dim
  var headSize = s.headSize
  var kvDim = s.kvDim
  var qDim = s.qDim
  var hiddenDim = s.hiddenDim
  var eps = s.rmsNormEps
  var nLayers = s.nLayers
  var nHeads = s.nHeads
  var nKvHeads = s.nKvHeads
  var invDim = s.invDim
  var invHeadSize = s.invHeadSize
  var headBytesQ8 = s.headBytesQ8
  var headSeqBytes = s.headSeqBytes
  var attnScale = s.attnScale
  var half = headSize >> 1

  var keyCache = s.keyCache
  var valueCache = s.valueCache
  var keyCacheInt8 = s.keyCacheInt8
  var valueCacheInt8 = s.valueCacheInt8
  var qQ8 = s.qQ8
  var qQ8i8 = s.qQ8i8

  // One buffer of each kind per token
  var bX = s.batchX
  var bXb = s.batchXb
  var bXb2 = s.batchXb2
  var bQ = s.batchQ
  var bK = s.batchK
  var bV = s.batchV
  var bHb = s.batchHb
  var bHb2 = s.batchHb2

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
      if (!isGemma) {
        rmsnorm(bXb[b], bX[b], lw.rmsAttWeight, dim, invDim)
      } else if (l === 0) {
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
        rmsnormGemma(bXb[b], bX[b], lw.rmsAttWeight, dim, eps, invDim)
      }
    }

    // QKV matmuls
    if (!kvOnly) {
      matmulQuantizedBatch(bQ, bXb, lw.wq, n)
    }
    matmulQuantizedBatch(bK, bXb, lw.wk, n)
    matmulQuantizedBatch(bV, bXb, lw.wv, n)

    // Per token: QK norms, RoPE, KV cache write, attention
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
              rmsnormGemmaAt(
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

      zeroFloats(xbArr, qDim)

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
    matmulQuantizedBatch(bXb2, bXb, lw.wo, n)
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma && lw.attnPostNorm) {
        rmsnormGemma(bXb2[b], bXb2[b], lw.attnPostNorm, dim, eps, invDim)
      }
      accum(bX[b], bXb2[b], dim)
    }

    // FFN
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma) {
        rmsnormGemma(bXb[b], bX[b], lw.rmsFfnWeight, dim, eps, invDim)
      } else {
        rmsnorm(bXb[b], bX[b], lw.rmsFfnWeight, dim, invDim)
      }
    }
    matmulQuantizedBatch(bHb, bXb, lw.w1, n)
    matmulQuantizedBatch(bHb2, bXb, lw.w3, n)
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma) {
        geluGate(bHb[b], bHb2[b], hiddenDim)
      } else {
        siluGate(bHb[b], bHb2[b], hiddenDim)
      }
    }
    matmulQuantizedBatch(bXb, bHb, lw.w2, n)
    for (var b = 0; b < n; b = b + 1) {
      if (isGemma && lw.ffnPostNorm) {
        rmsnormGemma(bXb[b], bXb[b], lw.ffnPostNorm, dim, eps, invDim)
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

// Dequantize one token's embedding row into dst (through the matrix view when
// the format has a row dequantizer)
function embedToken(dst, token) {
  var emb = weights.tokenEmbedding
  if (emb.deqRowFunc) {
    emb.deqRowFunc(
      emb.deqView,
      emb.base + (token * emb.rowSize) / emb.deqView.BYTES_PER_ELEMENT,
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

// Vocab strings stay as bytes in the GGUF buffer and are decoded on demand.
// Offset of token i: its sparse checkpoint plus the lengths of the tokens
// before it in the bucket (at most 255 adds).
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

// Tokenizer lookup: chained hash table over the bytes of every non-empty token,
// built on the first bpeEncode() of a generate() call and released once the
// prompt is encoded (head[hash & mask] = highest id of the chain, next[id] =
// the next lower one, -1 ends it).
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
  // Running byte offset of token i, re-anchored at every sparse checkpoint
  // (same walk as findSpecialToken)
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

// First use of the vocabulary (first generate() call): per-token byte lengths
// (Uint8 when every token is shorter than 256 bytes), one offset checkpoint
// every 256 tokens, and the model-specific end-of-turn token
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

// Id of a special token given its text: a linear scan of the token strings in
// the file, done once per text (the few lookups of a model are cached)
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

// Longest token whose bytes are a prefix of bytes[pos .. pos + limit), or -1:
// the hash of every prefix is looked up as it grows, the last hit wins.
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

  // Use streaming UTF-8 decoder for Llama to handle multi-byte sequences across token boundaries
  var streamDecoder = config.isGemma ? null : createStreamingUTF8Decoder()

  var effectiveMaxTokens = maxTokens
  if (effectiveMaxTokens <= 0 || effectiveMaxTokens > config.seqLen) {
    effectiveMaxTokens = config.seqLen
  }

  // Reuse the KV cache of the previous call: the positions whose tokens match
  // the start of this prompt keep their keys and values. The last prompt token
  // always runs again (its logits are needed). With Q8_0 layer matrices only
  // the positions written by the batched prefill are kept: the single-token
  // Q8_0 kernel sums in another order, so a position it wrote is not
  // bit-identical to a prefilled one.
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

  // A prompt longer than the context window ran past the end of the KV cache:
  // nothing of it can be trusted by the next call.
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

if (typeof module !== "undefined") {
  module.exports = llama3pure
}
