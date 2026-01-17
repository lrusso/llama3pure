/*
----------------------------------------------------------------------------

Designed by Leonardo Javier Russo
https://www.lrusso.com

Web Worker for LLM Inference - Llama-3 and Gemma-3 Transformer models.
Supports GGUF file format with various quantization types.

----------------------------------------------------------------------------
*/

;(function () {
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
  var offset = 0

  var temperature = 0.9
  var systemPrompt = "You are a helpful assistant."
  var maxTokens = 256
  var contextSize = 2048

  // QuantizedTensor structure: { dataOffset, type, rows, cols }
  // Stores metadata to read quantized weights on-the-fly during matmul

  // ----------------------------------------------------------------------------
  // DataView helpers

  function readUint8() {
    var val = dataView.getUint8(offset)
    offset += 1
    return val
  }

  function readUint16() {
    var val = dataView.getUint16(offset, true)
    offset += 2
    return val
  }

  function readUint32() {
    var val = dataView.getUint32(offset, true)
    offset += 4
    return val
  }

  function readUint64() {
    var low = dataView.getUint32(offset, true)
    var high = dataView.getUint32(offset + 4, true)
    offset += 8
    return low + high * 0x100000000
  }

  function readInt8() {
    var val = dataView.getInt8(offset)
    offset += 1
    return val
  }

  function readInt32() {
    var val = dataView.getInt32(offset, true)
    offset += 4
    return val
  }

  function readInt64() {
    var low = dataView.getUint32(offset, true)
    var high = dataView.getInt32(offset + 4, true)
    offset += 8
    return low + high * 0x100000000
  }

  function readFloat32() {
    var val = dataView.getFloat32(offset, true)
    offset += 4
    return val
  }

  function readFloat64() {
    var val = dataView.getFloat64(offset, true)
    offset += 8
    return val
  }

  function readString() {
    var len = readUint64()
    var bytes = new Uint8Array(ggufData, offset, len)
    offset += len
    // Decode UTF-8 properly
    var decoder = new TextDecoder("utf-8")
    return decoder.decode(bytes)
  }

  function readBytes(n) {
    var arr = new Uint8Array(ggufData, offset, n)
    offset += n
    return arr
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
    for (var h = 0; h < 65536; h++) {
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
          exp--
        }
        exp++
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
      if (exp < -10) return sign // Too small, return signed zero
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

    for (var i = 0; i < nb; i++) {
      var blockStart = srcOffset + (i << 5) // i * 32

      // Find max absolute value in block
      var amax = 0.0
      for (var j = 0; j < 32; j++) {
        var av = src[blockStart + j]
        if (av < 0) av = -av
        if (av > amax) amax = av
      }

      // Compute scale
      var d = amax / 127.0
      var id = d > 0 ? 127.0 / amax : 0.0

      // Store scale as FP16
      var dFp16 = fp32ToFp16(d)
      dst[bo] = dFp16 & 0xff
      dst[bo + 1] = (dFp16 >> 8) & 0xff

      // Quantize and store values
      for (var j = 0; j < 32; j++) {
        var v = src[blockStart + j] * id
        // Round to nearest int8
        var q = v > 0 ? (v + 0.5) | 0 : (v - 0.5) | 0
        if (q > 127) q = 127
        if (q < -128) q = -128
        dstInt8[bo + 2 + j] = q
      }

      bo += Q8_0_BLOCK_SIZE
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

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(cache[bo] | (cache[bo + 1] << 8))
      var qOff = bo + 2

      var blockSum = 0.0
      for (var j = 0; j < 32; j++) {
        blockSum += x[xb + j] * cacheInt8[qOff + j]
      }
      sum += blockSum * d
      bo += Q8_0_BLOCK_SIZE
      xb += 32
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
    var nb = count >> 5
    var bo = cacheOffset
    var ob = outOffset

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(cache[bo] | (cache[bo + 1] << 8))
      var scale = d * weight
      var qOff = bo + 2

      for (var j = 0; j < 32; j++) {
        out[ob + j] += cacheInt8[qOff + j] * scale
      }
      bo += Q8_0_BLOCK_SIZE
      ob += 32
    }
  }

  // ----------------------------------------------------------------------------
  // Dequantization functions

  function dequantizeF16(srcOffset, dst, dstOffset, count) {
    var src = getUint16ArrayAt(srcOffset, count)
    for (var i = 0; i < count; i++) {
      dst[dstOffset + i] = fp16ToFp32(src[i])
    }
  }

  function dequantizeBF16(srcOffset, dst, dstOffset, count) {
    var src = getUint16ArrayAt(srcOffset, count)
    for (var i = 0; i < count; i++) {
      dst[dstOffset + i] = bf16ToFp32(src[i])
    }
  }

  function dequantizeF32(srcOffset, dst, dstOffset, count) {
    var src = getFloat32ArrayAt(srcOffset, count)
    for (var i = 0; i < count; i++) {
      dst[dstOffset + i] = src[i]
    }
  }

  function dequantizeQ4_0(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK4_0)
    var blockSize = 2 + QK4_0 / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))

      for (var j = 0; j < QK4_0 / 2; j++) {
        var qsByte = src[blockOffset + 2 + j]
        var x0 = (qsByte & 0x0f) - 8
        var x1 = (qsByte >> 4) - 8

        dst[dstOffset + i * QK4_0 + j] = x0 * d
        dst[dstOffset + i * QK4_0 + j + QK4_0 / 2] = x1 * d
      }
    }
  }

  function dequantizeQ4_1(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK4_1)
    var blockSize = 2 + 2 + QK4_1 / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
      var m = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))

      for (var j = 0; j < QK4_1 / 2; j++) {
        var qsByte = src[blockOffset + 4 + j]
        var x0 = qsByte & 0x0f
        var x1 = qsByte >> 4

        dst[dstOffset + i * QK4_1 + j] = x0 * d + m
        dst[dstOffset + i * QK4_1 + j + QK4_1 / 2] = x1 * d + m
      }
    }
  }

  function dequantizeQ8_0(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK8_0)
    var blockSize = 2 + QK8_0
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)
    var srcSigned = getInt8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))

      for (var j = 0; j < QK8_0; j++) {
        dst[dstOffset + i * QK8_0 + j] = srcSigned[blockOffset + 2 + j] * d
      }
    }
  }

  function dequantizeQ5_0(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK5_0)
    var blockSize = 2 + 4 + QK5_0 / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
      var qh =
        src[blockOffset + 2] |
        (src[blockOffset + 3] << 8) |
        (src[blockOffset + 4] << 16) |
        (src[blockOffset + 5] << 24)

      for (var j = 0; j < QK5_0 / 2; j++) {
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
    var nb = Math.floor(count / QK5_1)
    var blockSize = 2 + 2 + 4 + QK5_1 / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
      var m = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))
      var qh =
        src[blockOffset + 4] |
        (src[blockOffset + 5] << 8) |
        (src[blockOffset + 6] << 16) |
        (src[blockOffset + 7] << 24)

      for (var j = 0; j < QK5_1 / 2; j++) {
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
    var nb = Math.floor(count / QK_K)
    var blockSize = QK_K / 16 + QK_K / 4 + 2 + 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
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

      for (var n = 0; n < QK_K; n += 128) {
        var shift = 0
        for (var j = 0; j < 4; ++j) {
          var sc = scales[is++]
          var dl = d * (sc & 0xf)
          var ml = dmin * (sc >> 4)
          for (var l = 0; l < 16; ++l) {
            dst[y++] = dl * ((qs[qIdx + l] >> shift) & 3) - ml
          }

          sc = scales[is++]
          dl = d * (sc & 0xf)
          ml = dmin * (sc >> 4)
          for (var l = 0; l < 16; ++l) {
            dst[y++] = dl * ((qs[qIdx + l + 16] >> shift) & 3) - ml
          }

          shift += 2
        }
        qIdx += 32
      }
    }
  }

  function dequantizeQ3_K(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK_K)
    var blockSize = QK_K / 8 + QK_K / 4 + 12 + 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    var kmask1 = 0x03030303
    var kmask2 = 0x0f0f0f0f

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var hmask = src.subarray(blockOffset, blockOffset + QK_K / 8)
      var qs = src.subarray(
        blockOffset + QK_K / 8,
        blockOffset + QK_K / 8 + QK_K / 4
      )
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
      var scales = new Int8Array(16)
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

      for (var si = 0; si < 16; si++) {
        if (scales[si] > 127) scales[si] -= 256
      }

      var y = dstOffset + i * QK_K
      var is = 0
      var m = 1
      var qIdx = 0

      for (var n = 0; n < QK_K; n += 128) {
        var shift = 0
        for (var j = 0; j < 4; ++j) {
          var dl = d_all * (scales[is++] - 32)
          for (var l = 0; l < 16; ++l) {
            var q = (qs[qIdx + l] >> shift) & 3
            var h = hmask[l] & m ? 0 : 4
            dst[y++] = dl * (q - h)
          }

          dl = d_all * (scales[is++] - 32)
          for (var l = 0; l < 16; ++l) {
            var q = (qs[qIdx + l + 16] >> shift) & 3
            var h = hmask[l + 16] & m ? 0 : 4
            dst[y++] = dl * (q - h)
          }

          shift += 2
          m <<= 1
        }
        qIdx += 32
      }
    }
  }

  function dequantizeQ5_K(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK_K)
    var blockSize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
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

      for (var j = 0; j < QK_K; j += 64) {
        var sc, m
        if (is < 4) {
          sc = scales[is] & 63
          m = scales[is + 4] & 63
        } else {
          sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
          m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
        }
        var d1 = d * sc
        var m1 = dmin * m

        is++
        if (is < 4) {
          sc = scales[is] & 63
          m = scales[is + 4] & 63
        } else {
          sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
          m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
        }
        var d2 = d * sc
        var m2 = dmin * m
        is++

        for (var l = 0; l < 32; ++l) {
          dst[y + j + l] = d1 * ((ql[qlIdx + l] & 0xf) + (qh[l] & u1 ? 16 : 0)) - m1
        }
        for (var l = 0; l < 32; ++l) {
          dst[y + j + l + 32] =
            d2 * ((ql[qlIdx + l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2
        }

        qlIdx += 32
        u1 <<= 2
        u2 <<= 2
      }
    }
  }

  function dequantizeQ4_K(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK_K)
    var blockSize = 2 + 2 + 12 + QK_K / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))
      var dmin = fp16ToFp32(src[blockOffset + 2] | (src[blockOffset + 3] << 8))
      var scales = src.subarray(blockOffset + 4, blockOffset + 16)
      var qs = src.subarray(blockOffset + 16, blockOffset + 16 + QK_K / 2)

      var is = 0
      var y = dstOffset + i * QK_K

      for (var j = 0; j < QK_K; j += 64) {
        var sc, m
        if (is < 4) {
          sc = scales[is] & 63
          m = scales[is + 4] & 63
        } else {
          sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
          m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
        }
        var d1 = d * sc
        var m1 = dmin * m

        is++
        if (is < 4) {
          sc = scales[is] & 63
          m = scales[is + 4] & 63
        } else {
          sc = (scales[is + 4] & 0xf) | ((scales[is - 4] >> 6) << 4)
          m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4)
        }
        var d2 = d * sc
        var m2 = dmin * m
        is++

        var qIdx = j / 2
        for (var l = 0; l < 32; ++l) {
          var qByte = qs[qIdx + l]
          dst[y + j + l] = d1 * (qByte & 0xf) - m1
          dst[y + j + l + 32] = d2 * (qByte >> 4) - m2
        }
      }
    }
  }

  function dequantizeQ6_K(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK_K)
    var blockSize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)
    var srcSigned = getInt8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var ql = src.subarray(blockOffset, blockOffset + QK_K / 2)
      var qh = src.subarray(
        blockOffset + QK_K / 2,
        blockOffset + QK_K / 2 + QK_K / 4
      )
      var scalesOffset = blockOffset + QK_K / 2 + QK_K / 4
      var dOffset = blockOffset + QK_K / 2 + QK_K / 4 + QK_K / 16
      var d = fp16ToFp32(src[dOffset] | (src[dOffset + 1] << 8))

      var y = dstOffset + i * QK_K

      for (var n = 0; n < QK_K; n += 128) {
        for (var l = 0; l < 32; ++l) {
          var is = Math.floor(l / 16)
          var scBase = scalesOffset + Math.floor(n / 128) * 8
          var q1 = ((ql[n / 2 + l] & 0xf) | (((qh[n / 4 + l] >> 0) & 3) << 4)) - 32
          var q2 =
            ((ql[n / 2 + l + 32] & 0xf) | (((qh[n / 4 + l] >> 2) & 3) << 4)) - 32
          var q3 = ((ql[n / 2 + l] >> 4) | (((qh[n / 4 + l] >> 4) & 3) << 4)) - 32
          var q4 =
            ((ql[n / 2 + l + 32] >> 4) | (((qh[n / 4 + l] >> 6) & 3) << 4)) - 32

          dst[y + n + l] = d * srcSigned[scBase + is + 0] * q1
          dst[y + n + l + 32] = d * srcSigned[scBase + is + 2] * q2
          dst[y + n + l + 64] = d * srcSigned[scBase + is + 4] * q3
          dst[y + n + l + 96] = d * srcSigned[scBase + is + 6] * q4
        }
      }
    }
  }

  function dequantizeIQ4_NL(srcOffset, dst, dstOffset, count) {
    var nb = Math.floor(count / QK4_NL)
    var blockSize = 2 + QK4_NL / 2
    var totalBytes = nb * blockSize
    var src = getUint8ArrayAt(srcOffset, totalBytes)

    for (var i = 0; i < nb; i++) {
      var blockOffset = i * blockSize
      var d = fp16ToFp32(src[blockOffset] | (src[blockOffset + 1] << 8))

      for (var j = 0; j < QK4_NL / 2; j++) {
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
    return Math.floor(nCols / blockSize) * typeSize
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

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(ggufUint8[bo] | (ggufUint8[bo + 1] << 8))

      var blockSum = 0.0
      for (var j = 0; j < 16; j++) {
        var qsByte = ggufUint8[bo + 2 + j]
        var x0 = (qsByte & 0x0f) - 8
        var x1 = (qsByte >> 4) - 8
        blockSum += x[xb + j] * x0 + x[xb + j + 16] * x1
      }
      sum += blockSum * d
      bo += 18 // 2 + 16
      xb += 32
    }
    return sum
  }

  // Fused dot product for Q4_1
  function vecDotQ4_1(x, srcOffset, n) {
    var nb = n >> 5
    var sum = 0.0
    var bo = srcOffset
    var xb = 0

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(ggufUint8[bo] | (ggufUint8[bo + 1] << 8))
      var m = fp16ToFp32(ggufUint8[bo + 2] | (ggufUint8[bo + 3] << 8))

      var blockSum = 0.0
      var xSum = 0.0
      for (var j = 0; j < 16; j++) {
        var qsByte = ggufUint8[bo + 4 + j]
        var x0 = qsByte & 0x0f
        var x1 = qsByte >> 4
        blockSum += x[xb + j] * x0 + x[xb + j + 16] * x1
        xSum += x[xb + j] + x[xb + j + 16]
      }
      sum += blockSum * d + xSum * m
      bo += 20 // 2 + 2 + 16
      xb += 32
    }
    return sum
  }

  // Fused dot product for Q5_0
  function vecDotQ5_0(x, srcOffset, n) {
    var nb = n >> 5
    var sum = 0.0
    var bo = srcOffset
    var xb = 0

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(ggufUint8[bo] | (ggufUint8[bo + 1] << 8))
      var qh =
        ggufUint8[bo + 2] |
        (ggufUint8[bo + 3] << 8) |
        (ggufUint8[bo + 4] << 16) |
        (ggufUint8[bo + 5] << 24)

      var blockSum = 0.0
      for (var j = 0; j < 16; j++) {
        var xh_0 = ((qh >> j) & 1) << 4
        var xh_1 = ((qh >> (j + 16)) & 1) << 4
        var qsByte = ggufUint8[bo + 6 + j]
        var x0 = ((qsByte & 0x0f) | xh_0) - 16
        var x1 = ((qsByte >> 4) | xh_1) - 16
        blockSum += x[xb + j] * x0 + x[xb + j + 16] * x1
      }
      sum += blockSum * d
      bo += 22 // 2 + 4 + 16
      xb += 32
    }
    return sum
  }

  // Fused dot product for Q5_1
  function vecDotQ5_1(x, srcOffset, n) {
    var nb = n >> 5
    var sum = 0.0
    var bo = srcOffset
    var xb = 0

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(ggufUint8[bo] | (ggufUint8[bo + 1] << 8))
      var m = fp16ToFp32(ggufUint8[bo + 2] | (ggufUint8[bo + 3] << 8))
      var qh =
        ggufUint8[bo + 4] |
        (ggufUint8[bo + 5] << 8) |
        (ggufUint8[bo + 6] << 16) |
        (ggufUint8[bo + 7] << 24)

      var blockSum = 0.0
      var xSum = 0.0
      for (var j = 0; j < 16; j++) {
        var xh_0 = ((qh >> j) & 1) << 4
        var xh_1 = ((qh >> (j + 16)) & 1) << 4
        var qsByte = ggufUint8[bo + 8 + j]
        var x0 = (qsByte & 0x0f) | xh_0
        var x1 = (qsByte >> 4) | xh_1
        blockSum += x[xb + j] * x0 + x[xb + j + 16] * x1
        xSum += x[xb + j] + x[xb + j + 16]
      }
      sum += blockSum * d + xSum * m
      bo += 24 // 2 + 2 + 4 + 16
      xb += 32
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

    for (var i = 0; i < nb; i++) {
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

      sum += blockSum * d
      bo += 34
      xb += 32
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

    for (var i = 0; i < nb; i++) {
      var scOff = bo
      var qsOff = bo + 16
      var dOff = bo + 80
      var d = fp16ToFp32(u8[dOff] | (u8[dOff + 1] << 8))
      var dmin = fp16ToFp32(u8[dOff + 2] | (u8[dOff + 3] << 8))

      var is = 0
      var qIdx = 0
      var blockSum = 0.0

      for (var nOuter = 0; nOuter < 256; nOuter += 128) {
        var shift = 0
        for (var j = 0; j < 4; ++j) {
          var sc = u8[scOff + is++]
          var dl = d * (sc & 0xf)
          var ml = dmin * (sc >> 4)
          var baseIdx = xb + nOuter + j * 32
          var qBase = qsOff + qIdx
          for (var l = 0; l < 16; ++l) {
            blockSum += x[baseIdx + l] * (dl * ((u8[qBase + l] >> shift) & 3) - ml)
          }

          sc = u8[scOff + is++]
          dl = d * (sc & 0xf)
          ml = dmin * (sc >> 4)
          for (var l = 0; l < 16; ++l) {
            blockSum +=
              x[baseIdx + 16 + l] * (dl * ((u8[qBase + l + 16] >> shift) & 3) - ml)
          }
          shift += 2
        }
        qIdx += 32
      }
      sum += blockSum
      bo += 84
      xb += 256
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

    for (var i = 0; i < nb; i++) {
      var hmOff = bo
      var qsOff = bo + 32
      var scOff = bo + 96
      var dOff = bo + 108
      var dAll = fp16ToFp32(ggufUint8[dOff] | (ggufUint8[dOff + 1] << 8))

      var aux0 =
        ggufUint8[scOff] |
        (ggufUint8[scOff + 1] << 8) |
        (ggufUint8[scOff + 2] << 16) |
        (ggufUint8[scOff + 3] << 24)
      var aux1 =
        ggufUint8[scOff + 4] |
        (ggufUint8[scOff + 5] << 8) |
        (ggufUint8[scOff + 6] << 16) |
        (ggufUint8[scOff + 7] << 24)
      var aux2 =
        ggufUint8[scOff + 8] |
        (ggufUint8[scOff + 9] << 8) |
        (ggufUint8[scOff + 10] << 16) |
        (ggufUint8[scOff + 11] << 24)

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

      for (var si = 0; si < 16; si++) {
        if (q3kScales[si] > 127) q3kScales[si] -= 256
      }

      var is = 0
      var m = 1
      var qIdx = 0
      var blockSum = 0.0

      for (var nOuter = 0; nOuter < 256; nOuter += 128) {
        var shift = 0
        for (var j = 0; j < 4; ++j) {
          var dl = dAll * (q3kScales[is++] - 32)
          for (var l = 0; l < 16; ++l) {
            var q = (ggufUint8[qsOff + qIdx + l] >> shift) & 3
            var h = ggufUint8[hmOff + l] & m ? 0 : 4
            blockSum += x[xb + nOuter + j * 32 + l] * dl * (q - h)
          }

          dl = dAll * (q3kScales[is++] - 32)
          for (var l = 0; l < 16; ++l) {
            var q = (ggufUint8[qsOff + qIdx + l + 16] >> shift) & 3
            var h = ggufUint8[hmOff + l + 16] & m ? 0 : 4
            blockSum += x[xb + nOuter + j * 32 + 16 + l] * dl * (q - h)
          }
          shift += 2
          m <<= 1
        }
        qIdx += 32
      }
      sum += blockSum
      bo += 110 // 32 + 64 + 12 + 2
      xb += 256
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

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(u8[bo] | (u8[bo + 1] << 8))
      var dmin = fp16ToFp32(u8[bo + 2] | (u8[bo + 3] << 8))
      var scOff = bo + 4
      var qsOff = bo + 16

      var blockSum = 0.0

      // Unrolled: j=0 (is=0,1)
      var sc0 = u8[scOff] & 63,
        m0 = u8[scOff + 4] & 63
      var sc1 = u8[scOff + 1] & 63,
        m1 = u8[scOff + 5] & 63
      var d1 = d * sc0,
        dm1 = dmin * m0
      var d2 = d * sc1,
        dm2 = dmin * m1
      for (var l = 0; l < 32; ++l) {
        var qByte = u8[qsOff + l]
        blockSum += x[xb + l] * (d1 * (qByte & 0xf) - dm1)
        blockSum += x[xb + l + 32] * (d2 * (qByte >> 4) - dm2)
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
      for (var l = 0; l < 32; ++l) {
        var qByte = u8[qsOff + 32 + l]
        blockSum += x[xb + 64 + l] * (d1 * (qByte & 0xf) - dm1)
        blockSum += x[xb + 64 + l + 32] * (d2 * (qByte >> 4) - dm2)
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
      for (var l = 0; l < 32; ++l) {
        var qByte = u8[qsOff + 64 + l]
        blockSum += x[xb + 128 + l] * (d1 * (qByte & 0xf) - dm1)
        blockSum += x[xb + 128 + l + 32] * (d2 * (qByte >> 4) - dm2)
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
      for (var l = 0; l < 32; ++l) {
        var qByte = u8[qsOff + 96 + l]
        blockSum += x[xb + 192 + l] * (d1 * (qByte & 0xf) - dm1)
        blockSum += x[xb + 192 + l + 32] * (d2 * (qByte >> 4) - dm2)
      }

      sum += blockSum
      bo += 144
      xb += 256
    }
    return sum
  }

  // Fused dot product for Q5_K
  function vecDotQ5_K(x, srcOffset, n) {
    var nb = n >> 8
    var sum = 0.0
    var bo = srcOffset
    var xb = 0

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(ggufUint8[bo] | (ggufUint8[bo + 1] << 8))
      var dmin = fp16ToFp32(ggufUint8[bo + 2] | (ggufUint8[bo + 3] << 8))
      var scOff = bo + 4
      var qhOff = bo + 16
      var qlOff = bo + 48

      var is = 0
      var u1 = 1
      var u2 = 2
      var qlIdx = 0
      var blockSum = 0.0

      for (var j = 0; j < 256; j += 64) {
        var sc, m
        if (is < 4) {
          sc = ggufUint8[scOff + is] & 63
          m = ggufUint8[scOff + is + 4] & 63
        } else {
          sc =
            (ggufUint8[scOff + is + 4] & 0xf) |
            ((ggufUint8[scOff + is - 4] >> 6) << 4)
          m = (ggufUint8[scOff + is + 4] >> 4) | ((ggufUint8[scOff + is] >> 6) << 4)
        }
        var d1 = d * sc
        var m1 = dmin * m
        is++

        if (is < 4) {
          sc = ggufUint8[scOff + is] & 63
          m = ggufUint8[scOff + is + 4] & 63
        } else {
          sc =
            (ggufUint8[scOff + is + 4] & 0xf) |
            ((ggufUint8[scOff + is - 4] >> 6) << 4)
          m = (ggufUint8[scOff + is + 4] >> 4) | ((ggufUint8[scOff + is] >> 6) << 4)
        }
        var d2 = d * sc
        var m2 = dmin * m
        is++

        for (var l = 0; l < 32; ++l) {
          blockSum +=
            x[xb + j + l] *
            (d1 *
              ((ggufUint8[qlOff + qlIdx + l] & 0xf) +
                (ggufUint8[qhOff + l] & u1 ? 16 : 0)) -
              m1)
          blockSum +=
            x[xb + j + l + 32] *
            (d2 *
              ((ggufUint8[qlOff + qlIdx + l] >> 4) +
                (ggufUint8[qhOff + l] & u2 ? 16 : 0)) -
              m2)
        }
        qlIdx += 32
        u1 <<= 2
        u2 <<= 2
      }
      sum += blockSum
      bo += 176 // 2 + 2 + 12 + 32 + 128
      xb += 256
    }
    return sum
  }

  // Fused dot product for Q6_K
  function vecDotQ6_K(x, srcOffset, n) {
    var nb = n >> 8
    var sum = 0.0
    var bo = srcOffset
    var xb = 0

    for (var i = 0; i < nb; i++) {
      var qlOff = bo
      var qhOff = bo + 128
      var scOff = bo + 192
      var dOff = bo + 208
      var d = fp16ToFp32(ggufUint8[dOff] | (ggufUint8[dOff + 1] << 8))

      var blockSum = 0.0
      for (var nOuter = 0; nOuter < 256; nOuter += 128) {
        var scBase = scOff + (nOuter >> 7) * 8
        var qlBase = qlOff + (nOuter >> 1)
        var qhBase = qhOff + (nOuter >> 2)
        for (var l = 0; l < 32; ++l) {
          var is = l >> 4
          var q1 =
            ((ggufUint8[qlBase + l] & 0xf) |
              (((ggufUint8[qhBase + l] >> 0) & 3) << 4)) -
            32
          var q2 =
            ((ggufUint8[qlBase + l + 32] & 0xf) |
              (((ggufUint8[qhBase + l] >> 2) & 3) << 4)) -
            32
          var q3 =
            ((ggufUint8[qlBase + l] >> 4) |
              (((ggufUint8[qhBase + l] >> 4) & 3) << 4)) -
            32
          var q4 =
            ((ggufUint8[qlBase + l + 32] >> 4) |
              (((ggufUint8[qhBase + l] >> 6) & 3) << 4)) -
            32

          blockSum += x[xb + nOuter + l] * d * ggufInt8[scBase + is] * q1
          blockSum += x[xb + nOuter + l + 32] * d * ggufInt8[scBase + is + 2] * q2
          blockSum += x[xb + nOuter + l + 64] * d * ggufInt8[scBase + is + 4] * q3
          blockSum += x[xb + nOuter + l + 96] * d * ggufInt8[scBase + is + 6] * q4
        }
      }
      sum += blockSum
      bo += 210 // 128 + 64 + 16 + 2
      xb += 256
    }
    return sum
  }

  // Fused dot product for IQ4_NL
  function vecDotIQ4_NL(x, srcOffset, n) {
    var nb = n >> 5
    var sum = 0.0
    var bo = srcOffset
    var xb = 0

    for (var i = 0; i < nb; i++) {
      var d = fp16ToFp32(ggufUint8[bo] | (ggufUint8[bo + 1] << 8))

      var blockSum = 0.0
      for (var j = 0; j < 16; ++j) {
        var qsByte = ggufUint8[bo + 2 + j]
        blockSum += x[xb + j] * kvalues_iq4nl[qsByte & 0xf]
        blockSum += x[xb + j + 16] * kvalues_iq4nl[qsByte >> 4]
      }
      sum += blockSum * d
      bo += 18 // 2 + 16
      xb += 32
    }
    return sum
  }

  // Fused dot product for F16
  function vecDotF16(x, srcOffset, n) {
    var sum = 0.0
    var bo = srcOffset
    for (var i = 0; i < n; i++) {
      var h = ggufUint8[bo] | (ggufUint8[bo + 1] << 8)
      sum += x[i] * fp16ToFp32(h)
      bo += 2
    }
    return sum
  }

  // Fused dot product for BF16
  function vecDotBF16(x, srcOffset, n) {
    var sum = 0.0
    var bo = srcOffset
    for (var i = 0; i < n; i++) {
      var h = ggufUint8[bo] | (ggufUint8[bo + 1] << 8)
      sum += x[i] * bf16ToFp32(h)
      bo += 2
    }
    return sum
  }

  // Fused dot product for F32
  function vecDotF32(x, srcOffset, n) {
    var sum = 0.0
    var bo = srcOffset
    for (var i = 0; i < n; i++) {
      // Read 4 bytes as little-endian float
      convInt[0] =
        ggufUint8[bo] |
        (ggufUint8[bo + 1] << 8) |
        (ggufUint8[bo + 2] << 16) |
        (ggufUint8[bo + 3] << 24)
      sum += x[i] * convFloat[0]
      bo += 4
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

  function matmulQuantized(out, x, qw) {
    var rows = qw.rows
    var cols = qw.cols
    var baseOffset = qw.dataOffset
    var rowSize = qw.rowSize

    // Get function once before loop (avoids switch in hot path)
    var dotFunc = qw.dotFunc

    for (var i = 0; i < rows; i++) {
      out[i] = dotFunc(x, baseOffset + i * rowSize, cols)
    }
  }

  // ----------------------------------------------------------------------------
  // Math functions

  function rmsnorm(out, x, w, size, invSize, eps) {
    eps = eps || 1e-5
    invSize = invSize || 1.0 / size
    var ss = 0.0
    // Loop unrolling: process 4 elements at a time
    var size4 = size & ~3 // size - (size % 4)
    var i = 0
    for (; i < size4; i += 4) {
      var x0 = x[i],
        x1 = x[i + 1],
        x2 = x[i + 2],
        x3 = x[i + 3]
      ss += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
    }
    for (; i < size; i++) {
      ss += x[i] * x[i]
    }
    ss = 1.0 / Math.sqrt(ss * invSize + eps)
    i = 0
    for (; i < size4; i += 4) {
      out[i] = w[i] * ss * x[i]
      out[i + 1] = w[i + 1] * ss * x[i + 1]
      out[i + 2] = w[i + 2] * ss * x[i + 2]
      out[i + 3] = w[i + 3] * ss * x[i + 3]
    }
    for (; i < size; i++) {
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
    for (; i < size4; i += 4) {
      var x0 = x[i],
        x1 = x[i + 1],
        x2 = x[i + 2],
        x3 = x[i + 3]
      ss += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
    }
    for (; i < size; i++) {
      ss += x[i] * x[i]
    }
    ss = 1.0 / Math.sqrt(ss * invSize + eps)
    i = 0
    for (; i < size4; i += 4) {
      out[i] = w[i] * ss * x[i]
      out[i + 1] = w[i + 1] * ss * x[i + 1]
      out[i + 2] = w[i + 2] * ss * x[i + 2]
      out[i + 3] = w[i + 3] * ss * x[i + 3]
    }
    for (; i < size; i++) {
      out[i] = w[i] * ss * x[i]
    }
  }

  function softmax(x, size) {
    var maxVal = x[0]
    var size4 = size & ~3
    var i = 1
    // Find max with unrolled loop
    for (; i < size4; i += 4) {
      if (x[i] > maxVal) maxVal = x[i]
      if (x[i + 1] > maxVal) maxVal = x[i + 1]
      if (x[i + 2] > maxVal) maxVal = x[i + 2]
      if (x[i + 3] > maxVal) maxVal = x[i + 3]
    }
    for (; i < size; i++) {
      if (x[i] > maxVal) maxVal = x[i]
    }
    // Exp and sum with unrolled loop
    var sum = 0.0
    i = 0
    for (; i < size4; i += 4) {
      var e0 = Math.exp(x[i] - maxVal)
      var e1 = Math.exp(x[i + 1] - maxVal)
      var e2 = Math.exp(x[i + 2] - maxVal)
      var e3 = Math.exp(x[i + 3] - maxVal)
      x[i] = e0
      x[i + 1] = e1
      x[i + 2] = e2
      x[i + 3] = e3
      sum += e0 + e1 + e2 + e3
    }
    for (; i < size; i++) {
      x[i] = Math.exp(x[i] - maxVal)
      sum += x[i]
    }
    // Normalize with reciprocal multiply
    var invSum = 1.0 / sum
    i = 0
    for (; i < size4; i += 4) {
      x[i] *= invSum
      x[i + 1] *= invSum
      x[i + 2] *= invSum
      x[i + 3] *= invSum
    }
    for (; i < size; i++) {
      x[i] *= invSum
    }
  }

  function accum(a, b, size) {
    var size4 = size & ~3
    var i = 0
    for (; i < size4; i += 4) {
      a[i] += b[i]
      a[i + 1] += b[i + 1]
      a[i + 2] += b[i + 2]
      a[i + 3] += b[i + 3]
    }
    for (; i < size; i++) {
      a[i] += b[i]
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
    var i, key, valueType

    for (i = 0; i < nKV; i++) {
      key = readString()
      valueType = readUint32()
      metadata[key] = readGGUFValue(valueType)
    }

    var tensors = {}
    for (i = 0; i < nTensors; i++) {
      var name = readString()
      var nDims = readUint32()
      var dims = []
      for (var d = 0; d < nDims; d++) {
        dims.push(readUint64())
      }
      var type = readUint32()
      var tensorOffset = readUint64()

      var nElements = 1
      for (var d = 0; d < dims.length; d++) {
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
        return dataView.getInt16(offset, true)
        offset += 2
        return
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
        for (var i = 0; i < arrLen; i++) {
          arr.push(readGGUFValue(arrType))
        }
        return arr
      default:
        throw new Error("Unknown GGUF type: " + type)
    }
  }

  // ----------------------------------------------------------------------------
  // Model loading

  function loadModel(arrayBuffer) {
    // Reset sorted vocab cache when loading a new model
    sortedVocab = null
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
    }

    if (config.headDim === 0) {
      config.headDim = Math.floor(config.dim / config.nHeads)
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
      if (!t) return null
      return dequantizeTensor(baseOffset + t.offset, t.nElements, t.type)
    }

    // Load tensor keeping it quantized (for large weight matrices)
    // Returns a QuantizedTensor object with pre-computed rowSize and dotFunc
    function loadTensorQuantized(name, rows, cols) {
      var t = tensors[name]
      if (!t) return null
      return {
        dataOffset: baseOffset + t.offset,
        type: t.type,
        rows: rows,
        cols: cols,
        rowSize: getRowSize(cols, t.type),
        dotFunc: getVecDotFunc(t.type),
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

    for (var l = 0; l < config.nLayers; l++) {
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
      }
    }

    return w
  }

  function createRunState(p) {
    var headSize = p.headDim
    var kvDim = p.nKvHeads * headSize
    var qDim = p.nHeads * headSize
    var maxDim = Math.max(p.dim, qDim)

    // Pre-compute RoPE frequency table (freqs_cis equivalent)
    // For each position and each dimension pair, we store cos and sin
    var ropeSize = headSize / 2
    var ropeFreqs = new Float32Array(ropeSize)
    for (var i = 0; i < ropeSize; i++) {
      var dimIdx = i * 2
      ropeFreqs[i] = 1.0 / Math.pow(p.ropeTheta, dimIdx / headSize)
    }

    // Q8_0 KV cache: 34 bytes per 32 floats
    // Number of Q8_0 blocks per KV vector = kvDim / 32
    var kvBlocksPerVec = kvDim >> 5 // kvDim / 32
    var kvCacheBytesPerVec = kvBlocksPerVec * Q8_0_BLOCK_SIZE // 34 bytes per block
    var kvCacheTotalBytes = p.nLayers * p.seqLen * kvCacheBytesPerVec

    // Create ArrayBuffer for KV caches with both Uint8 and Int8 views
    var keyCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)
    var valueCacheBuffer = new ArrayBuffer(kvCacheTotalBytes)

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
      ropeFreqs: ropeFreqs,
      // RoPE sin/cos cache (avoids recomputing sin/cos for each head)
      ropeCos: new Float32Array(ropeSize),
      ropeSin: new Float32Array(ropeSize),
      ropeCachePos: -1, // position for which ropeCos/ropeSin are valid
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
    }
  }

  // ----------------------------------------------------------------------------
  // Transformer forward pass

  function transformer(token, pos) {
    var w = weights
    var s = state
    // Use cached values from state instead of config lookups
    var dim = s.dim
    var headSize = s.headSize
    var kvDim = s.kvDim
    var qDim = s.qDim
    var kvMul = s.kvMul
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
      for (var i = 0; i < dim; i++) {
        xArr[i] *= scale
      }
    }

    for (var l = 0; l < nLayers; l++) {
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
        for (var h = 0; h < nHeads; h++) {
          rmsnormGemma(
            s.q.subarray(h * headSize, (h + 1) * headSize),
            s.q.subarray(h * headSize, (h + 1) * headSize),
            lw.attnQNorm,
            headSize,
            eps,
            invHeadSize
          )
        }
        for (var h = 0; h < nKvHeads; h++) {
          rmsnormGemma(
            s.k.subarray(h * headSize, (h + 1) * headSize),
            s.k.subarray(h * headSize, (h + 1) * headSize),
            lw.attnKNorm,
            headSize,
            eps,
            invHeadSize
          )
        }
      }

      // Apply RoPE using pre-computed frequencies and cached sin/cos
      var ropeFreqs = s.ropeFreqs
      var ropeCos = s.ropeCos
      var ropeSin = s.ropeSin
      var half = headSize >> 1
      // Cache q and k array references for RoPE
      var qArr = s.q
      var kArr = s.k

      // Compute sin/cos once per position (cached across all layers)
      if (s.ropeCachePos !== pos) {
        for (var i = 0; i < half; i++) {
          var val = pos * ropeFreqs[i]
          ropeCos[i] = Math.cos(val)
          ropeSin[i] = Math.sin(val)
        }
        s.ropeCachePos = pos
      }

      if (isGemma) {
        for (var h = 0; h < nHeads; h++) {
          var idx = h * headSize
          for (var i = 0; i < half; i++) {
            var fcr = ropeCos[i]
            var fci = ropeSin[i]
            var v0 = qArr[idx + i]
            var v1 = qArr[idx + i + half]
            qArr[idx + i] = v0 * fcr - v1 * fci
            qArr[idx + i + half] = v0 * fci + v1 * fcr
          }
        }
        for (var h = 0; h < nKvHeads; h++) {
          var idx = h * headSize
          for (var i = 0; i < half; i++) {
            var fcr = ropeCos[i]
            var fci = ropeSin[i]
            var v0 = kArr[idx + i]
            var v1 = kArr[idx + i + half]
            kArr[idx + i] = v0 * fcr - v1 * fci
            kArr[idx + i + half] = v0 * fci + v1 * fcr
          }
        }
      } else {
        for (var i = 0; i < qDim; i += 2) {
          var freqIdx = (i >> 1) % half
          var fcr = ropeCos[freqIdx]
          var fci = ropeSin[freqIdx]

          var v0 = qArr[i]
          var v1 = qArr[i + 1]
          qArr[i] = v0 * fcr - v1 * fci
          qArr[i + 1] = v0 * fci + v1 * fcr

          if (i < kvDim) {
            v0 = kArr[i]
            v1 = kArr[i + 1]
            kArr[i] = v0 * fcr - v1 * fci
            kArr[i + 1] = v0 * fci + v1 * fcr
          }
        }
      }

      // Gemma: Scale Q by attention_scale - use pre-computed value
      if (isGemma) {
        var attnScale = s.attnScale
        for (var i = 0; i < qDim; i++) {
          qArr[i] *= attnScale
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
      var seqLen = s.seqLen

      // Zero xb once for all heads - more efficient than per-head zeroing
      for (var i = 0; i < qDim; i++) {
        xbArr[i] = 0
      }

      // Bytes per KV head = (headSize / 32) * 34
      var headBytesQ8 = (headSize >> 5) * Q8_0_BLOCK_SIZE

      for (var h = 0; h < nHeads; h++) {
        var qOffset = h * headSize
        var attOffset = h * seqLen
        // Use integer division via bitwise for kvMul (kvMul is always power of 2 or 1)
        var kvHeadIdx = (h / kvMul) | 0
        var kvHeadByteOff = kvHeadIdx * headBytesQ8

        // Compute attention scores using Q8_0 dot product
        var kBase = loff + kvHeadByteOff
        for (var t = 0; t <= pos; t++) {
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

        // Inline softmax to avoid subarray allocation
        var softmaxSize = pos + 1
        var maxVal = sAtt[attOffset]
        for (var i = 1; i < softmaxSize; i++) {
          if (sAtt[attOffset + i] > maxVal) maxVal = sAtt[attOffset + i]
        }
        var expSum = 0.0
        for (var i = 0; i < softmaxSize; i++) {
          var e = Math.exp(sAtt[attOffset + i] - maxVal)
          sAtt[attOffset + i] = e
          expSum += e
        }
        var invSum = 1.0 / expSum
        for (var i = 0; i < softmaxSize; i++) {
          sAtt[attOffset + i] *= invSum
        }

        // Accumulate weighted values using Q8_0 cache
        var xbOffset = h * headSize
        var vBase = loff + kvHeadByteOff
        for (var t = 0; t <= pos; t++) {
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
      if (isGemma) {
        // GELU activation
        for (var i = 0; i < hd4; i += 4) {
          var x0 = hbArr[i],
            x1 = hbArr[i + 1],
            x2 = hbArr[i + 2],
            x3 = hbArr[i + 3]
          hbArr[i] =
            0.5 *
            x0 *
            (1.0 + Math.tanh(0.7978845608 * x0 * (1.0 + 0.044715 * x0 * x0))) *
            hb2Arr[i]
          hbArr[i + 1] =
            0.5 *
            x1 *
            (1.0 + Math.tanh(0.7978845608 * x1 * (1.0 + 0.044715 * x1 * x1))) *
            hb2Arr[i + 1]
          hbArr[i + 2] =
            0.5 *
            x2 *
            (1.0 + Math.tanh(0.7978845608 * x2 * (1.0 + 0.044715 * x2 * x2))) *
            hb2Arr[i + 2]
          hbArr[i + 3] =
            0.5 *
            x3 *
            (1.0 + Math.tanh(0.7978845608 * x3 * (1.0 + 0.044715 * x3 * x3))) *
            hb2Arr[i + 3]
        }
        for (var i = hd4; i < hiddenDim; i++) {
          var x = hbArr[i]
          hbArr[i] =
            0.5 *
            x *
            (1.0 + Math.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x))) *
            hb2Arr[i]
        }
      } else {
        // SiLU activation
        for (var i = 0; i < hd4; i += 4) {
          var v0 = hbArr[i],
            v1 = hbArr[i + 1],
            v2 = hbArr[i + 2],
            v3 = hbArr[i + 3]
          hbArr[i] = (v0 / (1.0 + Math.exp(-v0))) * hb2Arr[i]
          hbArr[i + 1] = (v1 / (1.0 + Math.exp(-v1))) * hb2Arr[i + 1]
          hbArr[i + 2] = (v2 / (1.0 + Math.exp(-v2))) * hb2Arr[i + 2]
          hbArr[i + 3] = (v3 / (1.0 + Math.exp(-v3))) * hb2Arr[i + 3]
        }
        for (var i = hd4; i < hiddenDim; i++) {
          var val = hbArr[i]
          hbArr[i] = (val / (1.0 + Math.exp(-val))) * hb2Arr[i]
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

    // Classifier into logits - use quantized matmul for both cases
    // (tied embeddings are now kept quantized to save memory)
    matmulQuantized(s.logits, xArr, w.wcls)

    if (isGemma && config.finalLogitSoftcapping > 0) {
      var cap = config.finalLogitSoftcapping
      var vocabSize = s.vocabSize
      for (var i = 0; i < vocabSize; i++) {
        s.logits[i] = cap * Math.tanh(s.logits[i] / cap)
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
    for (var i = 1; i < n; i++) {
      if (logits[i] > maxP) {
        maxI = i
        maxP = logits[i]
      }
    }
    return maxI
  }

  function sampleMultinomial(probabilities, n) {
    var r = randomF32()
    var cdf = 0.0
    for (var i = 0; i < n; i++) {
      cdf += probabilities[i]
      if (r < cdf) {
        return i
      }
    }
    return n - 1
  }

  function sample(logits, temp) {
    if (temp === 0.0) {
      return sampleArgmax(logits, config.vocabSize)
    }

    for (var i = 0; i < config.vocabSize; i++) {
      logits[i] /= temp
    }

    softmax(logits, config.vocabSize)
    return sampleMultinomial(logits, config.vocabSize)
  }

  // ----------------------------------------------------------------------------
  // Tokenizer

  var sortedVocab = null
  var vocabMap = null

  function buildSortedVocab() {
    if (sortedVocab) return

    sortedVocab = []
    vocabMap = {}

    for (var i = 0; i < tokenizer.vocab.length; i++) {
      var token = tokenizer.vocab[i]
      if (token && token.length > 0) {
        sortedVocab.push({ id: i, len: token.length, token: token })
        vocabMap[token] = i
      }
    }

    // Sort by length descending (longest first)
    sortedVocab.sort(function (a, b) {
      return b.len - a.len
    })
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
    if (tiktokenByteToUnicode) return
    tiktokenByteToUnicode = {}

    // This is OpenAI's bytes_to_unicode() function
    // Printable ASCII and some extended chars map to themselves
    var n = 0
    for (var b = 0; b < 256; b++) {
      // These byte ranges map directly: ! to ~, ¡ to ¬, ® to ÿ
      if (
        (b >= 33 && b <= 126) ||
        (b >= 161 && b <= 172) ||
        (b >= 174 && b <= 255)
      ) {
        tiktokenByteToUnicode[b] = b
      } else {
        // Other bytes (0-32, 127-160, 173) map to 256+n, 257+n, etc.
        tiktokenByteToUnicode[b] = 256 + n
        n++
      }
    }
  }

  function textToTiktoken(text) {
    buildTiktokenByteToUnicodeMap()

    var result = ""
    for (var i = 0; i < text.length; i++) {
      var code = text.charCodeAt(i)

      // Handle UTF-16 surrogate pairs
      if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
        var low = text.charCodeAt(i + 1)
        if (low >= 0xdc00 && low <= 0xdfff) {
          code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00)
          i++
        }
      }

      // Convert unicode to UTF-8 bytes, then map each byte to tiktoken unicode
      if (code < 0x80) {
        result += String.fromCharCode(tiktokenByteToUnicode[code])
      } else if (code < 0x800) {
        result += String.fromCharCode(tiktokenByteToUnicode[0xc0 | (code >> 6)])
        result += String.fromCharCode(tiktokenByteToUnicode[0x80 | (code & 0x3f)])
      } else if (code < 0x10000) {
        result += String.fromCharCode(tiktokenByteToUnicode[0xe0 | (code >> 12)])
        result += String.fromCharCode(
          tiktokenByteToUnicode[0x80 | ((code >> 6) & 0x3f)]
        )
        result += String.fromCharCode(tiktokenByteToUnicode[0x80 | (code & 0x3f)])
      } else {
        result += String.fromCharCode(tiktokenByteToUnicode[0xf0 | (code >> 18)])
        result += String.fromCharCode(
          tiktokenByteToUnicode[0x80 | ((code >> 12) & 0x3f)]
        )
        result += String.fromCharCode(
          tiktokenByteToUnicode[0x80 | ((code >> 6) & 0x3f)]
        )
        result += String.fromCharCode(tiktokenByteToUnicode[0x80 | (code & 0x3f)])
      }
    }
    return result
  }

  function textToSentencePiece(text) {
    var result = ""
    var needPrefix = true // Add ▁ before first alphanumeric char

    for (var i = 0; i < text.length; i++) {
      var c = text.charAt(i)
      var code = text.charCodeAt(i)

      if (c === " ") {
        // Space -> ▁ (U+2581)
        result += "\u2581"
        needPrefix = false // ▁ already added for the space
      } else if (c === "\n" || c === "\t" || c === "\r") {
        // Control characters are kept as-is
        result += c
        needPrefix = true // Next word needs prefix
      } else {
        // Regular character - add prefix if this is start of a word
        if (
          needPrefix &&
          ((code >= 65 && code <= 90) ||
            (code >= 97 && code <= 122) ||
            (code >= 48 && code <= 57))
        ) {
          result += "\u2581"
        }
        result += c
        needPrefix = false
      }
    }

    return result
  }

  // Streaming UTF-8 decoder for Llama tokens
  var utf8Decoder = new TextDecoder("utf-8")

  // Build tiktoken unicode-to-byte mapping (inverse of bytes_to_unicode)
  var tiktokenUnicodeToByte = null

  function buildTiktokenMap() {
    if (tiktokenUnicodeToByte) return
    tiktokenUnicodeToByte = {}

    // This is the inverse of OpenAI's bytes_to_unicode() function
    // Printable ASCII and some extended chars map to themselves
    var n = 0
    for (var b = 0; b < 256; b++) {
      // These byte ranges map directly: ! to ~, ¡ to ¬, ® to ÿ
      if (
        (b >= 33 && b <= 126) ||
        (b >= 161 && b <= 172) ||
        (b >= 174 && b <= 255)
      ) {
        tiktokenUnicodeToByte[b] = b
      } else {
        // Other bytes (0-32, 127-160, 173) map to 256+n, 257+n, etc.
        tiktokenUnicodeToByte[256 + n] = b
        n++
      }
    }
  }

  function tokenToBytes(token) {
    if (token < 0 || token >= tokenizer.vocab.length) {
      return new Uint8Array(0)
    }
    var piece = tokenizer.vocab[token]
    if (!piece) return new Uint8Array(0)

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

    // Collect bytes from tiktoken encoding
    var bytes = []
    for (var i = 0; i < piece.length; i++) {
      var code = piece.charCodeAt(i)

      // Handle UTF-16 surrogate pairs
      if (code >= 0xd800 && code <= 0xdbff && i + 1 < piece.length) {
        var low = piece.charCodeAt(i + 1)
        if (low >= 0xdc00 && low <= 0xdfff) {
          code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00)
          i++
        }
      }

      // Look up in tiktoken mapping
      if (tiktokenUnicodeToByte[code] !== undefined) {
        bytes.push(tiktokenUnicodeToByte[code])
      } else {
        // Fallback: encode unknown unicode as UTF-8
        if (code < 0x80) {
          bytes.push(code)
        } else if (code < 0x800) {
          bytes.push(0xc0 | (code >> 6))
          bytes.push(0x80 | (code & 0x3f))
        } else if (code < 0x10000) {
          bytes.push(0xe0 | (code >> 12))
          bytes.push(0x80 | ((code >> 6) & 0x3f))
          bytes.push(0x80 | (code & 0x3f))
        } else {
          bytes.push(0xf0 | (code >> 18))
          bytes.push(0x80 | ((code >> 12) & 0x3f))
          bytes.push(0x80 | ((code >> 6) & 0x3f))
          bytes.push(0x80 | (code & 0x3f))
        }
      }
    }

    return new Uint8Array(bytes)
  }

  function decodeToken(token) {
    if (token < 0 || token >= tokenizer.vocab.length) {
      return ""
    }
    var piece = tokenizer.vocab[token]
    if (!piece) return ""

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
    var missedChars = 0

    while (pos < encodedText.length) {
      var bestId = -1
      var bestLen = 0

      // Greedy longest-match
      for (var i = 0; i < sortedVocab.length; i++) {
        var entry = sortedVocab[i]

        // Skip if token is longer than remaining text
        if (entry.len > encodedText.length - pos) continue

        // Skip if shorter than best match we already found (optimization)
        if (bestLen > 0 && entry.len <= bestLen) break

        // Check if vocab entry matches
        if (encodedText.substring(pos, pos + entry.len) === entry.token) {
          bestId = entry.id
          bestLen = entry.len
          break
        }
      }

      if (bestId !== -1) {
        tokens.push(bestId)
        pos += bestLen
      } else {
        // No match found - try to find single character token
        var singleChar = encodedText.charAt(pos)
        var singleId = vocabMap[singleChar]
        if (singleId !== undefined) {
          tokens.push(singleId)
        } else {
          missedChars++
        }
        // Skip this character regardless
        pos++
      }
    }

    return tokens
  }

  function encodeLlama3Chat(prompt, sysPrompt) {
    var tokens = []

    // Find special tokens
    var bosToken = findSpecialToken("<|begin_of_text|>")
    if (bosToken < 0) bosToken = 128000

    var startHeader = findSpecialToken("<|start_header_id|>")
    if (startHeader < 0) startHeader = 128006

    var endHeader = findSpecialToken("<|end_header_id|>")
    if (endHeader < 0) endHeader = 128007

    var eotToken = findSpecialToken("<|eot_id|>")
    if (eotToken < 0) eotToken = 128009

    // <|begin_of_text|>
    tokens.push(bosToken)

    // System prompt if provided
    if (sysPrompt && sysPrompt.length > 0) {
      tokens.push(startHeader)
      var sysTokens = bpeEncode("system")
      for (var i = 0; i < sysTokens.length; i++) tokens.push(sysTokens[i])
      tokens.push(endHeader)

      var sysTextTokens = bpeEncode("\n\n" + sysPrompt)
      for (var i = 0; i < sysTextTokens.length; i++) tokens.push(sysTextTokens[i])
      tokens.push(eotToken)
    }

    // User prompt
    tokens.push(startHeader)
    var userTokens = bpeEncode("user")
    for (var i = 0; i < userTokens.length; i++) tokens.push(userTokens[i])
    tokens.push(endHeader)

    var userTextTokens = bpeEncode("\n\n" + prompt)
    for (var i = 0; i < userTextTokens.length; i++) tokens.push(userTextTokens[i])
    tokens.push(eotToken)

    // Assistant header
    tokens.push(startHeader)
    var assistantTokens = bpeEncode("assistant")
    for (var i = 0; i < assistantTokens.length; i++) tokens.push(assistantTokens[i])
    tokens.push(endHeader)

    var newlineTokens = bpeEncode("\n\n")
    for (var i = 0; i < newlineTokens.length; i++) tokens.push(newlineTokens[i])

    return tokens
  }

  function encodeGemma3Chat(prompt, sysPrompt) {
    var tokens = []

    // Find special tokens
    var bosToken = findSpecialToken("<bos>")
    if (bosToken < 0) bosToken = 2 // Default Gemma3 BOS

    var startTurn = findSpecialToken("<start_of_turn>")
    if (startTurn < 0) startTurn = 106 // Default Gemma3 start_of_turn

    var endTurn = findSpecialToken("<end_of_turn>")
    if (endTurn < 0) endTurn = 107 // Default Gemma3 end_of_turn

    // <bos>
    tokens.push(bosToken)

    // Prepare full prompt (optionally include system prompt)
    var fullPrompt = prompt
    if (sysPrompt && sysPrompt.length > 0) {
      fullPrompt = sysPrompt + "\n\n" + prompt
    }

    // <start_of_turn>
    tokens.push(startTurn)

    // "user\n" + prompt
    var userTokens = bpeEncode("user\n" + fullPrompt)
    for (var i = 0; i < userTokens.length; i++) tokens.push(userTokens[i])

    // <end_of_turn>
    tokens.push(endTurn)

    // "\n"
    var newlineTokens = bpeEncode("\n")
    for (var i = 0; i < newlineTokens.length; i++) tokens.push(newlineTokens[i])

    // <start_of_turn>
    tokens.push(startTurn)

    // "model\n"
    var modelTokens = bpeEncode("model\n")
    for (var i = 0; i < modelTokens.length; i++) tokens.push(modelTokens[i])

    return tokens
  }

  // ----------------------------------------------------------------------------
  // Generation

  function generate(prompt) {
    var promptTokens
    if (config.isGemma) {
      promptTokens = encodeGemma3Chat(prompt, systemPrompt)
    } else {
      promptTokens = encodeLlama3Chat(prompt, systemPrompt)
    }

    if (promptTokens.length === 0) {
      promptTokens = [tokenizer.bosToken]
    }

    state.keyCache.fill(0)
    state.valueCache.fill(0)

    var token = promptTokens[0]
    var pos = 0
    var output = ""
    var numPromptTokens = promptTokens.length
    var pendingNewline = false

    // Use streaming TextDecoder for Llama to handle UTF-8 across token boundaries
    var streamDecoder = config.isGemma
      ? null
      : new TextDecoder("utf-8", { fatal: false })

    for (var step = 0; step < maxTokens; step++) {
      transformer(token, pos)

      var next
      if (pos < numPromptTokens - 1) {
        next = promptTokens[pos + 1]
      } else {
        next = sample(state.logits, temperature)
      }

      if (pos >= numPromptTokens - 1) {
        if (next === tokenizer.eosToken) break

        // Check for model-specific end tokens
        if (config.isGemma) {
          if (tokenizer.vocab[next] === "<end_of_turn>") break
        } else {
          // Llama: check for EOT token
          if (tokenizer.vocab[next] === "<|eot_id|>") break
          if (next === 128009) break // Default EOT token ID
        }

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
            output += "\n"
            postMessage({ type: "token", token: "\n" })
            pendingNewline = false
          }
          if (decoded.length > 0) {
            output += decoded
            postMessage({ type: "token", token: decoded })
          }
        }
      }

      token = next
      pos++
    }

    // Flush any remaining bytes in the decoder
    if (!config.isGemma) {
      var remaining = streamDecoder.decode()
      if (remaining.length > 0) {
        output += remaining
        postMessage({ type: "token", token: remaining })
      }
    }

    postMessage({ type: "complete", output: output })

    return output
  }

  // ----------------------------------------------------------------------------
  // Message handler

  self.onmessage = function (e) {
    var data = e.data

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
          var result = loadModel(data.arrayBuffer)
          postMessage({
            type: "loaded",
            config: result.config,
            maxTokens: maxTokens,
            contextSize: contextSize,
          })
          break

        case "generate":
          generate(data.prompt)
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
})()
