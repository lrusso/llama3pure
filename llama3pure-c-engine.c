/*
----------------------------------------------------------------------------

Designed by Leonardo Javier Russo
https://www.lrusso.com

LLM Inference - Llama-3 and Gemma-3 Transformer models.
Supports GGUF file format with various quantization types.

----------------------------------------------------------------------------
*/

#if defined _WIN32
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include <windows.h>
    #include <io.h>
    #include <malloc.h>

    #define open    _open
    #define close   _close
    #define lseek   _lseeki64
    #define strdup  _strdup
    typedef __int64 off_t;

    #define PROT_READ     0x1
    #define MAP_PRIVATE   0x02
    #define MAP_FAILED    ((void*)-1)

    static void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
        (void)addr; (void)prot; (void)flags; (void)offset;
        HANDLE hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
        HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMapping) return MAP_FAILED;
        void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMapping);
        return ptr ? ptr : MAP_FAILED;
    }

    static int munmap(void* addr, size_t length) {
        (void)length;
        return UnmapViewOfFile(addr) ? 0 : -1;
    }
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Global debug flag (set via -debug command line argument)
int debug_mode = 0;

// ----------------------------------------------------------------------------
// GGUF file format definitions

#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian

// GGUF value types
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// GGML tensor types (quantization formats)
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 29,
    GGML_TYPE_COUNT,
};

// Block sizes for quantized types
#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK_K 256
#define QK4_NL 32  // IQ4_NL block size

// Quantization block structures
typedef struct {
    uint16_t d;       // delta (fp16)
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

typedef struct {
    uint16_t d;       // delta (fp16)
    uint16_t m;       // min (fp16)
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

typedef struct {
    uint16_t d;       // delta (fp16)
    uint8_t qh[4];    // 5th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;

typedef struct {
    uint16_t d;       // delta (fp16)
    uint16_t m;       // min (fp16)
    uint8_t qh[4];    // 5th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;

typedef struct {
    uint16_t d;       // delta (fp16)
    int8_t  qs[QK8_0];    // quants
} block_q8_0;

// Q4_K block
typedef struct {
    uint16_t d;       // super-block scale (fp16)
    uint16_t dmin;    // super-block min (fp16)
    uint8_t scales[12];  // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];  // 4-bit quants
} block_q4_K;

// Q2_K block
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants, 2 bits per weight
    uint16_t d;              // super-block scale (fp16)
    uint16_t dmin;           // super-block min (fp16)
} block_q2_K;

// Q3_K block
typedef struct {
    uint8_t hmask[QK_K/8];   // quants - high bit
    uint8_t qs[QK_K/4];      // quants - low 2 bits
    uint8_t scales[12];      // scales, quantized with 6 bits
    uint16_t d;              // super-block scale (fp16)
} block_q3_K;

// Q5_K block
typedef struct {
    uint16_t d;              // super-block scale (fp16)
    uint16_t dmin;           // super-block min (fp16)
    uint8_t scales[12];      // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;

// Q6_K block
typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;           // super-block scale (fp16)
} block_q6_K;

// IQ4_NL block - non-linear 4-bit quantization
typedef struct {
    uint16_t d;           // delta (fp16)
    uint8_t qs[QK4_NL/2]; // nibbles / quants
} block_iq4_nl;

// IQ4_NL lookup table for dequantization (non-linear values)
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim;           // transformer dimension
    int hidden_dim;    // for ffn layers
    int n_layers;      // number of layers
    int n_heads;       // number of query heads
    int n_kv_heads;    // number of key/value heads (can be < query heads because of multiquery/GQA)
    int vocab_size;    // vocabulary size
    int seq_len;       // max sequence length
    float rope_theta;  // RoPE theta base frequency (500000.0 for Llama 3, 1000000.0 for Gemma3)
    int head_dim;      // explicit head dimension (for Gemma3 where it differs from dim/n_heads)
    // Gemma3-specific parameters
    int is_gemma3;                    // flag to indicate Gemma3 architecture
    float final_logit_softcapping;    // final logit softcapping value (30.0 for Gemma3)
    float rms_norm_eps;               // RMS norm epsilon
    // Sliding window attention parameters
    int swa_window;                   // sliding window attention window size (0 = no SWA)
    float rope_theta_swa;             // RoPE theta for SWA layers (default 10000.0)
    int swa_pattern;                  // SWA pattern (6 for Gemma3: every 6th layer is dense)
} Config;

// Quantized tensor - keeps weights in compressed form
typedef struct {
    void* data;           // raw quantized data (points into mmap'd file)
    enum ggml_type type;  // quantization type
    int n_elements;       // total number of elements (for output dimension)
    int rows;             // number of rows (output dim for matmul)
    int cols;             // number of cols (input dim for matmul)
} QuantizedTensor;

typedef struct {
    // token embedding table - kept as float for fast lookup
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms - small, kept as float
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls - QUANTIZED - note: for GQA, wk and wv have different sizes
    QuantizedTensor* wq; // (layer) each: (n_heads * head_size, dim)
    QuantizedTensor* wk; // (layer) each: (n_kv_heads * head_size, dim)
    QuantizedTensor* wv; // (layer) each: (n_kv_heads * head_size, dim)
    QuantizedTensor* wo; // (layer) each: (dim, n_heads * head_size)
    // weights for ffn - QUANTIZED
    QuantizedTensor* w1; // (layer) each: (hidden_dim, dim)
    QuantizedTensor* w2; // (layer) each: (dim, hidden_dim)
    QuantizedTensor* w3; // (layer) each: (hidden_dim, dim)
    // final rmsnorm - small, kept as float
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits - QUANTIZED
    QuantizedTensor wcls;

    // Gemma3-specific weights - small, kept as float
    float* attn_q_norm;     // (layer, head_dim) Q projection normalization
    float* attn_k_norm;     // (layer, head_dim) K projection normalization
    float* attn_post_norm;  // (layer, dim) post-attention normalization
    float* ffn_post_norm;   // (layer, dim) post-FFN normalization
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x;      // activation at current time stamp (dim,)
    float *xb;     // same, but inside a residual branch (dim,)
    float *xb2;    // an additional buffer just for convenience (dim,)
    float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;      // query (n_heads * head_size,)
    float *k;      // key (n_kv_heads * head_size,)
    float *v;      // value (n_kv_heads * head_size,)
    float *att;    // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache - sized for GQA
    float* key_cache;   // (layer, seq_len, n_kv_heads * head_size)
    float* value_cache; // (layer, seq_len, n_kv_heads * head_size)
    // pre-computed RoPE frequencies (avoids powf in hot loop)
    float* rope_freqs;      // (head_size / 2,) - for dense (non-SWA) layers
    float* rope_freqs_swa;  // (head_size / 2,) - for SWA layers (different theta)
    // Pre-computed RoPE sin/cos for ALL positions (eliminates sin/cos from hot loop)
    float* rope_cos_all;     // (seq_len * head_size/2) - cos for all positions (dense layers)
    float* rope_sin_all;     // (seq_len * head_size/2) - sin for all positions (dense layers)
    float* rope_cos_swa_all; // (seq_len * head_size/2) - cos for all positions (SWA layers)
    float* rope_sin_swa_all; // (seq_len * head_size/2) - sin for all positions (SWA layers)
    // cached constants to avoid recomputation in transformer
    int head_size;
    int kv_dim;
    int q_dim;
    int kv_mul;
    int kv_cache_layer_size;  // seq_len * kv_dim (for layer offset calculation)
    float attn_scale;         // 1/sqrt(head_size) for attention scaling
    float embed_scale;        // sqrt(dim) for Gemma3 embedding scaling
} RunState;

// Global memory tracking for cleanup
static void* mapped_data = NULL;
static size_t mapped_size = 0;
static float** weight_allocations = NULL;
static int num_weight_allocations = 0;

void malloc_run_state(RunState* s, Config* p) {
    // Use explicit head_dim for Gemma3, computed for others
    int head_size = p->head_dim > 0 ? p->head_dim : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int q_dim = p->n_heads * head_size;  // total Q dimension (for Gemma3)

    // For Gemma3, q_dim may differ from dim, so use max of both for buffers that need both
    int max_dim = p->dim > q_dim ? p->dim : q_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(max_dim, sizeof(float));  // needs dim for rmsnorm input, q_dim for attention output
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(q_dim, sizeof(float));   // use q_dim for query
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));

    // Pre-compute RoPE frequencies to avoid powf in hot loop
    int rope_size = head_size / 2;
    s->rope_freqs = calloc(rope_size, sizeof(float));
    for (int i = 0; i < rope_size; i++) {
        s->rope_freqs[i] = 1.0f / powf(p->rope_theta, (float)(i * 2) / (float)head_size);
    }

    // Pre-compute SWA RoPE frequencies (for Gemma3 sliding window attention layers)
    s->rope_freqs_swa = calloc(rope_size, sizeof(float));
    float swa_theta = p->rope_theta_swa > 0 ? p->rope_theta_swa : 10000.0f;
    for (int i = 0; i < rope_size; i++) {
        s->rope_freqs_swa[i] = 1.0f / powf(swa_theta, (float)(i * 2) / (float)head_size);
    }

    // Pre-compute RoPE sin/cos for ALL positions (eliminates sin/cos from transformer hot loop)
    int seq_len = p->seq_len;
    s->rope_cos_all = calloc((size_t)seq_len * rope_size, sizeof(float));
    s->rope_sin_all = calloc((size_t)seq_len * rope_size, sizeof(float));
    for (int pos = 0; pos < seq_len; pos++) {
        int base = pos * rope_size;
        for (int i = 0; i < rope_size; i++) {
            float val = pos * s->rope_freqs[i];
            s->rope_cos_all[base + i] = cosf(val);
            s->rope_sin_all[base + i] = sinf(val);
        }
    }
    // Pre-compute SWA sin/cos for all positions
    s->rope_cos_swa_all = calloc((size_t)seq_len * rope_size, sizeof(float));
    s->rope_sin_swa_all = calloc((size_t)seq_len * rope_size, sizeof(float));
    for (int pos = 0; pos < seq_len; pos++) {
        int base = pos * rope_size;
        for (int i = 0; i < rope_size; i++) {
            float val = pos * s->rope_freqs_swa[i];
            s->rope_cos_swa_all[base + i] = cosf(val);
            s->rope_sin_swa_all[base + i] = sinf(val);
        }
    }

    // Cache frequently used values
    s->head_size = head_size;
    s->kv_dim = kv_dim;
    s->q_dim = q_dim;
    s->kv_mul = p->n_heads / p->n_kv_heads;
    s->kv_cache_layer_size = p->seq_len * kv_dim;
    s->attn_scale = 1.0f / sqrtf((float)head_size);
    s->embed_scale = sqrtf((float)p->dim);

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || !s->rope_freqs || !s->rope_freqs_swa
     || !s->rope_cos_all || !s->rope_sin_all
     || !s->rope_cos_swa_all || !s->rope_sin_swa_all) {
        fprintf(stderr, "malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->rope_freqs);
    free(s->rope_freqs_swa);
    free(s->rope_cos_all);
    free(s->rope_sin_all);
    free(s->rope_cos_swa_all);
    free(s->rope_sin_swa_all);
}

// ----------------------------------------------------------------------------
// FP16 conversion utilities - Pre-computed lookup table for O(1) conversion

// Pre-computed FP16 to FP32 lookup table (256KB - all 65536 possible FP16 values)
static float fp16_table[65536];

static void init_fp16_table(void) {
    for (int i = 0; i < 65536; i++) {
        uint16_t h = (uint16_t)i;
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) {
                uint32_t result = sign;
                fp16_table[i] = *(float*)&result;
                continue;
            } else {
                while (!(mant & 0x400)) {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= ~0x400;
            }
        } else if (exp == 31) {
            uint32_t result = sign | 0x7F800000 | (mant << 13);
            fp16_table[i] = *(float*)&result;
            continue;
        }

        exp = exp + (127 - 15);
        mant = mant << 13;
        uint32_t result = sign | (exp << 23) | mant;
        fp16_table[i] = *(float*)&result;
    }
}

static inline float fp16_to_fp32(uint16_t h) {
    return fp16_table[h];
}

// FP32 to FP16 conversion (for Q8_0 input quantization)
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | (mant >> 13);
}

// BF16 (bfloat16) to FP32 conversion
// BF16 is simply the upper 16 bits of a float32
static inline float bf16_to_fp32(uint16_t h) {
    uint32_t result = (uint32_t)h << 16;
    return *(float*)&result;
}

// ----------------------------------------------------------------------------
// Dequantization functions

void dequantize_row_q4_0(const void* src, float* dst, int k) {
    const block_q4_0* blocks = (const block_q4_0*)src;
    int nb = k / QK4_0;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);

        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (blocks[i].qs[j] & 0x0F) - 8;
            const int x1 = (blocks[i].qs[j] >> 4) - 8;

            dst[i*QK4_0 + j]           = x0 * d;
            dst[i*QK4_0 + j + QK4_0/2] = x1 * d;
        }
    }
}

void dequantize_row_q4_1(const void* src, float* dst, int k) {
    const block_q4_1* blocks = (const block_q4_1*)src;
    int nb = k / QK4_1;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float m = fp16_to_fp32(blocks[i].m);

        for (int j = 0; j < QK4_1/2; j++) {
            const int x0 = blocks[i].qs[j] & 0x0F;
            const int x1 = blocks[i].qs[j] >> 4;

            dst[i*QK4_1 + j]           = x0 * d + m;
            dst[i*QK4_1 + j + QK4_1/2] = x1 * d + m;
        }
    }
}

void dequantize_row_q5_0(const void* src, float* dst, int k) {
    const block_q5_0* blocks = (const block_q5_0*)src;
    int nb = k / QK5_0;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        uint32_t qh = blocks[i].qh[0] | (blocks[i].qh[1] << 8) |
                      (blocks[i].qh[2] << 16) | (blocks[i].qh[3] << 24);

        for (int j = 0; j < QK5_0/2; j++) {
            const int xh_0 = ((qh >> j) & 1) << 4;
            const int xh_1 = ((qh >> (j + 16)) & 1) << 4;

            const int x0 = (blocks[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (blocks[i].qs[j] >> 4) | xh_1;

            dst[i*QK5_0 + j]           = (x0 - 16) * d;
            dst[i*QK5_0 + j + QK5_0/2] = (x1 - 16) * d;
        }
    }
}

void dequantize_row_q5_1(const void* src, float* dst, int k) {
    const block_q5_1* blocks = (const block_q5_1*)src;
    int nb = k / QK5_1;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float m = fp16_to_fp32(blocks[i].m);
        uint32_t qh = blocks[i].qh[0] | (blocks[i].qh[1] << 8) |
                      (blocks[i].qh[2] << 16) | (blocks[i].qh[3] << 24);

        for (int j = 0; j < QK5_1/2; j++) {
            const int xh_0 = ((qh >> j) & 1) << 4;
            const int xh_1 = ((qh >> (j + 16)) & 1) << 4;

            const int x0 = (blocks[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (blocks[i].qs[j] >> 4) | xh_1;

            dst[i*QK5_1 + j]           = x0 * d + m;
            dst[i*QK5_1 + j + QK5_1/2] = x1 * d + m;
        }
    }
}

void dequantize_row_q8_0(const void* src, float* dst, int k) {
    const block_q8_0* blocks = (const block_q8_0*)src;
    int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);

        for (int j = 0; j < QK8_0; j++) {
            dst[i*QK8_0 + j] = blocks[i].qs[j] * d;
        }
    }
}

// Helper function to decode scale and min for Q4_K/Q5_K formats
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_row_q4_K(const void* src, float* dst, int k) {
    const block_q4_K* blocks = (const block_q4_K*)src;
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);

        const uint8_t* q = blocks[i].qs;
        uint8_t sc, m;
        float* y = dst + i * QK_K;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

void dequantize_row_q2_K(const void* src, float* dst, int k) {
    const block_q2_K* blocks = (const block_q2_K*)src;
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float min = fp16_to_fp32(blocks[i].dmin);

        const uint8_t* q = blocks[i].qs;
        const uint8_t* scales = blocks[i].scales;

        int is = 0;
        float* y = dst + i * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((q[l] >> shift) & 3) - ml;
                }

                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                }

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q3_K(const void* src, float* dst, int k) {
    const block_q3_K* x = (const block_q3_K*)src;
    int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(x[i].d);

        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        float* y = dst + i * QK_K;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_row_q5_K(const void* src, float* dst, int k) {
    const block_q5_K* x = (const block_q5_K*)src;
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;

        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        float* y = dst + i * QK_K;

        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const void* src, float* dst, int k) {
    const block_q6_K* x = (const block_q6_K*)src;
    int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        float* y = dst + i * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row_f16(const void* src, float* dst, int k) {
    const uint16_t* src16 = (const uint16_t*)src;
    for (int i = 0; i < k; i++) {
        dst[i] = fp16_to_fp32(src16[i]);
    }
}

void dequantize_row_bf16(const void* src, float* dst, int k) {
    const uint16_t* src16 = (const uint16_t*)src;
    for (int i = 0; i < k; i++) {
        dst[i] = bf16_to_fp32(src16[i]);
    }
}

// IQ4_NL dequantization - uses non-linear lookup table
void dequantize_row_iq4_nl(const void* src, float* dst, int k) {
    const block_iq4_nl* blocks = (const block_iq4_nl*)src;
    int nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* qs = blocks[i].qs;

        for (int j = 0; j < QK4_NL/2; ++j) {
            dst[i*QK4_NL + j]             = d * kvalues_iq4nl[qs[j] & 0xf];
            dst[i*QK4_NL + j + QK4_NL/2]  = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

// Generic dequantization dispatcher
float* dequantize_tensor(const void* src, int n_elements, enum ggml_type type) {
    float* dst = malloc(n_elements * sizeof(float));
    if (!dst) {
        fprintf(stderr, "Failed to allocate memory for dequantization\n");
        exit(1);
    }

    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n_elements * sizeof(float));
            break;
        case GGML_TYPE_F16:
            dequantize_row_f16(src, dst, n_elements);
            break;
        case GGML_TYPE_Q4_0:
            dequantize_row_q4_0(src, dst, n_elements);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_row_q4_1(src, dst, n_elements);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_row_q5_0(src, dst, n_elements);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_row_q5_1(src, dst, n_elements);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0(src, dst, n_elements);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K(src, dst, n_elements);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K(src, dst, n_elements);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(src, dst, n_elements);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K(src, dst, n_elements);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(src, dst, n_elements);
            break;
        case GGML_TYPE_IQ4_NL:
            dequantize_row_iq4_nl(src, dst, n_elements);
            break;
        case GGML_TYPE_BF16:
        case 30:  // Some GGUF files use type 30 for BF16
            dequantize_row_bf16(src, dst, n_elements);
            break;
        default:
            fprintf(stderr, "Unsupported quantization type: %d\n", type);
            free(dst);
            exit(1);
    }

    return dst;
}

// Get block size for quantization type
int get_block_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 1;
        case GGML_TYPE_F16: return 1;
        case GGML_TYPE_Q4_0: return QK4_0;
        case GGML_TYPE_Q4_1: return QK4_1;
        case GGML_TYPE_Q5_0: return QK5_0;
        case GGML_TYPE_Q5_1: return QK5_1;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_Q4_K: return QK_K;
        case GGML_TYPE_Q6_K: return QK_K;
        case GGML_TYPE_Q2_K: return QK_K;
        case GGML_TYPE_Q3_K: return QK_K;
        case GGML_TYPE_Q5_K: return QK_K;
        case GGML_TYPE_IQ4_NL: return QK4_NL;
        case GGML_TYPE_BF16: return 1;
        case 30: return 1;  // BF16 alternate type
        default: return 1;
    }
}

// Get bytes per block for quantization type
size_t get_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return sizeof(float);
        case GGML_TYPE_F16: return sizeof(uint16_t);
        case GGML_TYPE_BF16: return sizeof(uint16_t);
        case 30: return sizeof(uint16_t);  // BF16 alternate type
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        default: return 0;
    }
}

// ----------------------------------------------------------------------------
// Fused quantized vector-matrix multiplication
// These compute dot products directly from quantized weights without full dequantization
// Each row is processed independently: xout[i] = dot(x, w_row[i])

// Fused dot product for Q4_0: computes dot(x, quantized_row) for one row
static inline float vec_dot_q4_0(const float* x, const void* w, int n) {
    const block_q4_0* blocks = (const block_q4_0*)w;
    int nb = n / QK4_0;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* qs = blocks[i].qs;
        const float* xb = x + i * QK4_0;

        float block_sum = 0.0f;
        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >> 4) - 8;
            block_sum += xb[j] * x0 + xb[j + QK4_0/2] * x1;
        }
        sum += block_sum * d;
    }
    return sum;
}

// Fused dot product for Q4_1
static inline float vec_dot_q4_1(const float* x, const void* w, int n) {
    const block_q4_1* blocks = (const block_q4_1*)w;
    int nb = n / QK4_1;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float m = fp16_to_fp32(blocks[i].m);
        const uint8_t* qs = blocks[i].qs;
        const float* xb = x + i * QK4_1;

        float block_sum = 0.0f;
        float x_sum = 0.0f;
        for (int j = 0; j < QK4_1/2; j++) {
            const int x0 = qs[j] & 0x0F;
            const int x1 = qs[j] >> 4;
            block_sum += xb[j] * x0 + xb[j + QK4_1/2] * x1;
            x_sum += xb[j] + xb[j + QK4_1/2];
        }
        sum += block_sum * d + x_sum * m;
    }
    return sum;
}

// Fused dot product for Q5_0
static inline float vec_dot_q5_0(const float* x, const void* w, int n) {
    const block_q5_0* blocks = (const block_q5_0*)w;
    int nb = n / QK5_0;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        uint32_t qh = blocks[i].qh[0] | (blocks[i].qh[1] << 8) |
                      (blocks[i].qh[2] << 16) | (blocks[i].qh[3] << 24);
        const uint8_t* qs = blocks[i].qs;
        const float* xb = x + i * QK5_0;

        float block_sum = 0.0f;
        for (int j = 0; j < QK5_0/2; j++) {
            const int xh_0 = ((qh >> j) & 1) << 4;
            const int xh_1 = ((qh >> (j + 16)) & 1) << 4;
            const int x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            const int x1 = ((qs[j] >> 4) | xh_1) - 16;
            block_sum += xb[j] * x0 + xb[j + QK5_0/2] * x1;
        }
        sum += block_sum * d;
    }
    return sum;
}

// Fused dot product for Q5_1
static inline float vec_dot_q5_1(const float* x, const void* w, int n) {
    const block_q5_1* blocks = (const block_q5_1*)w;
    int nb = n / QK5_1;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float m = fp16_to_fp32(blocks[i].m);
        uint32_t qh = blocks[i].qh[0] | (blocks[i].qh[1] << 8) |
                      (blocks[i].qh[2] << 16) | (blocks[i].qh[3] << 24);
        const uint8_t* qs = blocks[i].qs;
        const float* xb = x + i * QK5_1;

        float block_sum = 0.0f;
        float x_sum = 0.0f;
        for (int j = 0; j < QK5_1/2; j++) {
            const int xh_0 = ((qh >> j) & 1) << 4;
            const int xh_1 = ((qh >> (j + 16)) & 1) << 4;
            const int x0 = (qs[j] & 0x0F) | xh_0;
            const int x1 = (qs[j] >> 4) | xh_1;
            block_sum += xb[j] * x0 + xb[j + QK5_1/2] * x1;
            x_sum += xb[j] + xb[j + QK5_1/2];
        }
        sum += block_sum * d + x_sum * m;
    }
    return sum;
}

// Fused dot product for Q8_0
static inline float vec_dot_q8_0(const float* x, const void* w, int n) {
    const block_q8_0* blocks = (const block_q8_0*)w;
    int nb = n / QK8_0;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const int8_t* qs = blocks[i].qs;
        const float* xb = x + i * QK8_0;

        float block_sum = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            block_sum += xb[j] * qs[j];
        }
        sum += block_sum * d;
    }
    return sum;
}

// Fused dot product for Q2_K
static inline float vec_dot_q2_K(const float* x, const void* w, int n) {
    const block_q2_K* blocks = (const block_q2_K*)w;
    int nb = n / QK_K;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const uint8_t* q = blocks[i].qs;
        const uint8_t* scales = blocks[i].scales;
        const float* xb = x + i * QK_K;

        int is = 0;
        float block_sum = 0.0f;

        for (int n_outer = 0; n_outer < QK_K; n_outer += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);

                for (int l = 0; l < 16; ++l) {
                    int idx = n_outer + j * 32 + l;
                    float w_val = dl * ((q[l] >> shift) & 3) - ml;
                    block_sum += xb[idx] * w_val;
                }

                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);

                for (int l = 0; l < 16; ++l) {
                    int idx = n_outer + j * 32 + 16 + l;
                    float w_val = dl * ((q[l + 16] >> shift) & 3) - ml;
                    block_sum += xb[idx] * w_val;
                }

                shift += 2;
            }
            q += 32;
        }
        sum += block_sum;
    }
    return sum;
}

// Fused dot product for Q3_K
static inline float vec_dot_q3_K(const float* x, const void* w, int n) {
    const block_q3_K* blocks = (const block_q3_K*)w;
    int nb = n / QK_K;
    float sum = 0.0f;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(blocks[i].d);
        const uint8_t* q = blocks[i].qs;
        const uint8_t* hm = blocks[i].hmask;
        const float* xb = x + i * QK_K;

        uint32_t aux[4];
        const int8_t* scales = (const int8_t*)aux;
        memcpy(aux, blocks[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float block_sum = 0.0f;
        uint8_t m = 1;

        for (int n_outer = 0; n_outer < QK_K; n_outer += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = n_outer + j * 32 + l;
                    int8_t w_val = ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    block_sum += xb[idx] * dl * w_val;
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int idx = n_outer + j * 32 + 16 + l;
                    int8_t w_val = ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    block_sum += xb[idx] * dl * w_val;
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        sum += block_sum;
    }
    return sum;
}

// Fused dot product for Q4_K
static inline float vec_dot_q4_K(const float* x, const void* w, int n) {
    const block_q4_K* blocks = (const block_q4_K*)w;
    int nb = n / QK_K;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const uint8_t* q = blocks[i].qs;
        const float* xb = x + i * QK_K;

        int is = 0;
        float block_sum = 0.0f;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                float w_val = d1 * (q[l] & 0xF) - m1;
                block_sum += xb[j + l] * w_val;
            }
            for (int l = 0; l < 32; ++l) {
                float w_val = d2 * (q[l] >> 4) - m2;
                block_sum += xb[j + 32 + l] * w_val;
            }
            q += 32;
            is += 2;
        }
        sum += block_sum;
    }
    return sum;
}

// Fused dot product for Q5_K
static inline float vec_dot_q5_K(const float* x, const void* w, int n) {
    const block_q5_K* blocks = (const block_q5_K*)w;
    int nb = n / QK_K;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t* ql = blocks[i].qs;
        const uint8_t* qh = blocks[i].qh;
        const float d = fp16_to_fp32(blocks[i].d);
        const float dmin = fp16_to_fp32(blocks[i].dmin);
        const float* xb = x + i * QK_K;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        float block_sum = 0.0f;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, blocks[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = dmin * m;
            get_scale_min_k4(is + 1, blocks[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = dmin * m;

            for (int l = 0; l < 32; ++l) {
                float w_val = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                block_sum += xb[j + l] * w_val;
            }
            for (int l = 0; l < 32; ++l) {
                float w_val = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                block_sum += xb[j + 32 + l] * w_val;
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += block_sum;
    }
    return sum;
}

// Fused dot product for Q6_K
static inline float vec_dot_q6_K(const float* x, const void* w, int n) {
    const block_q6_K* blocks = (const block_q6_K*)w;
    int nb = n / QK_K;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t* sc = blocks[i].scales;
        const float* xb = x + i * QK_K;

        float block_sum = 0.0f;

        for (int n_outer = 0; n_outer < QK_K; n_outer += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                block_sum += xb[n_outer + l +  0] * d * sc[is + 0] * q1;
                block_sum += xb[n_outer + l + 32] * d * sc[is + 2] * q2;
                block_sum += xb[n_outer + l + 64] * d * sc[is + 4] * q3;
                block_sum += xb[n_outer + l + 96] * d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
        sum += block_sum;
    }
    return sum;
}

// Fused dot product for IQ4_NL
static inline float vec_dot_iq4_nl(const float* x, const void* w, int n) {
    const block_iq4_nl* blocks = (const block_iq4_nl*)w;
    int nb = n / QK4_NL;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* qs = blocks[i].qs;
        const float* xb = x + i * QK4_NL;

        float block_sum = 0.0f;
        for (int j = 0; j < QK4_NL/2; ++j) {
            block_sum += xb[j] * kvalues_iq4nl[qs[j] & 0xf];
            block_sum += xb[j + QK4_NL/2] * kvalues_iq4nl[qs[j] >> 4];
        }
        sum += block_sum * d;
    }
    return sum;
}

// Fused dot product for F16
static inline float vec_dot_f16(const float* x, const void* w, int n) {
    const uint16_t* w16 = (const uint16_t*)w;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * fp16_to_fp32(w16[i]);
    }
    return sum;
}

// Fused dot product for BF16
static inline float vec_dot_bf16(const float* x, const void* w, int n) {
    const uint16_t* w16 = (const uint16_t*)w;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * bf16_to_fp32(w16[i]);
    }
    return sum;
}

// Fused dot product for F32 (reference)
static inline float vec_dot_f32(const float* x, const void* w, int n) {
    const float* wf = (const float*)w;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * wf[i];
    }
    return sum;
}

// ----------------------------------------------------------------------------
// Quantized input (Q8_0) dot products - integer arithmetic for faster matmul
// Quantize input x to Q8_0 once, then use int8*int dot products per row

// Global Q8_0 buffer for quantized input (allocated once, reused per matmul)
static block_q8_0* q8_buf = NULL;

// Quantize a float vector to Q8_0 format
static void quantize_row_q8_0(const float* x, block_q8_0* y, int n) {
    int nb = n / QK8_0;
    for (int i = 0; i < nb; i++) {
        const float* xb = x + i * QK8_0;
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = xb[j] < 0 ? -xb[j] : xb[j];
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 127.0f / amax : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) {
            float v = xb[j] * id;
            int q = (int)roundf(v);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            y[i].qs[j] = (int8_t)q;
        }
    }
}

// Q8_0 input dot product for Q4_0 weights
static inline float vec_dot_q4_0_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q4_0* y = (const block_q4_0*)vy;
    int nb = n / QK4_0;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float dw = fp16_to_fp32(y[i].d);
        float dx = fp16_to_fp32(xq[i].d);
        const uint8_t* qs = y[i].qs;
        const int8_t* xqs = xq[i].qs;
        int isum = 0;
        for (int j = 0; j < 16; j++) {
            uint8_t qByte = qs[j];
            isum += xqs[j] * ((qByte & 0x0F) - 8)
                  + xqs[j + 16] * ((qByte >> 4) - 8);
        }
        sum += dw * dx * (float)isum;
    }
    return sum;
}

// Q8_0 input dot product for Q4_1 weights
static inline float vec_dot_q4_1_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q4_1* y = (const block_q4_1*)vy;
    int nb = n / QK4_1;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float dw = fp16_to_fp32(y[i].d);
        float mw = fp16_to_fp32(y[i].m);
        float dx = fp16_to_fp32(xq[i].d);
        const uint8_t* qs = y[i].qs;
        const int8_t* xqs = xq[i].qs;
        int isum = 0, xsum = 0;
        for (int j = 0; j < 16; j++) {
            uint8_t qByte = qs[j];
            isum += xqs[j] * (qByte & 0x0F) + xqs[j + 16] * (qByte >> 4);
            xsum += xqs[j] + xqs[j + 16];
        }
        sum += dw * dx * (float)isum + mw * dx * (float)xsum;
    }
    return sum;
}

// Q8_0 input dot product for Q5_0 weights
static inline float vec_dot_q5_0_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q5_0* y = (const block_q5_0*)vy;
    int nb = n / QK5_0;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float dw = fp16_to_fp32(y[i].d);
        float dx = fp16_to_fp32(xq[i].d);
        uint32_t qh = y[i].qh[0] | ((uint32_t)y[i].qh[1] << 8)
                     | ((uint32_t)y[i].qh[2] << 16) | ((uint32_t)y[i].qh[3] << 24);
        const uint8_t* qs = y[i].qs;
        const int8_t* xqs = xq[i].qs;
        int isum = 0;
        for (int j = 0; j < 16; j++) {
            int xh_0 = ((qh >> j) & 1) << 4;
            int xh_1 = ((qh >> (j + 16)) & 1) << 4;
            uint8_t qByte = qs[j];
            isum += xqs[j] * (((qByte & 0x0F) | xh_0) - 16)
                  + xqs[j + 16] * (((qByte >> 4) | xh_1) - 16);
        }
        sum += dw * dx * (float)isum;
    }
    return sum;
}

// Q8_0 input dot product for Q5_1 weights
static inline float vec_dot_q5_1_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q5_1* y = (const block_q5_1*)vy;
    int nb = n / QK5_1;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float dw = fp16_to_fp32(y[i].d);
        float mw = fp16_to_fp32(y[i].m);
        float dx = fp16_to_fp32(xq[i].d);
        uint32_t qh = y[i].qh[0] | ((uint32_t)y[i].qh[1] << 8)
                     | ((uint32_t)y[i].qh[2] << 16) | ((uint32_t)y[i].qh[3] << 24);
        const uint8_t* qs = y[i].qs;
        const int8_t* xqs = xq[i].qs;
        int isum = 0, xsum = 0;
        for (int j = 0; j < 16; j++) {
            int xh_0 = ((qh >> j) & 1) << 4;
            int xh_1 = ((qh >> (j + 16)) & 1) << 4;
            uint8_t qByte = qs[j];
            isum += xqs[j] * ((qByte & 0x0F) | xh_0)
                  + xqs[j + 16] * ((qByte >> 4) | xh_1);
            xsum += xqs[j] + xqs[j + 16];
        }
        sum += dw * dx * (float)isum + mw * dx * (float)xsum;
    }
    return sum;
}

// Q8_0 input dot product for Q8_0 weights (int8 * int8)
static inline float vec_dot_q8_0_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q8_0* y = (const block_q8_0*)vy;
    int nb = n / QK8_0;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float dw = fp16_to_fp32(y[i].d);
        float dx = fp16_to_fp32(xq[i].d);
        const int8_t* wqs = y[i].qs;
        const int8_t* xqs = xq[i].qs;
        int isum = 0;
        for (int j = 0; j < 32; j++) {
            isum += xqs[j] * wqs[j];
        }
        sum += dw * dx * (float)isum;
    }
    return sum;
}

// Q8_0 input dot product for Q2_K weights
static inline float vec_dot_q2_K_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q2_K* y = (const block_q2_K*)vy;
    int nb = n / QK_K;
    float sum = 0.0f;
    int xq_idx = 0;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(y[i].d);
        float dmin = fp16_to_fp32(y[i].dmin);
        const uint8_t* q = y[i].qs;
        int is = 0;
        int q_idx = 0;
        float blockSum = 0.0f;
        for (int nOuter = 0; nOuter < 256; nOuter += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dx = fp16_to_fp32(xq[xq_idx].d);
                const int8_t* xqs = xq[xq_idx].qs;
                uint8_t sc = y[i].scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                int isum1 = 0, xsum1 = 0;
                for (int l = 0; l < 16; l++) {
                    isum1 += xqs[l] * ((q[q_idx + l] >> shift) & 3);
                    xsum1 += xqs[l];
                }
                sc = y[i].scales[is++];
                float dl2 = d * (sc & 0xF);
                float ml2 = dmin * (sc >> 4);
                int isum2 = 0, xsum2 = 0;
                for (int l = 0; l < 16; l++) {
                    isum2 += xqs[16 + l] * ((q[q_idx + 16 + l] >> shift) & 3);
                    xsum2 += xqs[16 + l];
                }
                blockSum += dx * (dl * isum1 - ml * xsum1 + dl2 * isum2 - ml2 * xsum2);
                shift += 2;
                xq_idx++;
            }
            q_idx += 32;
        }
        sum += blockSum;
    }
    return sum;
}

// Q8_0 input dot product for Q3_K weights
static inline float vec_dot_q3_K_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q3_K* y = (const block_q3_K*)vy;
    int nb = n / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    float sum = 0.0f;
    int xq_idx = 0;
    for (int i = 0; i < nb; i++) {
        float d_all = fp16_to_fp32(y[i].d);
        const uint8_t* q = y[i].qs;
        const uint8_t* hm = y[i].hmask;

        uint32_t aux[4];
        const int8_t* scales = (const int8_t*)aux;
        memcpy(aux, y[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        uint8_t m = 1;
        int q_idx = 0;
        float blockSum = 0.0f;
        for (int nOuter = 0; nOuter < 256; nOuter += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dx = fp16_to_fp32(xq[xq_idx].d);
                const int8_t* xqs = xq[xq_idx].qs;
                float dl = d_all * (scales[is++] - 32);
                int isum1 = 0;
                for (int l = 0; l < 16; l++) {
                    int qv = ((q[q_idx + l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    isum1 += xqs[l] * qv;
                }
                float dl2 = d_all * (scales[is++] - 32);
                int isum2 = 0;
                for (int l = 0; l < 16; l++) {
                    int qv = ((q[q_idx + l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    isum2 += xqs[16 + l] * qv;
                }
                blockSum += dx * (dl * isum1 + dl2 * isum2);
                shift += 2;
                m <<= 1;
                xq_idx++;
            }
            q_idx += 32;
        }
        sum += blockSum;
    }
    return sum;
}

// Q8_0 input dot product for Q4_K weights
static inline float vec_dot_q4_K_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q4_K* y = (const block_q4_K*)vy;
    int nb = n / QK_K;
    float sum = 0.0f;
    int xq_idx = 0;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(y[i].d);
        float dmin = fp16_to_fp32(y[i].dmin);
        const uint8_t* q = y[i].qs;
        int is = 0;
        float blockSum = 0.0f;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m_val;
            get_scale_min_k4(is + 0, y[i].scales, &sc, &m_val);
            float d1 = d * sc;
            float dm1 = dmin * m_val;
            get_scale_min_k4(is + 1, y[i].scales, &sc, &m_val);
            float d2 = d * sc;
            float dm2 = dmin * m_val;

            float dx0 = fp16_to_fp32(xq[xq_idx].d);
            const int8_t* xqs0 = xq[xq_idx].qs;
            float dx1 = fp16_to_fp32(xq[xq_idx + 1].d);
            const int8_t* xqs1 = xq[xq_idx + 1].qs;
            int isum0 = 0, xsum0 = 0, isum1 = 0, xsum1 = 0;
            for (int l = 0; l < 32; l++) {
                uint8_t qByte = q[l];
                isum0 += xqs0[l] * (qByte & 0xF);
                xsum0 += xqs0[l];
                isum1 += xqs1[l] * (qByte >> 4);
                xsum1 += xqs1[l];
            }
            blockSum += d1 * dx0 * isum0 - dm1 * dx0 * xsum0
                      + d2 * dx1 * isum1 - dm2 * dx1 * xsum1;
            q += 32;
            is += 2;
            xq_idx += 2;
        }
        sum += blockSum;
    }
    return sum;
}

// Q8_0 input dot product for Q5_K weights
static inline float vec_dot_q5_K_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q5_K* y = (const block_q5_K*)vy;
    int nb = n / QK_K;
    float sum = 0.0f;
    int xq_idx = 0;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(y[i].d);
        float dmin = fp16_to_fp32(y[i].dmin);
        const uint8_t* ql = y[i].qs;
        const uint8_t* qh = y[i].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        int ql_idx = 0;
        float blockSum = 0.0f;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m_val;
            get_scale_min_k4(is + 0, y[i].scales, &sc, &m_val);
            float d1 = d * sc;
            float m1 = dmin * m_val;
            get_scale_min_k4(is + 1, y[i].scales, &sc, &m_val);
            float d2 = d * sc;
            float m2 = dmin * m_val;

            float dx0 = fp16_to_fp32(xq[xq_idx].d);
            const int8_t* xqs0 = xq[xq_idx].qs;
            float dx1 = fp16_to_fp32(xq[xq_idx + 1].d);
            const int8_t* xqs1 = xq[xq_idx + 1].qs;
            int isum0 = 0, xsum0 = 0, isum1 = 0, xsum1 = 0;
            for (int l = 0; l < 32; l++) {
                int qw_lo = (ql[ql_idx + l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                int qw_hi = (ql[ql_idx + l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                isum0 += xqs0[l] * qw_lo;
                xsum0 += xqs0[l];
                isum1 += xqs1[l] * qw_hi;
                xsum1 += xqs1[l];
            }
            blockSum += d1 * dx0 * isum0 - m1 * dx0 * xsum0
                      + d2 * dx1 * isum1 - m2 * dx1 * xsum1;
            ql_idx += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
            xq_idx += 2;
        }
        sum += blockSum;
    }
    return sum;
}

// Q8_0 input dot product for Q6_K weights
static inline float vec_dot_q6_K_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_q6_K* y = (const block_q6_K*)vy;
    int nb = n / QK_K;
    float sum = 0.0f;
    int xq_idx = 0;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(y[i].d);
        const uint8_t* ql = y[i].ql;
        const uint8_t* qh = y[i].qh;
        const int8_t* sc = y[i].scales;
        float blockSum = 0.0f;
        for (int nOuter = 0; nOuter < 256; nOuter += 128) {
            int sc_base = (nOuter >> 7) * 8;
            int ql_base = nOuter >> 1;
            int qh_base = nOuter >> 2;
            float dx0 = fp16_to_fp32(xq[xq_idx].d);
            const int8_t* xqs0 = xq[xq_idx].qs;
            float dx1 = fp16_to_fp32(xq[xq_idx + 1].d);
            const int8_t* xqs1 = xq[xq_idx + 1].qs;
            float dx2 = fp16_to_fp32(xq[xq_idx + 2].d);
            const int8_t* xqs2 = xq[xq_idx + 2].qs;
            float dx3 = fp16_to_fp32(xq[xq_idx + 3].d);
            const int8_t* xqs3 = xq[xq_idx + 3].qs;
            float ds0 = d * sc[sc_base + 0], ds1 = d * sc[sc_base + 1];
            float ds2 = d * sc[sc_base + 2], ds3 = d * sc[sc_base + 3];
            float ds4 = d * sc[sc_base + 4], ds5 = d * sc[sc_base + 5];
            float ds6 = d * sc[sc_base + 6], ds7 = d * sc[sc_base + 7];
            int isum0a = 0, isum1a = 0, isum2a = 0, isum3a = 0;
            for (int l = 0; l < 16; l++) {
                int q1 = ((ql[ql_base + l] & 0xF) | (((qh[qh_base + l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[ql_base + l + 32] & 0xF) | (((qh[qh_base + l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) - 32;
                isum0a += xqs0[l] * q1;
                isum1a += xqs1[l] * q2;
                isum2a += xqs2[l] * q3;
                isum3a += xqs3[l] * q4;
            }
            int isum0b = 0, isum1b = 0, isum2b = 0, isum3b = 0;
            for (int l = 16; l < 32; l++) {
                int q1 = ((ql[ql_base + l] & 0xF) | (((qh[qh_base + l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[ql_base + l + 32] & 0xF) | (((qh[qh_base + l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) - 32;
                isum0b += xqs0[l] * q1;
                isum1b += xqs1[l] * q2;
                isum2b += xqs2[l] * q3;
                isum3b += xqs3[l] * q4;
            }
            blockSum += dx0 * (ds0 * isum0a + ds1 * isum0b)
                      + dx1 * (ds2 * isum1a + ds3 * isum1b)
                      + dx2 * (ds4 * isum2a + ds5 * isum2b)
                      + dx3 * (ds6 * isum3a + ds7 * isum3b);
            xq_idx += 4;
        }
        sum += blockSum;
    }
    return sum;
}

// Q8_0 input dot product for IQ4_NL weights
static inline float vec_dot_iq4_nl_q8_0(const block_q8_0* xq, const void* vy, int n) {
    const block_iq4_nl* y = (const block_iq4_nl*)vy;
    int nb = n / QK4_NL;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float dw = fp16_to_fp32(y[i].d);
        float dx = fp16_to_fp32(xq[i].d);
        const uint8_t* qs = y[i].qs;
        const int8_t* xqs = xq[i].qs;
        int isum = 0;
        for (int j = 0; j < 16; j++) {
            uint8_t qByte = qs[j];
            isum += xqs[j] * kvalues_iq4nl[qByte & 0xF]
                  + xqs[j + 16] * kvalues_iq4nl[qByte >> 4];
        }
        sum += dw * dx * (float)isum;
    }
    return sum;
}

// Function pointer type for Q8_0 input dot products
typedef float (*vec_dot_q8_func)(const block_q8_0*, const void*, int);

// Get the appropriate Q8_0 input dot function for a quantization type
// Returns NULL for types that don't benefit from Q8_0 input (F16, BF16, F32)
static inline vec_dot_q8_func get_vec_dot_q8_func(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:  return vec_dot_q4_0_q8_0;
        case GGML_TYPE_Q4_1:  return vec_dot_q4_1_q8_0;
        case GGML_TYPE_Q5_0:  return vec_dot_q5_0_q8_0;
        case GGML_TYPE_Q5_1:  return vec_dot_q5_1_q8_0;
        case GGML_TYPE_Q8_0:  return vec_dot_q8_0_q8_0;
        case GGML_TYPE_Q2_K:  return vec_dot_q2_K_q8_0;
        case GGML_TYPE_Q3_K:  return vec_dot_q3_K_q8_0;
        case GGML_TYPE_Q4_K:  return vec_dot_q4_K_q8_0;
        case GGML_TYPE_Q5_K:  return vec_dot_q5_K_q8_0;
        case GGML_TYPE_Q6_K:  return vec_dot_q6_K_q8_0;
        case GGML_TYPE_IQ4_NL: return vec_dot_iq4_nl_q8_0;
        default:              return NULL;
    }
}

// Function pointer type for vec_dot functions
typedef float (*vec_dot_func)(const float*, const void*, int);

// Get the appropriate vec_dot function for a quantization type
static inline vec_dot_func get_vec_dot_func(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:  return vec_dot_q4_0;
        case GGML_TYPE_Q4_1:  return vec_dot_q4_1;
        case GGML_TYPE_Q5_0:  return vec_dot_q5_0;
        case GGML_TYPE_Q5_1:  return vec_dot_q5_1;
        case GGML_TYPE_Q8_0:  return vec_dot_q8_0;
        case GGML_TYPE_Q2_K:  return vec_dot_q2_K;
        case GGML_TYPE_Q3_K:  return vec_dot_q3_K;
        case GGML_TYPE_Q4_K:  return vec_dot_q4_K;
        case GGML_TYPE_Q5_K:  return vec_dot_q5_K;
        case GGML_TYPE_Q6_K:  return vec_dot_q6_K;
        case GGML_TYPE_IQ4_NL: return vec_dot_iq4_nl;
        case GGML_TYPE_F16:   return vec_dot_f16;
        case GGML_TYPE_BF16:
        case 30:              return vec_dot_bf16;
        case GGML_TYPE_F32:   return vec_dot_f32;
        default:              return NULL;
    }
}

// Get row size in bytes for a quantized type
static inline size_t get_row_size(int n_cols, enum ggml_type type) {
    int block_size = get_block_size(type);
    size_t type_size = get_type_size(type);
    return (n_cols / block_size) * type_size;
}

// Fused quantized matrix-vector multiplication
// Computes xout = W @ x where W is quantized (d rows, n cols)
// W is stored row-major: each row has n elements in quantized form
// Uses Q8_0 quantized input path when available for integer dot products
void matmul_quantized(float* xout, const float* x, const QuantizedTensor* qw) {
    int d = qw->rows;
    int n = qw->cols;
    enum ggml_type type = qw->type;
    size_t row_size = get_row_size(n, type);
    const char* data = (const char*)qw->data;

    // Try Q8_0 quantized input path (integer dot products - faster)
    vec_dot_q8_func q8_func = get_vec_dot_q8_func(type);
    if (q8_func && q8_buf) {
        // Quantize x to Q8_0 once before the parallel loop
        quantize_row_q8_0(x, q8_buf, n);

        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            const void* row = data + i * row_size;
            xout[i] = q8_func(q8_buf, row, n);
        }
        return;
    }

    // Fallback: float dot products (for F16, BF16, F32 weights)
    vec_dot_func dot_func = get_vec_dot_func(type);
    if (!dot_func) {
        fprintf(stderr, "Unsupported quantization type in matmul: %d\n", type);
        exit(1);
    }

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        const void* row = data + i * row_size;
        xout[i] = dot_func(x, row, n);
    }
}

// ----------------------------------------------------------------------------
// GGUF parsing

typedef struct {
    char* key;
    enum gguf_type type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        uint64_t u64;
        int64_t  i64;
        float    f32;
        double   f64;
        bool     b;
        char*    str;
    } value;
} GGUFKeyValue;

typedef struct {
    char* name;
    uint32_t n_dims;
    uint64_t ne[4];  // dimensions
    enum ggml_type type;
    uint64_t offset;  // offset from start of tensor data
    void* data;       // pointer to tensor data after loading
} GGUFTensor;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    GGUFKeyValue* kv;
    GGUFTensor* tensors;
    void* tensor_data_start;
    // Tokenizer vocabulary from GGUF
    char** vocab_tokens;
    float* vocab_scores;
    uint64_t vocab_size;
} GGUFFile;

// Helper to read a string from GGUF file
char* read_gguf_string(FILE* f) {
    uint64_t len;
    if (fread(&len, sizeof(uint64_t), 1, f) != 1) return NULL;

    char* str = malloc(len + 1);
    if (!str) return NULL;

    if (fread(str, 1, len, f) != len) {
        free(str);
        return NULL;
    }
    str[len] = '\0';
    return str;
}

// Read a GGUF value based on type
int read_gguf_value(FILE* f, enum gguf_type type, GGUFKeyValue* kv) {
    kv->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:
            return fread(&kv->value.u8, sizeof(uint8_t), 1, f) == 1;
        case GGUF_TYPE_INT8:
            return fread(&kv->value.i8, sizeof(int8_t), 1, f) == 1;
        case GGUF_TYPE_UINT16:
            return fread(&kv->value.u16, sizeof(uint16_t), 1, f) == 1;
        case GGUF_TYPE_INT16:
            return fread(&kv->value.i16, sizeof(int16_t), 1, f) == 1;
        case GGUF_TYPE_UINT32:
            return fread(&kv->value.u32, sizeof(uint32_t), 1, f) == 1;
        case GGUF_TYPE_INT32:
            return fread(&kv->value.i32, sizeof(int32_t), 1, f) == 1;
        case GGUF_TYPE_UINT64:
            return fread(&kv->value.u64, sizeof(uint64_t), 1, f) == 1;
        case GGUF_TYPE_INT64:
            return fread(&kv->value.i64, sizeof(int64_t), 1, f) == 1;
        case GGUF_TYPE_FLOAT32:
            return fread(&kv->value.f32, sizeof(float), 1, f) == 1;
        case GGUF_TYPE_FLOAT64:
            return fread(&kv->value.f64, sizeof(double), 1, f) == 1;
        case GGUF_TYPE_BOOL:
            return fread(&kv->value.b, sizeof(bool), 1, f) == 1;
        case GGUF_TYPE_STRING:
            kv->value.str = read_gguf_string(f);
            return kv->value.str != NULL;
        case GGUF_TYPE_ARRAY:
            // Skip arrays for now (used for tokenizer)
            {
                uint32_t arr_type;
                uint64_t arr_len;
                if (fread(&arr_type, sizeof(uint32_t), 1, f) != 1) return 0;
                if (fread(&arr_len, sizeof(uint64_t), 1, f) != 1) return 0;
                // Skip array contents
                for (uint64_t i = 0; i < arr_len; i++) {
                    GGUFKeyValue dummy;
                    if (!read_gguf_value(f, arr_type, &dummy)) return 0;
                    if (arr_type == GGUF_TYPE_STRING) free(dummy.value.str);
                }
            }
            return 1;
        default:
            return 0;
    }
}

// Find a key-value pair by key
GGUFKeyValue* find_gguf_kv(GGUFFile* gguf, const char* key) {
    for (uint64_t i = 0; i < gguf->n_kv; i++) {
        if (gguf->kv[i].key && strcmp(gguf->kv[i].key, key) == 0) {
            return &gguf->kv[i];
        }
    }
    return NULL;
}

// Find a tensor by name
GGUFTensor* find_gguf_tensor(GGUFFile* gguf, const char* name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (gguf->tensors[i].name && strcmp(gguf->tensors[i].name, name) == 0) {
            return &gguf->tensors[i];
        }
    }
    return NULL;
}

// Get integer value from key-value
int64_t get_gguf_int(GGUFFile* gguf, const char* key, int64_t default_val) {
    GGUFKeyValue* kv = find_gguf_kv(gguf, key);
    if (!kv) return default_val;

    switch (kv->type) {
        case GGUF_TYPE_UINT8:  return kv->value.u8;
        case GGUF_TYPE_INT8:   return kv->value.i8;
        case GGUF_TYPE_UINT16: return kv->value.u16;
        case GGUF_TYPE_INT16:  return kv->value.i16;
        case GGUF_TYPE_UINT32: return kv->value.u32;
        case GGUF_TYPE_INT32:  return kv->value.i32;
        case GGUF_TYPE_UINT64: return kv->value.u64;
        case GGUF_TYPE_INT64:  return kv->value.i64;
        default: return default_val;
    }
}

// Get float value from key-value
float get_gguf_float(GGUFFile* gguf, const char* key, float default_val) {
    GGUFKeyValue* kv = find_gguf_kv(gguf, key);
    if (!kv) return default_val;

    switch (kv->type) {
        case GGUF_TYPE_FLOAT32: return kv->value.f32;
        case GGUF_TYPE_FLOAT64: return (float)kv->value.f64;
        default: return default_val;
    }
}

// Get string value from key-value
const char* get_gguf_string(GGUFFile* gguf, const char* key) {
    GGUFKeyValue* kv = find_gguf_kv(gguf, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return NULL;
    return kv->value.str;
}

// Parse GGUF file header and metadata
GGUFFile* parse_gguf_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }

    // Read magic number
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Failed to read magic number\n");
        fclose(f);
        return NULL;
    }

    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: expected 0x%X, got 0x%X\n", GGUF_MAGIC, magic);
        fclose(f);
        return NULL;
    }

    GGUFFile* gguf = calloc(1, sizeof(GGUFFile));
    if (!gguf) {
        fclose(f);
        return NULL;
    }

    // Read version
    if (fread(&gguf->version, sizeof(uint32_t), 1, f) != 1) {
        free(gguf);
        fclose(f);
        return NULL;
    }

    if (gguf->version < 2 || gguf->version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %d\n", gguf->version);
        free(gguf);
        fclose(f);
        return NULL;
    }

    // Read tensor count and kv count
    if (fread(&gguf->n_tensors, sizeof(uint64_t), 1, f) != 1) {
        free(gguf);
        fclose(f);
        return NULL;
    }
    if (fread(&gguf->n_kv, sizeof(uint64_t), 1, f) != 1) {
        free(gguf);
        fclose(f);
        return NULL;
    }

    if (debug_mode) {
        printf("GGUF version: %d, tensors: %llu, kv pairs: %llu\n",
               gguf->version, (unsigned long long)gguf->n_tensors, (unsigned long long)gguf->n_kv);
    }

    // Allocate and read key-value pairs
    gguf->kv = calloc(gguf->n_kv, sizeof(GGUFKeyValue));
    for (uint64_t i = 0; i < gguf->n_kv; i++) {
        gguf->kv[i].key = read_gguf_string(f);
        if (!gguf->kv[i].key) {
            fprintf(stderr, "Failed to read key %llu\n", (unsigned long long)i);
            // Cleanup and return
            fclose(f);
            return NULL;
        }

        uint32_t type;
        if (fread(&type, sizeof(uint32_t), 1, f) != 1) {
            fprintf(stderr, "Failed to read type for key: %s\n", gguf->kv[i].key);
            fclose(f);
            return NULL;
        }

        if (!read_gguf_value(f, type, &gguf->kv[i])) {
            fprintf(stderr, "Failed to read value for key: %s\n", gguf->kv[i].key);
            fclose(f);
            return NULL;
        }
    }

    // Allocate and read tensor info
    gguf->tensors = calloc(gguf->n_tensors, sizeof(GGUFTensor));
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf->tensors[i].name = read_gguf_string(f);
        if (!gguf->tensors[i].name) {
            fprintf(stderr, "Failed to read tensor name %llu\n", (unsigned long long)i);
            fclose(f);
            return NULL;
        }

        if (fread(&gguf->tensors[i].n_dims, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            return NULL;
        }

        for (uint32_t j = 0; j < gguf->tensors[i].n_dims; j++) {
            if (fread(&gguf->tensors[i].ne[j], sizeof(uint64_t), 1, f) != 1) {
                fclose(f);
                return NULL;
            }
        }

        uint32_t type;
        if (fread(&type, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        gguf->tensors[i].type = type;

        if (fread(&gguf->tensors[i].offset, sizeof(uint64_t), 1, f) != 1) {
            fclose(f);
            return NULL;
        }
    }

    // Calculate alignment and tensor data start
    long header_end = ftell(f);
    uint64_t alignment = get_gguf_int(gguf, "general.alignment", 32);
    long tensor_data_offset = ((header_end + alignment - 1) / alignment) * alignment;

    fclose(f);

    // Memory map the file for tensor data access
#if defined _WIN32
    int fd = open(filename, O_RDONLY | _O_BINARY);
#else
    int fd = open(filename, O_RDONLY);
#endif
    if (fd == -1) {
        fprintf(stderr, "Failed to open file for mmap\n");
        return NULL;
    }

    // Get file size
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    mapped_size = file_size;
    close(fd);

    if (mapped_data == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap file\n");
        mapped_data = NULL;
        return NULL;
    }

    gguf->tensor_data_start = (char*)mapped_data + tensor_data_offset;

    // Set data pointers for all tensors
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        gguf->tensors[i].data = (char*)gguf->tensor_data_start + gguf->tensors[i].offset;
    }

    // Initialize vocab pointers to NULL
    gguf->vocab_tokens = NULL;
    gguf->vocab_scores = NULL;
    gguf->vocab_size = 0;

    return gguf;
}

// Load tokenizer vocabulary from GGUF file
// This re-reads the file to extract the tokenizer.ggml.tokens and tokenizer.ggml.scores arrays
int load_gguf_tokenizer(GGUFFile* gguf, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;

    // Skip magic and version
    fseek(f, 8, SEEK_SET);

    uint64_t n_tensors, n_kv;
    if (fread(&n_tensors, sizeof(uint64_t), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&n_kv, sizeof(uint64_t), 1, f) != 1) { fclose(f); return 0; }

    // Scan through key-value pairs looking for tokenizer arrays
    for (uint64_t i = 0; i < n_kv; i++) {
        char* key = read_gguf_string(f);
        if (!key) { fclose(f); return 0; }

        uint32_t type;
        if (fread(&type, sizeof(uint32_t), 1, f) != 1) { free(key); fclose(f); return 0; }

        if (type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type;
            uint64_t arr_len;
            if (fread(&arr_type, sizeof(uint32_t), 1, f) != 1) { free(key); fclose(f); return 0; }
            if (fread(&arr_len, sizeof(uint64_t), 1, f) != 1) { free(key); fclose(f); return 0; }

            if (strcmp(key, "tokenizer.ggml.tokens") == 0 && arr_type == GGUF_TYPE_STRING) {
                // Read token strings
                if (debug_mode) {
                    printf("Loading vocabulary from GGUF: %llu tokens\n", (unsigned long long)arr_len);
                }
                gguf->vocab_size = arr_len;
                gguf->vocab_tokens = (char**)malloc(arr_len * sizeof(char*));
                if (!gguf->vocab_tokens) { free(key); fclose(f); return 0; }

                for (uint64_t j = 0; j < arr_len; j++) {
                    gguf->vocab_tokens[j] = read_gguf_string(f);
                    if (!gguf->vocab_tokens[j]) {
                        // Cleanup on failure
                        for (uint64_t k = 0; k < j; k++) free(gguf->vocab_tokens[k]);
                        free(gguf->vocab_tokens);
                        gguf->vocab_tokens = NULL;
                        free(key);
                        fclose(f);
                        return 0;
                    }
                }
            } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && arr_type == GGUF_TYPE_FLOAT32) {
                // Read token scores
                if (!gguf->vocab_scores) {
                    gguf->vocab_scores = (float*)malloc(arr_len * sizeof(float));
                }
                if (gguf->vocab_scores) {
                    if (fread(gguf->vocab_scores, sizeof(float), arr_len, f) != arr_len) {
                        free(gguf->vocab_scores);
                        gguf->vocab_scores = NULL;
                    }
                }
            } else {
                // Skip other arrays
                for (uint64_t j = 0; j < arr_len; j++) {
                    switch (arr_type) {
                        case GGUF_TYPE_UINT8:  fseek(f, sizeof(uint8_t), SEEK_CUR); break;
                        case GGUF_TYPE_INT8:   fseek(f, sizeof(int8_t), SEEK_CUR); break;
                        case GGUF_TYPE_UINT16: fseek(f, sizeof(uint16_t), SEEK_CUR); break;
                        case GGUF_TYPE_INT16:  fseek(f, sizeof(int16_t), SEEK_CUR); break;
                        case GGUF_TYPE_UINT32: fseek(f, sizeof(uint32_t), SEEK_CUR); break;
                        case GGUF_TYPE_INT32:  fseek(f, sizeof(int32_t), SEEK_CUR); break;
                        case GGUF_TYPE_UINT64: fseek(f, sizeof(uint64_t), SEEK_CUR); break;
                        case GGUF_TYPE_INT64:  fseek(f, sizeof(int64_t), SEEK_CUR); break;
                        case GGUF_TYPE_FLOAT32: fseek(f, sizeof(float), SEEK_CUR); break;
                        case GGUF_TYPE_FLOAT64: fseek(f, sizeof(double), SEEK_CUR); break;
                        case GGUF_TYPE_BOOL:   fseek(f, sizeof(bool), SEEK_CUR); break;
                        case GGUF_TYPE_STRING: {
                            char* s = read_gguf_string(f);
                            free(s);
                            break;
                        }
                        default: break;
                    }
                }
            }
        } else {
            // Skip non-array values
            GGUFKeyValue dummy;
            read_gguf_value(f, type, &dummy);
            if (type == GGUF_TYPE_STRING) free(dummy.value.str);
        }
        free(key);
    }

    fclose(f);

    // If we didn't get scores, initialize them to 0
    if (gguf->vocab_tokens && !gguf->vocab_scores) {
        gguf->vocab_scores = (float*)calloc(gguf->vocab_size, sizeof(float));
    }

    return (gguf->vocab_tokens != NULL);
}

void free_gguf_file(GGUFFile* gguf) {
    if (!gguf) return;

    for (uint64_t i = 0; i < gguf->n_kv; i++) {
        free(gguf->kv[i].key);
        if (gguf->kv[i].type == GGUF_TYPE_STRING) {
            free(gguf->kv[i].value.str);
        }
    }
    free(gguf->kv);

    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        free(gguf->tensors[i].name);
    }
    free(gguf->tensors);

    // Free vocabulary if loaded
    if (gguf->vocab_tokens) {
        for (uint64_t i = 0; i < gguf->vocab_size; i++) {
            free(gguf->vocab_tokens[i]);
        }
        free(gguf->vocab_tokens);
    }
    free(gguf->vocab_scores);

    free(gguf);
}

// ----------------------------------------------------------------------------
// Load weights from GGUF into TransformerWeights

void track_allocation(float* ptr) {
    num_weight_allocations++;
    float** new_allocations = realloc(weight_allocations, num_weight_allocations * sizeof(float*));
    if (!new_allocations) {
        fprintf(stderr, "Failed to allocate memory for weight tracking (allocation #%d)\n", num_weight_allocations);
        exit(1);
    }
    weight_allocations = new_allocations;
    weight_allocations[num_weight_allocations - 1] = ptr;
}

void free_tracked_allocation(float* ptr) {
    if (!ptr) return;
    for (int i = num_weight_allocations - 1; i >= 0; i--) {
        if (weight_allocations[i] == ptr) {
            weight_allocations[i] = NULL;
            break;
        }
    }
    free(ptr);
}

// Load tensor as dequantized float (for small tensors like norms and embeddings)
float* load_tensor_float(GGUFFile* gguf, const char* name, int expected_elements) {
    GGUFTensor* tensor = find_gguf_tensor(gguf, name);
    if (!tensor) {
        fprintf(stderr, "Warning: tensor not found: %s\n", name);
        return NULL;
    }

    // Calculate total elements
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        n_elements *= tensor->ne[i];
    }

    if (expected_elements > 0 && n_elements != (uint64_t)expected_elements) {
        fprintf(stderr, "Warning: tensor %s has %llu elements, expected %d\n",
                name, (unsigned long long)n_elements, expected_elements);
    }

    // Dequantize if needed
    float* data;
    if (tensor->type == GGML_TYPE_F32) {
        // Already float32, just copy
        data = malloc(n_elements * sizeof(float));
        if (data) {
            memcpy(data, tensor->data, n_elements * sizeof(float));
            track_allocation(data);
        }
    } else {
        data = dequantize_tensor(tensor->data, n_elements, tensor->type);
        if (data) track_allocation(data);
    }

    return data;
}

// Load tensor keeping it in quantized form (for large weight matrices)
// Returns 1 on success, 0 on failure
int load_tensor_quantized(GGUFFile* gguf, const char* name, QuantizedTensor* qt, int rows, int cols) {
    GGUFTensor* tensor = find_gguf_tensor(gguf, name);
    if (!tensor) {
        fprintf(stderr, "Warning: tensor not found: %s\n", name);
        qt->data = NULL;
        return 0;
    }

    // Calculate total elements
    uint64_t n_elements = 1;
    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        n_elements *= tensor->ne[i];
    }

    // Verify dimensions match GGUF tensor (GGUF: ne[0]=cols, ne[1]=rows)
    if (debug_mode && tensor->n_dims >= 2) {
        int gguf_cols = (int)tensor->ne[0];
        int gguf_rows = (int)tensor->ne[1];
        if (gguf_rows != rows || gguf_cols != cols) {
            printf("[DEBUG] Tensor %s: expected (%d, %d), GGUF has (%d, %d)\n",
                   name, rows, cols, gguf_rows, gguf_cols);
        }
    }

    // Store quantized data directly - points into mmap'd file!
    qt->data = tensor->data;
    qt->type = tensor->type;
    qt->n_elements = (int)n_elements;
    qt->rows = rows;
    qt->cols = cols;

    return 1;
}

// Load layer-specific tensor as dequantized float
float* load_layer_tensor_float(GGUFFile* gguf, int layer, const char* suffix, int expected_elements) {
    char name[256];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return load_tensor_float(gguf, name, expected_elements);
}

// Load layer-specific tensor keeping it quantized
int load_layer_tensor_quantized(GGUFFile* gguf, int layer, const char* suffix, QuantizedTensor* qt, int rows, int cols) {
    char name[256];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return load_tensor_quantized(gguf, name, qt, rows, cols);
}

int init_weights_from_gguf(GGUFFile* gguf, Config* p, TransformerWeights* w) {
    // Use explicit head_dim for Gemma3, computed for others
    int head_size = p->head_dim > 0 ? p->head_dim : p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    int q_dim = p->n_heads * head_size;  // total Q dimension
    int n_layers = p->n_layers;

    // Token embeddings - must be dequantized for fast lookup
    w->token_embedding_table = load_tensor_float(gguf, "token_embd.weight", p->vocab_size * p->dim);
    if (!w->token_embedding_table) {
        fprintf(stderr, "Failed to load token embeddings\n");
        return 0;
    }

    // RMS norm weights - small, keep as float
    w->rms_att_weight = malloc(n_layers * p->dim * sizeof(float));
    w->rms_ffn_weight = malloc(n_layers * p->dim * sizeof(float));

    if (!w->rms_att_weight || !w->rms_ffn_weight) {
        fprintf(stderr, "Failed to allocate norm weight arrays\n");
        return 0;
    }

    // Allocate arrays of QuantizedTensor for each layer's weights
    // These are MUCH smaller than before - just pointers + metadata, not dequantized data!
    w->wq = calloc(n_layers, sizeof(QuantizedTensor));
    w->wk = calloc(n_layers, sizeof(QuantizedTensor));
    w->wv = calloc(n_layers, sizeof(QuantizedTensor));
    w->wo = calloc(n_layers, sizeof(QuantizedTensor));
    w->w1 = calloc(n_layers, sizeof(QuantizedTensor));
    w->w2 = calloc(n_layers, sizeof(QuantizedTensor));
    w->w3 = calloc(n_layers, sizeof(QuantizedTensor));

    if (!w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3) {
        fprintf(stderr, "Failed to allocate quantized weight arrays\n");
        return 0;
    }

    // Initialize Gemma3-specific weight pointers to NULL
    w->attn_q_norm = NULL;
    w->attn_k_norm = NULL;
    w->attn_post_norm = NULL;
    w->ffn_post_norm = NULL;

    // Allocate Gemma3-specific weight arrays if needed (small, keep as float)
    if (p->is_gemma3) {
        w->attn_q_norm = malloc(n_layers * head_size * sizeof(float));
        w->attn_k_norm = malloc(n_layers * head_size * sizeof(float));
        w->attn_post_norm = malloc(n_layers * p->dim * sizeof(float));
        w->ffn_post_norm = malloc(n_layers * p->dim * sizeof(float));

        if (!w->attn_q_norm || !w->attn_k_norm || !w->attn_post_norm || !w->ffn_post_norm) {
            fprintf(stderr, "Failed to allocate Gemma3-specific weight arrays\n");
            return 0;
        }
    }

    // Load per-layer weights
    for (int l = 0; l < n_layers; l++) {
        // Attention norm - small, dequantize
        float* attn_norm = load_layer_tensor_float(gguf, l, "attn_norm.weight", p->dim);
        if (attn_norm) {
            memcpy(w->rms_att_weight + l * p->dim, attn_norm, p->dim * sizeof(float));
            free_tracked_allocation(attn_norm);
        }

        // FFN norm - small, dequantize
        float* ffn_norm = load_layer_tensor_float(gguf, l, "ffn_norm.weight", p->dim);
        if (ffn_norm) {
            memcpy(w->rms_ffn_weight + l * p->dim, ffn_norm, p->dim * sizeof(float));
            free_tracked_allocation(ffn_norm);
        }

        // Q, K, V, O projections - KEEP QUANTIZED!
        // Weight matrices are (out_dim, in_dim) = (rows, cols)
        load_layer_tensor_quantized(gguf, l, "attn_q.weight", &w->wq[l], q_dim, p->dim);
        load_layer_tensor_quantized(gguf, l, "attn_k.weight", &w->wk[l], kv_dim, p->dim);
        load_layer_tensor_quantized(gguf, l, "attn_v.weight", &w->wv[l], kv_dim, p->dim);
        load_layer_tensor_quantized(gguf, l, "attn_output.weight", &w->wo[l], p->dim, q_dim);

        // FFN weights - KEEP QUANTIZED!
        load_layer_tensor_quantized(gguf, l, "ffn_gate.weight", &w->w1[l], p->hidden_dim, p->dim);
        load_layer_tensor_quantized(gguf, l, "ffn_down.weight", &w->w2[l], p->dim, p->hidden_dim);
        load_layer_tensor_quantized(gguf, l, "ffn_up.weight", &w->w3[l], p->hidden_dim, p->dim);

        // Load Gemma3-specific per-layer weights (small, dequantize)
        if (p->is_gemma3) {
            float* q_norm = load_layer_tensor_float(gguf, l, "attn_q_norm.weight", head_size);
            float* k_norm = load_layer_tensor_float(gguf, l, "attn_k_norm.weight", head_size);

            if (q_norm) {
                memcpy(w->attn_q_norm + l * head_size, q_norm, head_size * sizeof(float));
                free_tracked_allocation(q_norm);
            }
            if (k_norm) {
                memcpy(w->attn_k_norm + l * head_size, k_norm, head_size * sizeof(float));
                free_tracked_allocation(k_norm);
            }

            float* post_attn_norm = load_layer_tensor_float(gguf, l, "post_attention_norm.weight", p->dim);
            if (post_attn_norm) {
                memcpy(w->attn_post_norm + l * p->dim, post_attn_norm, p->dim * sizeof(float));
                free_tracked_allocation(post_attn_norm);
            }

            float* post_ffn_norm = load_layer_tensor_float(gguf, l, "post_ffw_norm.weight", p->dim);
            if (post_ffn_norm) {
                memcpy(w->ffn_post_norm + l * p->dim, post_ffn_norm, p->dim * sizeof(float));
                free_tracked_allocation(post_ffn_norm);
            }
        }
    }

    // Final norm - small, dequantize
    w->rms_final_weight = load_tensor_float(gguf, "output_norm.weight", p->dim);

    // Output projection (may be tied to embeddings)
    GGUFTensor* output_tensor = find_gguf_tensor(gguf, "output.weight");
    if (output_tensor) {
        // Load as quantized tensor
        load_tensor_quantized(gguf, "output.weight", &w->wcls, p->vocab_size, p->dim);


    } else {
        // Use tied embeddings - create a "fake" quantized tensor pointing to embeddings
        // The wcls will be used with matmul_f32 instead of matmul_quantized
        if (debug_mode) {
            printf("Using tied embeddings for output projection\n");
        }
        w->wcls.data = w->token_embedding_table;
        w->wcls.type = GGML_TYPE_F32;
        w->wcls.n_elements = p->vocab_size * p->dim;
        w->wcls.rows = p->vocab_size;
        w->wcls.cols = p->dim;


    }

    return 1;
}

// ----------------------------------------------------------------------------
// Fast math approximations (from JS engine)

// Fast tanh using Padé [3,3] rational approximation
// Accurate to ~1e-7 for |x| < 4, exact ±1 beyond
static inline float fast_tanhf(float x) {
    if (x < -4.0f) return -1.0f;
    if (x > 4.0f) return 1.0f;
    float x2 = x * x;
    return (x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)))) /
           (135135.0f + x2 * (62370.0f + x2 * (3150.0f + 28.0f * x2)));
}

// Fast exp using Padé [3,3] approximation for sigmoid range
static inline float fast_expf(float x) {
    if (x > 10.0f) return 22026.47f;
    if (x < -10.0f) return 0.0f;
    float x2 = x * x;
    float x3 = x2 * x;
    float num = 120.0f + 60.0f * x + 12.0f * x2 + x3;
    float den = 120.0f - 60.0f * x + 12.0f * x2 - x3;
    return num / den;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

// RMS norm with configurable epsilon (for Gemma3)
// NOTE: The GGUF conversion script already adds +1 to Gemma norm weights,
// so we just do standard multiplication here (no need to add +1 at runtime)
void rmsnorm_gemma(float* o, float* x, float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);  // Weights already have +1 baked in from GGUF conversion
    }
}

// Apply RMS norm per head (for Gemma3 Q/K normalization)
// input: (n_heads, head_size), weight: (head_size,)
// NOTE: The GGUF conversion script already adds +1 to Gemma norm weights
void rmsnorm_per_head_gemma(float* o, float* x, float* weight, int n_heads, int head_size, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float* head_x = x + h * head_size;
        float* head_o = o + h * head_size;

        float ss = 0.0f;
        for (int j = 0; j < head_size; j++) {
            ss += head_x[j] * head_x[j];
        }
        ss /= head_size;
        ss += eps;
        ss = 1.0f / sqrtf(ss);

        for (int j = 0; j < head_size; j++) {
            head_o[j] = weight[j] * (ss * head_x[j]);  // Weights already have +1 baked in
        }
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize using reciprocal multiplication (faster than division)
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// Transformer forward pass with GQA support and Gemma3 extensions
// compute_logits: 1 = compute logits (for sampling), 0 = skip (prompt prefill)
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w, int compute_logits) {
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    // Use cached values from RunState
    int head_size = s->head_size;
    int kv_dim = s->kv_dim;
    int q_dim = s->q_dim;
    int kv_mul = s->kv_mul;
    float eps = p->rms_norm_eps > 0 ? p->rms_norm_eps : 1e-5f;

    // copy the token embedding into x
    // GGUF stores embeddings with ne=[dim, vocab_size], meaning shape (vocab_size, dim) in row-major
    // Each token's embedding is contiguous: dim floats per token
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim * sizeof(*x));

    // Gemma3: Scale embeddings by sqrt(dim) - use pre-computed value
    if (p->is_gemma3) {
        float scale = s->embed_scale;
        for (int i = 0; i < dim; i++) {
            x[i] *= scale;
        }
    }

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        if (p->is_gemma3) {
            rmsnorm_gemma(s->xb, x, w->rms_att_weight + l * dim, dim, eps);
        } else {
            rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        }

        // qkv matmuls for this position - using quantized weights
        matmul_quantized(s->q, s->xb, &w->wq[l]);
        matmul_quantized(s->k, s->xb, &w->wk[l]);
        matmul_quantized(s->v, s->xb, &w->wv[l]);

        // Gemma3: Apply per-head Q and K normalization after projection
        if (p->is_gemma3 && w->attn_q_norm && w->attn_k_norm) {
            rmsnorm_per_head_gemma(s->q, s->q, w->attn_q_norm + l * head_size, p->n_heads, head_size, eps);
            rmsnorm_per_head_gemma(s->k, s->k, w->attn_k_norm + l * head_size, p->n_kv_heads, head_size, eps);
        }

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // Gemma3 uses NEOX mode: pairs (i, i+half) instead of consecutive (i, i+1)
        // Use pre-computed frequencies and cache sin/cos per position
        //
        // For Gemma3 with SWA: different layers use different RoPE frequencies
        // SWA layers (most layers): use rope_freqs_swa (theta=10000)
        // Dense layers (every swa_pattern-th): use rope_freqs (theta=1000000)
        // Pattern 6: layers 0,1,2,3,4 are SWA, layer 5 is dense, 6,7,8,9,10 are SWA, layer 11 is dense, etc.
        int is_swa_layer = (p->swa_pattern > 0) && (l % p->swa_pattern < p->swa_pattern - 1);
        int half = head_size / 2;

        // Use pre-computed sin/cos for this position (O(1) lookup, no sin/cos computation)
        int rope_base = pos * half;
        float* rope_cos = (p->is_gemma3 && is_swa_layer) ? s->rope_cos_swa_all + rope_base : s->rope_cos_all + rope_base;
        float* rope_sin = (p->is_gemma3 && is_swa_layer) ? s->rope_sin_swa_all + rope_base : s->rope_sin_all + rope_base;

        if (p->is_gemma3) {
            // NEOX RoPE: rotate dimension i with dimension i + half
            // Fused with attention scaling (1/sqrt(head_dim)) to avoid separate Q scaling loop
            float attn_scale_fused = s->attn_scale;
            // Apply to Q (all heads) - fused with attention scaling
            for (int h = 0; h < p->n_heads; h++) {
                float* q_head = s->q + h * head_size;
                for (int i = 0; i < half; i++) {
                    float fcr = rope_cos[i];
                    float fci = rope_sin[i];
                    float v0 = q_head[i];
                    float v1 = q_head[i + half];
                    q_head[i]        = (v0 * fcr - v1 * fci) * attn_scale_fused;
                    q_head[i + half] = (v0 * fci + v1 * fcr) * attn_scale_fused;
                }
            }
            // Apply to K (kv heads) - no scaling needed for K
            for (int h = 0; h < p->n_kv_heads; h++) {
                float* k_head = s->k + h * head_size;
                for (int i = 0; i < half; i++) {
                    float fcr = rope_cos[i];
                    float fci = rope_sin[i];
                    float v0 = k_head[i];
                    float v1 = k_head[i + half];
                    k_head[i]        = v0 * fcr - v1 * fci;
                    k_head[i + half] = v0 * fci + v1 * fcr;
                }
            }
        } else {
            // Standard RoPE: rotate consecutive pairs (i, i+1)
            for (int i = 0; i < q_dim; i += 2) {
                int head_dim_idx = (i % head_size) / 2;  // index into rope_freqs
                float fcr = rope_cos[head_dim_idx];
                float fci = rope_sin[head_dim_idx];
                int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q and k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }

        // Gemma3: Q attention scaling is fused into the RoPE loop above

        // save key,value at this time step (pos) to our kv cache
        // Use pre-computed kv_cache_layer_size for layer offset
        int loff = l * s->kv_cache_layer_size;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention with GQA support
        // Pre-fetch attention scale for non-Gemma3 models
        float attn_scale_score = s->attn_scale;
        int is_gemma3 = p->is_gemma3;
        // SWA: limit attention to sliding window for SWA layers (Gemma3)
        int start_t = 0;
        if (is_gemma3 && is_swa_layer && p->swa_window > 0) {
            start_t = pos - p->swa_window + 1;
            if (start_t < 0) start_t = 0;
        }
        int att_size = pos - start_t + 1;
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // compute kv_head once per head (constant for all timesteps)
            int kv_head = h / kv_mul;
            int kv_head_offset = kv_head * head_size;
            // iterate over timesteps in attention range
            for (int t = start_t; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + kv_head_offset;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                // For Gemma3, Q is already scaled by 1/sqrt(head_dim), so no scaling here
                // For other models, scale by 1/sqrt(head_dim) using pre-computed value
                if (!is_gemma3) {
                    score *= attn_scale_score;
                }
                // save the score to the attention buffer (offset by start_t for SWA)
                att[t - start_t] = score;
            }

            // softmax the scores to get attention weights
            softmax(att, att_size);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = start_t; t <= pos; t++) {
                // get the attention weight for this timestep
                float a = att[t - start_t];
                // skip near-zero attention weights (avoids unnecessary computation)
                if (a > -1e-8f && a < 1e-8f) continue;
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + kv_head_offset;
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention - using quantized weights
        matmul_quantized(s->xb2, s->xb, &w->wo[l]);

        // Gemma3: Apply post-attention normalization
        if (p->is_gemma3 && w->attn_post_norm) {
            rmsnorm_gemma(s->xb2, s->xb2, w->attn_post_norm + l * dim, dim, eps);
        }

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        if (p->is_gemma3) {
            rmsnorm_gemma(s->xb, x, w->rms_ffn_weight + l * dim, dim, eps);
        } else {
            rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        }

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x) - using quantized weights
        matmul_quantized(s->hb, s->xb, &w->w1[l]);
        matmul_quantized(s->hb2, s->xb, &w->w3[l]);

        // Gemma3 uses GeGLU activation: gelu(gate) * up
        // hb = w1(x) = ffn_gate(x) = gate
        // hb2 = w3(x) = ffn_up(x) = up
        // Formula: gelu(gate) * up = gelu(hb) * hb2
        if (p->is_gemma3) {
            // GeGLU: gelu(hb) * hb2 = gelu(gate) * up
            // GELU formula: 0.5*x*(1 + tanh(sqrt(2/pi)*x*(1 + 0.044715*x^2)))
            for (int i = 0; i < hidden_dim; i++) {
                float x = s->hb[i];  // gate value
                float gelu_val = 0.5f * x * (1.0f + fast_tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
                s->hb[i] = gelu_val * s->hb2[i];  // gelu(gate) * up
            }
        } else {
            // SwiGLU: silu(hb) * hb2 = silu(w1(x)) * w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val = val * (1.0f / (1.0f + fast_expf(-val)));
                s->hb[i] = val * s->hb2[i];
            }
        }

        // final matmul to get the output of the ffn - using quantized weights
        matmul_quantized(s->xb, s->hb, &w->w2[l]);

        // Gemma3: Apply post-FFN normalization
        if (p->is_gemma3 && w->ffn_post_norm) {
            rmsnorm_gemma(s->xb, s->xb, w->ffn_post_norm + l * dim, dim, eps);
        }

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    if (p->is_gemma3) {
        rmsnorm_gemma(x, x, w->rms_final_weight, dim, eps);
    } else {
        rmsnorm(x, x, w->rms_final_weight, dim);
    }

    // classifier into logits - skip during prompt prefill for speed
    if (compute_logits) {
        matmul_quantized(s->logits, x, &w->wcls);

        // Gemma3: Apply final logit softcapping if enabled
        if (p->is_gemma3 && p->final_logit_softcapping > 0.0f) {
            float cap = p->final_logit_softcapping;
            for (int i = 0; i < p->vocab_size; i++) {
                s->logits[i] = cap * fast_tanhf(s->logits[i] / cap);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Tokenizer for GGUF models (tiktoken-compatible)

// Special token IDs
#define LLAMA3_BOS_TOKEN 128000        // <|begin_of_text|>
#define LLAMA3_EOS_TOKEN 128001        // <|end_of_text|>
#define LLAMA3_START_HEADER 128006     // <|start_header_id|>
#define LLAMA3_END_HEADER 128007       // <|end_header_id|>
#define LLAMA3_EOT 128009              // <|eot_id|>

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned int max_token_length;
    int bos_token;
    int eos_token;
    // Special tokens
    int start_header_token;
    int end_header_token;
    int eot_token;
} Tokenizer;

Tokenizer tokenizer;

// Decode token string to printable text
// Handles both tiktoken (BPE) and SentencePiece formats:
// Decodes token strings to printable text
// - tiktoken (Llama): uses OpenAI's bytes_to_unicode() mapping
// - SentencePiece (Gemma): ▁ (U+2581) -> space, rest is already UTF-8
// Returns a newly allocated string that must be freed by caller

// Build the tiktoken unicode-to-byte mapping table
// This is the inverse of OpenAI's bytes_to_unicode() function
static int tiktoken_initialized = 0;
static int tiktoken_map[512];  // Maps unicode codepoint -> byte value (-1 if not mapped)

void init_tiktoken_decode_map(void) {
    if (tiktoken_initialized) return;

    for (int i = 0; i < 512; i++) tiktoken_map[i] = -1;

    // OpenAI's bytes_to_unicode() mapping:
    // - Bytes 33-126 (! to ~) map to themselves
    // - Bytes 161-172 (¡ to ¬) map to themselves
    // - Bytes 174-255 (® to ÿ) map to themselves
    // - All other bytes (0-32, 127-160, 173) map to 256, 257, 258, ...
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            tiktoken_map[b] = b;
        } else {
            tiktoken_map[256 + n] = b;
            n++;
        }
    }

    tiktoken_initialized = 1;
}

// Forward declaration
extern int use_sentencepiece;

// Decode SentencePiece token: just replace ▁ with space
char* decode_sentencepiece(const char* str) {
    if (!str) return NULL;

    size_t len = strlen(str);
    char* result = malloc(len + 1);
    if (!result) return NULL;

    int out_idx = 0;
    const unsigned char* ptr = (const unsigned char*)str;

    while (*ptr) {
        // Check for ▁ (U+2581) which is 0xE2 0x96 0x81 in UTF-8
        if (ptr[0] == 0xE2 && ptr[1] == 0x96 && ptr[2] == 0x81) {
            result[out_idx++] = ' ';
            ptr += 3;
        } else {
            result[out_idx++] = *ptr;
            ptr++;
        }
    }

    result[out_idx] = '\0';
    return result;
}

// Decode tiktoken token: convert OpenAI's byte-to-unicode mapping back to bytes
char* decode_tiktoken_internal(const char* tiktoken_str) {
    if (!tiktoken_str) return NULL;

    init_tiktoken_decode_map();

    size_t len = strlen(tiktoken_str);
    char* result = malloc(len * 4 + 1);  // Enough for UTF-8 expansion
    if (!result) return NULL;

    int out_idx = 0;
    const unsigned char* ptr = (const unsigned char*)tiktoken_str;

    while (*ptr) {
        int codepoint;
        int char_len;

        // Decode UTF-8 to get codepoint
        if ((*ptr & 0x80) == 0) {
            codepoint = *ptr;
            char_len = 1;
        } else if ((*ptr & 0xE0) == 0xC0 && (ptr[1] & 0xC0) == 0x80) {
            codepoint = (*ptr & 0x1F) << 6;
            codepoint |= (ptr[1] & 0x3F);
            char_len = 2;
        } else if ((*ptr & 0xF0) == 0xE0 && (ptr[1] & 0xC0) == 0x80 && (ptr[2] & 0xC0) == 0x80) {
            codepoint = (*ptr & 0x0F) << 12;
            codepoint |= (ptr[1] & 0x3F) << 6;
            codepoint |= (ptr[2] & 0x3F);
            char_len = 3;
        } else if ((*ptr & 0xF8) == 0xF0 && (ptr[1] & 0xC0) == 0x80 && (ptr[2] & 0xC0) == 0x80 && (ptr[3] & 0xC0) == 0x80) {
            codepoint = (*ptr & 0x07) << 18;
            codepoint |= (ptr[1] & 0x3F) << 12;
            codepoint |= (ptr[2] & 0x3F) << 6;
            codepoint |= (ptr[3] & 0x3F);
            char_len = 4;
        } else {
            // Invalid UTF-8, skip byte
            ptr++;
            continue;
        }

        // Look up in tiktoken map
        if (codepoint < 512 && tiktoken_map[codepoint] >= 0) {
            result[out_idx++] = (char)tiktoken_map[codepoint];
        } else {
            // Unknown codepoint - encode as UTF-8
            if (codepoint < 0x80) {
                result[out_idx++] = (char)codepoint;
            } else if (codepoint < 0x800) {
                result[out_idx++] = (char)(0xC0 | (codepoint >> 6));
                result[out_idx++] = (char)(0x80 | (codepoint & 0x3F));
            } else if (codepoint < 0x10000) {
                result[out_idx++] = (char)(0xE0 | (codepoint >> 12));
                result[out_idx++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                result[out_idx++] = (char)(0x80 | (codepoint & 0x3F));
            } else {
                result[out_idx++] = (char)(0xF0 | (codepoint >> 18));
                result[out_idx++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
                result[out_idx++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                result[out_idx++] = (char)(0x80 | (codepoint & 0x3F));
            }
        }

        ptr += char_len;
    }

    result[out_idx] = '\0';
    return result;
}

// Main decode function - dispatches based on tokenizer type
char* decode_tiktoken(const char* str) {
    if (use_sentencepiece) {
        return decode_sentencepiece(str);
    } else {
        return decode_tiktoken_internal(str);
    }
}

// Build tiktoken byte-to-unicode mapping (OpenAI's bytes_to_unicode)
static int tiktoken_encode_initialized = 0;
static int tiktoken_byte_to_unicode[256];  // Maps byte value -> unicode codepoint

void init_tiktoken_encode_map(void) {
    if (tiktoken_encode_initialized) return;

    // OpenAI's bytes_to_unicode() mapping:
    // - Bytes 33-126 (! to ~) map to themselves
    // - Bytes 161-172 (¡ to ¬) map to themselves
    // - Bytes 174-255 (® to ÿ) map to themselves
    // - All other bytes (0-32, 127-160, 173) map to 256, 257, 258, ...
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            tiktoken_byte_to_unicode[b] = b;
        } else {
            tiktoken_byte_to_unicode[b] = 256 + n;
            n++;
        }
    }

    tiktoken_encode_initialized = 1;
}

// Helper to write a unicode codepoint as UTF-8
static int write_utf8(char* out, int codepoint) {
    if (codepoint < 0x80) {
        out[0] = (char)codepoint;
        return 1;
    } else if (codepoint < 0x800) {
        out[0] = (char)(0xC0 | (codepoint >> 6));
        out[1] = (char)(0x80 | (codepoint & 0x3F));
        return 2;
    } else if (codepoint < 0x10000) {
        out[0] = (char)(0xE0 | (codepoint >> 12));
        out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[2] = (char)(0x80 | (codepoint & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | (codepoint >> 18));
        out[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[3] = (char)(0x80 | (codepoint & 0x3F));
        return 4;
    }
}

// Convert text to tiktoken format using OpenAI's bytes_to_unicode mapping
// Each byte in the input is mapped to a unicode codepoint, then encoded as UTF-8
// Returns a newly allocated string
char* text_to_tiktoken(const char* text) {
    if (!text) return NULL;

    init_tiktoken_encode_map();

    size_t len = strlen(text);
    // Each input byte could become up to 3 UTF-8 bytes (for codepoints 256-511)
    char* result = malloc(len * 3 + 1);
    if (!result) return NULL;

    int out_idx = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char byte = (unsigned char)text[i];
        int codepoint = tiktoken_byte_to_unicode[byte];
        out_idx += write_utf8(result + out_idx, codepoint);
    }
    result[out_idx] = '\0';
    return result;
}

// Convert text to SentencePiece format (space -> ▁)
// For SentencePiece, spaces are converted to ▁ (U+2581)
// Word-initial ▁ is added before the first non-space character if it's alphanumeric
// Returns a newly allocated string
char* text_to_sentencepiece(const char* text) {
    if (!text) return NULL;

    size_t len = strlen(text);
    // Each space becomes 3 bytes (▁), plus potential prefix
    char* result = malloc(len * 4 + 4);
    if (!result) return NULL;

    int out_idx = 0;
    int need_prefix = 1;  // Add ▁ before first alphanumeric char

    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ') {
            // Space -> ▁ (U+2581) = 0xE2 0x96 0x81
            result[out_idx++] = 0xE2;
            result[out_idx++] = 0x96;
            result[out_idx++] = 0x81;
            need_prefix = 0;  // ▁ already added for the space
        } else if (c == '\n' || c == '\t' || c == '\r') {
            // Control characters are kept as-is, no prefix needed after them
            result[out_idx++] = c;
            need_prefix = 1;  // Next word needs prefix
        } else {
            // Regular character - add prefix if this is start of a word
            if (need_prefix && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
                result[out_idx++] = 0xE2;  // U+2581 = ▁
                result[out_idx++] = 0x96;
                result[out_idx++] = 0x81;
            }
            result[out_idx++] = c;
            need_prefix = 0;
        }
    }
    result[out_idx] = '\0';
    return result;
}

// Build a sorted vocab index for faster longest-match lookup
typedef struct {
    int id;
    int len;
} VocabEntry;

static VocabEntry* sorted_vocab = NULL;
static int sorted_vocab_size = 0;

int vocab_entry_cmp(const void* a, const void* b) {
    // Sort by length descending (longest first)
    return ((VocabEntry*)b)->len - ((VocabEntry*)a)->len;
}

void build_sorted_vocab(char** vocab, int vocab_size) {
    if (sorted_vocab) return;  // Already built

    sorted_vocab = malloc(vocab_size * sizeof(VocabEntry));
    sorted_vocab_size = 0;

    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i] && strlen(vocab[i]) > 0) {
            sorted_vocab[sorted_vocab_size].id = i;
            sorted_vocab[sorted_vocab_size].len = strlen(vocab[i]);
            sorted_vocab_size++;
        }
    }

    qsort(sorted_vocab, sorted_vocab_size, sizeof(VocabEntry), vocab_entry_cmp);
}

// Greedy longest-match tokenizer
// This is simpler and works well with GGUF vocabularies that already contain merged tokens
// Global flag for tokenizer type (0 = tiktoken/BPE, 1 = SentencePiece)
int use_sentencepiece = 0;

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size,
                unsigned int max_token_length, int *tokens, int *n_tokens) {
    (void)vocab_scores;  // Not used in greedy approach
    (void)max_token_length;

    *n_tokens = 0;
    if (!text || !*text) return;

    // Convert text based on tokenizer type
    char* encoded_text;
    if (use_sentencepiece) {
        encoded_text = text_to_sentencepiece(text);
    } else {
        encoded_text = text_to_tiktoken(text);
    }
    if (!encoded_text) return;

    // Build sorted vocab for efficient lookup
    build_sorted_vocab(vocab, vocab_size);

    size_t text_len = strlen(encoded_text);
    size_t pos = 0;

    while (pos < text_len) {
        int best_id = -1;
        int best_len = 0;

        // Try to find longest matching token (vocab is sorted by length descending)
        for (int i = 0; i < sorted_vocab_size; i++) {
            int id = sorted_vocab[i].id;
            int len = sorted_vocab[i].len;

            // Skip if token is longer than remaining text
            if ((size_t)len > text_len - pos) continue;

            // Skip if shorter than best match we already found
            if (len <= best_len) break;  // Sorted by length, so no better match possible

            // Check if vocab entry matches
            if (strncmp(encoded_text + pos, vocab[id], len) == 0) {
                best_id = id;
                best_len = len;
                break;  // Found longest match
            }
        }

        if (best_id != -1) {
            tokens[(*n_tokens)++] = best_id;
            pos += best_len;
        } else {
            // No match found, skip this byte
            // This shouldn't happen if vocab contains all single bytes
            pos++;
        }
    }

    free(encoded_text);
}

// Initialize tokenizer from GGUF vocabulary
int init_tokenizer_from_gguf(GGUFFile* gguf, Config* p) {
    // Get special tokens from GGUF metadata
    int64_t bos = get_gguf_int(gguf, "tokenizer.ggml.bos_token_id", LLAMA3_BOS_TOKEN);
    int64_t eos = get_gguf_int(gguf, "tokenizer.ggml.eos_token_id", LLAMA3_EOS_TOKEN);
    tokenizer.bos_token = (int)bos;
    tokenizer.eos_token = (int)eos;
    tokenizer.max_token_length = 256;

    // Set special tokens
    tokenizer.start_header_token = LLAMA3_START_HEADER;
    tokenizer.end_header_token = LLAMA3_END_HEADER;
    tokenizer.eot_token = LLAMA3_EOT;

    // Use vocabulary from GGUF file
    if (gguf->vocab_tokens && gguf->vocab_size > 0) {
        if (debug_mode) {
            printf("Using vocabulary from GGUF file (%llu tokens)\n", (unsigned long long)gguf->vocab_size);
        }
        tokenizer.vocab_size = (int)gguf->vocab_size;
        tokenizer.vocab = (char**)malloc(tokenizer.vocab_size * sizeof(char*));
        tokenizer.vocab_scores = (float*)malloc(tokenizer.vocab_size * sizeof(float));

        if (!tokenizer.vocab || !tokenizer.vocab_scores) {
            fprintf(stderr, "Failed to allocate tokenizer\n");
            return 0;
        }

        // Copy vocabulary from GGUF
        for (int i = 0; i < tokenizer.vocab_size; i++) {
            if (gguf->vocab_tokens[i]) {
                tokenizer.vocab[i] = strdup(gguf->vocab_tokens[i]);
                // Track max token length
                size_t len = strlen(gguf->vocab_tokens[i]);
                if (len > tokenizer.max_token_length) {
                    tokenizer.max_token_length = (unsigned int)len;
                }
            } else {
                tokenizer.vocab[i] = NULL;
            }
            tokenizer.vocab_scores[i] = gguf->vocab_scores ? gguf->vocab_scores[i] : 0.0f;
        }

        // Update config vocab_size if GGUF has different size
        if (p->vocab_size != tokenizer.vocab_size) {
            printf("Note: Updating vocab_size from %d to %d based on GGUF\n", p->vocab_size, tokenizer.vocab_size);
            p->vocab_size = tokenizer.vocab_size;
        }

        return 1;
    }

    fprintf(stderr, "No vocabulary found in GGUF file\n");
    return 0;
}

void free_tokenizer(void) {
    if (tokenizer.vocab) {
        for (int i = 0; i < tokenizer.vocab_size; i++) {
            free(tokenizer.vocab[i]);
        }
        free(tokenizer.vocab);
    }
    free(tokenizer.vocab_scores);
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
#if defined _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (long)(counter.QuadPart * 1000 / freq.QuadPart);
#else
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
#endif
}

unsigned long long rng_seed;
unsigned int random_u32() {
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32() {
    return (random_u32() >> 8) / 16777216.0f;
}

int sample(float* probabilities, int n) {
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1;
}

int argmax(float* v, int n) {
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------
// Chat template encoding

// Gemma3 special token defaults
#define GEMMA3_BOS_TOKEN 2           // <bos>
#define GEMMA3_EOS_TOKEN 1           // <eos>
#define GEMMA3_START_TURN 106        // <start_of_turn>
#define GEMMA3_END_TURN 107          // <end_of_turn>

// Helper to find a special token in vocabulary by its string representation
int find_special_token(const char* token_str) {
    for (int i = 0; i < tokenizer.vocab_size; i++) {
        if (tokenizer.vocab[i] && strcmp(tokenizer.vocab[i], token_str) == 0) {
            return i;
        }
    }
    return -1;
}

// ----------------------------------------------------------------------------
// Chat history support

typedef struct {
    char* role;
    char* content;
} ChatMessage;

// Parse a JSON-escaped string value starting after the opening quote.
// Handles escape sequences: \", \\, \n, \t, \r, \/, \b, \f
// Returns pointer to character after the closing quote, or NULL on error.
static const char* parse_json_string(const char* p, char** out) {
    int cap = (int)strlen(p) + 1;
    char* buf = (char*)malloc(cap);
    int len = 0;

    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"':  buf[len++] = '"'; break;
                case '\\': buf[len++] = '\\'; break;
                case 'n':  buf[len++] = '\n'; break;
                case 't':  buf[len++] = '\t'; break;
                case 'r':  buf[len++] = '\r'; break;
                case '/':  buf[len++] = '/'; break;
                case 'b':  buf[len++] = '\b'; break;
                case 'f':  buf[len++] = '\f'; break;
                default:   buf[len++] = *p; break;
            }
        } else {
            buf[len++] = *p;
        }
        p++;
    }
    if (*p != '"') { free(buf); return NULL; }
    buf[len] = '\0';
    *out = buf;
    return p + 1;
}

// Parse a JSON chat history string into an array of ChatMessage.
// Expected format: [{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]
int parse_chat_history(const char* json, ChatMessage** messages, int* count) {
    *messages = NULL;
    *count = 0;

    int capacity = 16;
    ChatMessage* msgs = (ChatMessage*)malloc(capacity * sizeof(ChatMessage));
    int n = 0;

    const char* p = json;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '[') { free(msgs); return 0; }
    p++;

    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == ']') break;
        if (*p == ',') { p++; continue; }
        if (*p != '{') { free(msgs); return 0; }
        p++;

        char* role = NULL;
        char* content = NULL;

        for (;;) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (*p == '}') { p++; break; }
            if (*p == ',') { p++; continue; }
            if (*p != '"') { if (role) free(role); if (content) free(content); free(msgs); return 0; }
            p++;

            char* key = NULL;
            p = parse_json_string(p, &key);
            if (!p) { if (role) free(role); if (content) free(content); free(msgs); return 0; }

            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (*p != ':') { free(key); if (role) free(role); if (content) free(content); free(msgs); return 0; }
            p++;

            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (*p != '"') { free(key); if (role) free(role); if (content) free(content); free(msgs); return 0; }
            p++;

            char* value = NULL;
            p = parse_json_string(p, &value);
            if (!p) { free(key); if (role) free(role); if (content) free(content); free(msgs); return 0; }

            if (strcmp(key, "role") == 0) {
                if (role) free(role);
                role = value;
            } else if (strcmp(key, "content") == 0) {
                if (content) free(content);
                content = value;
            } else {
                free(value);
            }
            free(key);
        }

        if (role && content) {
            if (n >= capacity) {
                capacity *= 2;
                msgs = (ChatMessage*)realloc(msgs, capacity * sizeof(ChatMessage));
            }
            msgs[n].role = role;
            msgs[n].content = content;
            n++;
        } else {
            if (role) free(role);
            if (content) free(content);
        }
    }

    *messages = msgs;
    *count = n;
    return 1;
}

void free_chat_history(ChatMessage* messages, int count) {
    for (int i = 0; i < count; i++) {
        free(messages[i].role);
        free(messages[i].content);
    }
    free(messages);
}

// Encode chat history with Llama3 instruct format
void encode_llama3_chat_history(ChatMessage* messages, int msg_count, const char* sys_prompt, int* tokens, int* n_tokens) {
    *n_tokens = 0;
    int temp_tokens[8192];
    int temp_n = 0;

    int bos_token = find_special_token("<|begin_of_text|>");
    if (bos_token < 0) bos_token = LLAMA3_BOS_TOKEN;
    int start_header = find_special_token("<|start_header_id|>");
    if (start_header < 0) start_header = LLAMA3_START_HEADER;
    int end_header = find_special_token("<|end_header_id|>");
    if (end_header < 0) end_header = LLAMA3_END_HEADER;
    int eot_token = find_special_token("<|eot_id|>");
    if (eot_token < 0) eot_token = LLAMA3_EOT;

    tokens[(*n_tokens)++] = bos_token;

    if (sys_prompt && strlen(sys_prompt) > 0) {
        tokens[(*n_tokens)++] = start_header;
        bpe_encode("system", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
        tokens[(*n_tokens)++] = end_header;

        char* sys_text = malloc(strlen(sys_prompt) + 4);
        sprintf(sys_text, "\n\n%s", sys_prompt);
        bpe_encode(sys_text, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        free(sys_text);
        for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
        tokens[(*n_tokens)++] = eot_token;
    }

    for (int m = 0; m < msg_count; m++) {
        tokens[(*n_tokens)++] = start_header;
        bpe_encode(messages[m].role, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
        tokens[(*n_tokens)++] = end_header;

        char* msg_text = malloc(strlen(messages[m].content) + 4);
        sprintf(msg_text, "\n\n%s", messages[m].content);
        bpe_encode(msg_text, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        free(msg_text);
        for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
        tokens[(*n_tokens)++] = eot_token;
    }

    tokens[(*n_tokens)++] = start_header;
    bpe_encode("assistant", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
    tokens[(*n_tokens)++] = end_header;

    bpe_encode("\n\n", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
}

// Encode chat history with Gemma3 instruct format
void encode_gemma3_chat_history(ChatMessage* messages, int msg_count, const char* sys_prompt, int* tokens, int* n_tokens) {
    *n_tokens = 0;
    int temp_tokens[8192];
    int temp_n = 0;

    int bos_token = find_special_token("<bos>");
    if (bos_token < 0) bos_token = GEMMA3_BOS_TOKEN;
    int start_turn = find_special_token("<start_of_turn>");
    if (start_turn < 0) start_turn = GEMMA3_START_TURN;
    int end_turn = find_special_token("<end_of_turn>");
    if (end_turn < 0) end_turn = GEMMA3_END_TURN;

    tokens[(*n_tokens)++] = bos_token;

    int system_used = 0;
    for (int m = 0; m < msg_count; m++) {
        const char* role = messages[m].role;
        const char* content = messages[m].content;
        const char* gemma_role = (strcmp(role, "assistant") == 0) ? "model" : role;

        tokens[(*n_tokens)++] = start_turn;

        char* role_text;
        if (!system_used && strcmp(role, "user") == 0 && sys_prompt && strlen(sys_prompt) > 0) {
            role_text = malloc(strlen(gemma_role) + strlen(sys_prompt) + strlen(content) + 8);
            sprintf(role_text, "%s\n%s\n\n%s", gemma_role, sys_prompt, content);
            system_used = 1;
        } else {
            role_text = malloc(strlen(gemma_role) + strlen(content) + 4);
            sprintf(role_text, "%s\n%s", gemma_role, content);
        }

        bpe_encode(role_text, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        free(role_text);
        for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];

        tokens[(*n_tokens)++] = end_turn;

        bpe_encode("\n", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
    }

    tokens[(*n_tokens)++] = start_turn;
    bpe_encode("model\n", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) tokens[(*n_tokens)++] = temp_tokens[i];
}

// Encode a prompt with Gemma3 instruct format
// Format: <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
void encode_gemma3_chat(const char* prompt, const char* system_prompt, int* tokens, int* n_tokens) {
    *n_tokens = 0;
    int temp_tokens[8192];
    int temp_n = 0;

    // Look up special tokens in vocabulary (or use defaults)
    int bos_token = find_special_token("<bos>");
    if (bos_token < 0) bos_token = GEMMA3_BOS_TOKEN;

    int start_turn = find_special_token("<start_of_turn>");
    if (start_turn < 0) start_turn = GEMMA3_START_TURN;

    int end_turn = find_special_token("<end_of_turn>");
    if (end_turn < 0) end_turn = GEMMA3_END_TURN;

    // <bos>
    tokens[(*n_tokens)++] = bos_token;

    // Optional system prompt (Gemma puts it in first user turn or as separate turn)
    // Gemma3 prefers system instructions in the user message
    char* full_prompt;
    if (system_prompt && strlen(system_prompt) > 0) {
        // Prepend system prompt to user message
        full_prompt = malloc(strlen(system_prompt) + strlen(prompt) + 16);
        sprintf(full_prompt, "%s\n\n%s", system_prompt, prompt);
    } else {
        full_prompt = strdup(prompt);
    }

    // <start_of_turn>
    tokens[(*n_tokens)++] = start_turn;

    // "user\n" + prompt - encode via BPE
    char* user_text = malloc(strlen(full_prompt) + 8);
    sprintf(user_text, "user\n%s", full_prompt);
    bpe_encode(user_text, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    free(user_text);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }

    // <end_of_turn>
    tokens[(*n_tokens)++] = end_turn;

    // "\n" - encode via BPE
    bpe_encode("\n", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }

    // <start_of_turn>
    tokens[(*n_tokens)++] = start_turn;

    // "model\n" - encode via BPE
    bpe_encode("model\n", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }

    free(full_prompt);
}

// Encode a prompt with instruct format
// Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
void encode_llama3_chat(const char* prompt, const char* system_prompt, int* tokens, int* n_tokens) {
    *n_tokens = 0;
    int temp_tokens[8192];
    int temp_n = 0;

    // Look up special tokens in vocabulary (or use defaults)
    int bos_token = find_special_token("<|begin_of_text|>");
    if (bos_token < 0) bos_token = LLAMA3_BOS_TOKEN;

    int start_header = find_special_token("<|start_header_id|>");
    if (start_header < 0) start_header = LLAMA3_START_HEADER;

    int end_header = find_special_token("<|end_header_id|>");
    if (end_header < 0) end_header = LLAMA3_END_HEADER;

    int eot_token = find_special_token("<|eot_id|>");
    if (eot_token < 0) eot_token = LLAMA3_EOT;

    // <|begin_of_text|>
    tokens[(*n_tokens)++] = bos_token;

    // System prompt if provided
    if (system_prompt && strlen(system_prompt) > 0) {
        // <|start_header_id|>
        tokens[(*n_tokens)++] = start_header;

        // "system" - encode via BPE
        bpe_encode("system", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        for (int i = 0; i < temp_n; i++) {
            tokens[(*n_tokens)++] = temp_tokens[i];
        }

        // <|end_header_id|>
        tokens[(*n_tokens)++] = end_header;

        // "\n\n" + system_prompt (encode via BPE)
        char* sys_text = malloc(strlen(system_prompt) + 4);
        sprintf(sys_text, "\n\n%s", system_prompt);
        bpe_encode(sys_text, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                   tokenizer.max_token_length, temp_tokens, &temp_n);
        free(sys_text);
        for (int i = 0; i < temp_n; i++) {
            tokens[(*n_tokens)++] = temp_tokens[i];
        }

        // <|eot_id|>
        tokens[(*n_tokens)++] = eot_token;
    }

    // <|start_header_id|>
    tokens[(*n_tokens)++] = start_header;

    // "user" - encode via BPE
    bpe_encode("user", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }

    // <|end_header_id|>
    tokens[(*n_tokens)++] = end_header;

    // "\n\n" + prompt (encode via BPE)
    char* user_text = malloc(strlen(prompt) + 4);
    sprintf(user_text, "\n\n%s", prompt);
    bpe_encode(user_text, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    free(user_text);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }

    // <|eot_id|>
    tokens[(*n_tokens)++] = eot_token;

    // <|start_header_id|>
    tokens[(*n_tokens)++] = start_header;

    // "assistant" - encode via BPE
    bpe_encode("assistant", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }

    // <|end_header_id|>
    tokens[(*n_tokens)++] = end_header;

    // "\n\n" (encode via BPE)
    bpe_encode("\n\n", tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
               tokenizer.max_token_length, temp_tokens, &temp_n);
    for (int i = 0; i < temp_n; i++) {
        tokens[(*n_tokens)++] = temp_tokens[i];
    }
}

// ----------------------------------------------------------------------------
// Global state

float temperature = 0.9f;
float top_p = 0.9f;
int top_k = 40;
int max_tokens = 256;
int context_size = 0;  // 0 means use model's default
char *system_prompt = "You are a helpful assistant.";
Config config;
TransformerWeights weights;
RunState state;
int *prompt_tokens = NULL;
int num_prompt_tokens = 0;
int next;
int token;
int pos = 0;
GGUFFile* gguf_file = NULL;
int use_chat_template = 1;  // Enable chat template by default

// Track recently generated tokens for repetition penalty
static int recent_tokens[64];
static int recent_count = 0;
static float repetition_penalty = 1.0f;  // Disabled: Penalty factor for repeated tokens

// Buffer newlines - only output if followed by non-end token
static int pending_newline = 0;

// Buffer for output text to detect repetition loops
static char output_buffer[16384];
static int output_buffer_len = 0;
static int repetition_detected = 0;

void generate_token(void) {
    // Skip logit computation during prompt prefill (major speedup)
    int need_logits = (pos >= num_prompt_tokens - 1) ? 1 : 0;
    transformer(token, pos, &config, &state, &weights, need_logits);

    // Debug: print first few logits to see if they make sense
    if (debug_mode && pos >= num_prompt_tokens - 1 && pos < num_prompt_tokens + 3) {
        printf("\n[DEBUG pos=%d] Top 5 logits: ", pos);
        // Find top 5 logits
        int top_ids[5] = {0, 1, 2, 3, 4};
        float top_vals[5];
        for (int i = 0; i < 5; i++) top_vals[i] = state.logits[i];
        for (int i = 5; i < config.vocab_size; i++) {
            for (int j = 0; j < 5; j++) {
                if (state.logits[i] > top_vals[j]) {
                    for (int k = 4; k > j; k--) {
                        top_ids[k] = top_ids[k-1];
                        top_vals[k] = top_vals[k-1];
                    }
                    top_ids[j] = i;
                    top_vals[j] = state.logits[i];
                    break;
                }
            }
        }
        for (int i = 0; i < 5; i++) {
            printf("%d(%.2f) ", top_ids[i], top_vals[i]);
        }
        printf("\n");
    }

    // During prompt phase: use the next prompt token
    // During generation phase: sample from logits
    // Note: pos is 0-indexed, and prompt_tokens[0] was used as initial token
    if (pos < num_prompt_tokens - 1) {
        next = prompt_tokens[pos + 1];
    } else {
        // Apply repetition penalty to recently generated tokens
        for (int i = 0; i < recent_count; i++) {
            int tok = recent_tokens[i];
            if (tok >= 0 && tok < config.vocab_size) {
                if (state.logits[tok] > 0) {
                    state.logits[tok] /= repetition_penalty;
                } else {
                    state.logits[tok] *= repetition_penalty;
                }
            }
        }

        if (temperature == 0.0f) {
            next = argmax(state.logits, config.vocab_size);
        } else {
            int k = top_k;
            if (k > config.vocab_size) k = config.vocab_size;
#if defined _WIN32
            int* top_k_idx = (int*)_alloca(k * sizeof(int));
            float* top_k_val = (float*)_alloca(k * sizeof(float));
#else
            int top_k_idx[k];
            float top_k_val[k];
#endif

            // Initialize with first k logits
            for (int i = 0; i < k; i++) {
                top_k_idx[i] = i;
                top_k_val[i] = state.logits[i];
            }

            // Find current min in top-k
            int min_pos = 0;
            float min_val = top_k_val[0];
            for (int i = 1; i < k; i++) {
                if (top_k_val[i] < min_val) {
                    min_val = top_k_val[i];
                    min_pos = i;
                }
            }

            // Scan rest of vocab, replacing min when larger found
            for (int i = k; i < config.vocab_size; i++) {
                if (state.logits[i] > min_val) {
                    top_k_val[min_pos] = state.logits[i];
                    top_k_idx[min_pos] = i;
                    // Re-find min
                    min_val = top_k_val[0];
                    min_pos = 0;
                    for (int j = 1; j < k; j++) {
                        if (top_k_val[j] < min_val) {
                            min_val = top_k_val[j];
                            min_pos = j;
                        }
                    }
                }
            }

            // Apply temperature and fused max+softmax over just k values
            float inv_temp = 1.0f / temperature;
            float max_v = top_k_val[0];
            for (int i = 1; i < k; i++) {
                if (top_k_val[i] > max_v) max_v = top_k_val[i];
            }
            float max_vt = max_v * inv_temp;
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                float e = expf(top_k_val[i] * inv_temp - max_vt);
                top_k_val[i] = e;
                sum += e;
            }

            // Apply top-P (nucleus) filtering: sort by probability descending,
            // then keep only tokens whose cumulative probability reaches top_p
            int n = k;
            if (top_p < 1.0f) {
                // Insertion sort by probability descending (k is small, ~40)
                for (int i = 1; i < k; i++) {
                    float key_val = top_k_val[i];
                    int key_idx = top_k_idx[i];
                    int j = i - 1;
                    while (j >= 0 && top_k_val[j] < key_val) {
                        top_k_val[j + 1] = top_k_val[j];
                        top_k_idx[j + 1] = top_k_idx[j];
                        j--;
                    }
                    top_k_val[j + 1] = key_val;
                    top_k_idx[j + 1] = key_idx;
                }

                // Accumulate probabilities until we reach the top_p threshold
                float cum_sum = 0.0f;
                float threshold = top_p * sum;
                n = k;
                for (int i = 0; i < k; i++) {
                    cum_sum += top_k_val[i];
                    if (cum_sum >= threshold) {
                        n = i + 1;
                        break;
                    }
                }

                // Recompute sum over the kept tokens
                sum = 0.0f;
                for (int i = 0; i < n; i++) {
                    sum += top_k_val[i];
                }
            }

            // Normalize and sample from the filtered distribution
            float inv_sum = 1.0f / sum;
            for (int i = 0; i < n; i++) {
                top_k_val[i] *= inv_sum;
            }
            int s = sample(top_k_val, n);
            next = top_k_idx[s];
        }

        // Update recent tokens buffer (circular)
        if (recent_count < 64) {
            recent_tokens[recent_count++] = next;
        } else {
            // Shift left and add new token
            for (int i = 0; i < 63; i++) {
                recent_tokens[i] = recent_tokens[i + 1];
            }
            recent_tokens[63] = next;
        }
    }

    // Only print generated tokens (not prompt tokens) during generation
    // Skip special tokens like <|eot_id|>
    if (pos >= num_prompt_tokens - 1 && tokenizer.vocab[next] && next != tokenizer.eot_token && next != tokenizer.eos_token) {
        // Decode tiktoken representation to normal text
        char* decoded = decode_tiktoken(tokenizer.vocab[next]);
        if (decoded) {
            // Buffer newlines - only output if followed by non-end token
            if (strcmp(decoded, "\n") == 0) {
                pending_newline = 1;
            } else {
                if (pending_newline) {
                    printf("\n");
                    pending_newline = 0;
                }
                printf("%s", decoded);
            }
            fflush(stdout);

            // Accumulate text in output buffer for repetition detection
            int decoded_len = strlen(decoded);
            if (output_buffer_len + decoded_len >= (int)sizeof(output_buffer) - 1) {
                // Buffer full: shift the second half to the front to keep detection working
                int half = output_buffer_len / 2;
                memmove(output_buffer, output_buffer + half, output_buffer_len - half);
                output_buffer_len -= half;
                output_buffer[output_buffer_len] = '\0';
            }
            memcpy(output_buffer + output_buffer_len, decoded, decoded_len);
            output_buffer_len += decoded_len;
            output_buffer[output_buffer_len] = '\0';

            // Stop if the model is stuck repeating text
            if (output_buffer_len > 100) {
                char* to_find = output_buffer + output_buffer_len - 30;
                int count = 0;
                char* search = output_buffer;
                while ((search = strstr(search, to_find)) != NULL) {
                    count++;
                    search += 30;
                }
                if (count > 10) {
                    repetition_detected = 1;
                }
            }

            free(decoded);
        }
    }

    token = next;
    pos++;
}


int main(int argc, char *argv[]) {
    int exit_code = 0;
    char *checkpoint = NULL;
    char *prompt = NULL;
    char *chat_history_file = NULL;

    // Parse named arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-model") == 0 && i + 1 < argc) {
            checkpoint = argv[++i];
        } else if (strcmp(argv[i], "-temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-max_tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-context_size") == 0 && i + 1 < argc) {
            context_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-chathistory") == 0 && i + 1 < argc) {
            chat_history_file = argv[++i];
        } else if (strcmp(argv[i], "-system_prompt") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "-top_p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "-top_k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-debug") == 0) {
            debug_mode = 1;
        }
    }

    // Check required arguments
    if (checkpoint == NULL || (prompt == NULL && chat_history_file == NULL)) {
        printf("Usage: %s -model <model.gguf> -prompt <text> [options]\n", argv[0]);
        printf("       %s -model <model.gguf> -chathistory <file.txt> [options]\n", argv[0]);
        printf("\nRequired arguments:\n");
        printf("  -model         path to GGUF model file\n");
        printf("  -prompt        input prompt text\n");
        printf("  -chathistory   path to a .txt file with JSON chat history (alternative to -prompt)\n");
        printf("\nOptional arguments:\n");
        printf("  -system_prompt system prompt (default: \"You are a helpful assistant.\")\n");
        printf("  -temperature   sampling temperature (default: 0.9, use 0.0 for greedy)\n");
        printf("  -top_p         top-P nucleus sampling threshold (default: 0.9)\n");
        printf("  -top_k         top-K sampling count (default: 40)\n");
        printf("  -max_tokens    number of tokens to generate (default: 256)\n");
        printf("  -context_size  context size for the AI model (default: model's max)\n");
        printf("  -debug         show detailed model loading and performance logs\n");
        printf("\nExamples:\n");
        printf("  %s -model Llama3.gguf -prompt \"tell me what is microsoft\"\n", argv[0]);
        printf("  %s -model Llama3.gguf -chathistory history.txt\n", argv[0]);
        return 1;
    }

    rng_seed = (unsigned int)time(NULL);

    // Initialize FP16 lookup table for fast conversion
    init_fp16_table();

    // Load GGUF model
    if (debug_mode) {
        printf("Loading GGUF model: %s\n", checkpoint);
    }

    gguf_file = parse_gguf_file(checkpoint);
    if (!gguf_file) {
        fprintf(stderr, "Failed to parse GGUF file\n");
        exit_code = 1; goto cleanup;
    }

    // Load tokenizer vocabulary from GGUF (contains tokens and scores arrays)
    if (!load_gguf_tokenizer(gguf_file, checkpoint)) {
        fprintf(stderr, "Failed to load vocabulary from GGUF\n");
        exit_code = 1; goto cleanup;
    }

    // Extract config from GGUF metadata
    const char* arch = get_gguf_string(gguf_file, "general.architecture");
    if (debug_mode && arch) {
        printf("Model architecture: %s\n", arch);
    }

    // Detect architecture type and set appropriate GGUF key prefix
    config.is_gemma3 = 0;
    const char* key_prefix = "llama";  // default to llama keys

    if (arch && (strcmp(arch, "gemma3") == 0 || strcmp(arch, "gemma2") == 0 || strcmp(arch, "gemma") == 0)) {
        config.is_gemma3 = 1;
        use_sentencepiece = 1;  // Gemma uses SentencePiece tokenizer
        key_prefix = "gemma3";
        // Try gemma3 keys first, fall back to gemma2, then gemma
        if (get_gguf_int(gguf_file, "gemma3.embedding_length", 0) == 0) {
            if (get_gguf_int(gguf_file, "gemma2.embedding_length", 0) != 0) {
                key_prefix = "gemma2";
            } else if (get_gguf_int(gguf_file, "gemma.embedding_length", 0) != 0) {
                key_prefix = "gemma";
            }
        }
        if (debug_mode) {
            printf("Detected Gemma architecture, using %s.* keys\n", key_prefix);
        }
    }

    // Build config key names dynamically based on architecture
    char key_dim[64], key_hidden[64], key_layers[64], key_heads[64];
    char key_kv_heads[64], key_vocab[64], key_ctx[64], key_rope[64];
    char key_head_dim[64], key_rms_eps[64], key_softcap[64];

    snprintf(key_dim, sizeof(key_dim), "%s.embedding_length", key_prefix);
    snprintf(key_hidden, sizeof(key_hidden), "%s.feed_forward_length", key_prefix);
    snprintf(key_layers, sizeof(key_layers), "%s.block_count", key_prefix);
    snprintf(key_heads, sizeof(key_heads), "%s.attention.head_count", key_prefix);
    snprintf(key_kv_heads, sizeof(key_kv_heads), "%s.attention.head_count_kv", key_prefix);
    snprintf(key_vocab, sizeof(key_vocab), "%s.vocab_size", key_prefix);
    snprintf(key_ctx, sizeof(key_ctx), "%s.context_length", key_prefix);
    snprintf(key_rope, sizeof(key_rope), "%s.rope.freq_base", key_prefix);
    snprintf(key_head_dim, sizeof(key_head_dim), "%s.attention.key_length", key_prefix);
    snprintf(key_rms_eps, sizeof(key_rms_eps), "%s.attention.layer_norm_rms_epsilon", key_prefix);
    snprintf(key_softcap, sizeof(key_softcap), "%s.final_logit_softcapping", key_prefix);

    // Read sliding window attention parameters
    char key_swa[64], key_rope_swa[64];
    snprintf(key_swa, sizeof(key_swa), "%s.attention.sliding_window", key_prefix);
    snprintf(key_rope_swa, sizeof(key_rope_swa), "%s.rope.freq_base_swa", key_prefix);

    config.swa_window = get_gguf_int(gguf_file, key_swa, 0);
    config.rope_theta_swa = get_gguf_float(gguf_file, key_rope_swa, 10000.0f);  // default 10000 for SWA
    config.swa_pattern = 6;  // Gemma3 uses pattern 6: layers 0-4 are SWA, layer 5 is dense, etc.

    if (debug_mode) {
        if (config.swa_window > 0) {
            printf("[DEBUG] Sliding window attention: %d\n", config.swa_window);
        }
        if (config.rope_theta_swa > 0) {
            printf("[DEBUG] SWA RoPE freq base: %.0f\n", config.rope_theta_swa);
        }
        // Check for rope dimension count
        char key_rope_dim[64];
        snprintf(key_rope_dim, sizeof(key_rope_dim), "%s.rope.dimension_count", key_prefix);
        int rope_dim = get_gguf_int(gguf_file, key_rope_dim, 0);
        if (rope_dim > 0) {
            printf("[DEBUG] RoPE dimension count: %d\n", rope_dim);
        }
    }

    config.dim = get_gguf_int(gguf_file, key_dim, 4096);
    config.hidden_dim = get_gguf_int(gguf_file, key_hidden, 11008);
    config.n_layers = get_gguf_int(gguf_file, key_layers, 32);
    config.n_heads = get_gguf_int(gguf_file, key_heads, 32);
    config.n_kv_heads = get_gguf_int(gguf_file, key_kv_heads, config.n_heads);
    config.vocab_size = get_gguf_int(gguf_file, key_vocab, 32000);
    config.seq_len = get_gguf_int(gguf_file, key_ctx, 2048);

    // For Gemma3, RoPE theta defaults to 1M
    float default_rope_theta = config.is_gemma3 ? 1000000.0f : 500000.0f;
    config.rope_theta = get_gguf_float(gguf_file, key_rope, default_rope_theta);

    // Get head dimension (explicit for Gemma3, computed for others)
    config.head_dim = get_gguf_int(gguf_file, key_head_dim, config.dim / config.n_heads);

    // Gemma3-specific parameters
    config.rms_norm_eps = get_gguf_float(gguf_file, key_rms_eps, 1e-6f);
    config.final_logit_softcapping = get_gguf_float(gguf_file, key_softcap, 0.0f);  // 0 means disabled

    // Update vocab_size from GGUF vocabulary if available (before loading weights)
    // This is important because Gemma models may have larger vocab than metadata suggests
    if (gguf_file->vocab_size > 0 && (int64_t)config.vocab_size != gguf_file->vocab_size) {
        if (debug_mode) {
            printf("Note: Updating vocab_size from %d to %llu based on GGUF vocabulary\n",
                   config.vocab_size, (unsigned long long)gguf_file->vocab_size);
        }
        config.vocab_size = (int)gguf_file->vocab_size;
    }

    if (debug_mode) {
        printf("Config: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
               config.dim, config.hidden_dim, config.n_layers, config.n_heads,
               config.n_kv_heads, config.vocab_size, config.seq_len);
        printf("RoPE theta: %.1f, head_dim: %d\n", config.rope_theta, config.head_dim);
        if (config.is_gemma3) {
            printf("Gemma3: rms_norm_eps=%.2e, final_logit_softcapping=%.1f\n",
                   config.rms_norm_eps, config.final_logit_softcapping);
        }
    }

    // Load weights (using updated vocab_size)
    if (!init_weights_from_gguf(gguf_file, &config, &weights)) {
        fprintf(stderr, "Failed to load weights from GGUF\n");
        exit_code = 1; goto cleanup;
    }

    // Initialize tokenizer
    if (!init_tokenizer_from_gguf(gguf_file, &config)) {
        fprintf(stderr, "Failed to initialize tokenizer\n");
        exit_code = 1; goto cleanup;
    }

    // Initial token will be set from prompt_tokens[0] if prompt provided
    token = tokenizer.bos_token;

    // Apply context_size override if specified
    if (context_size > 0) {
        config.seq_len = context_size;
    }

    if (max_tokens <= 0 || max_tokens > config.seq_len) {
        max_tokens = config.seq_len;
    }

    malloc_run_state(&state, &config);

    // Allocate Q8_0 buffer for quantized input matmul optimization
    // Size: max input dimension (max of dim, q_dim, hidden_dim) in Q8_0 blocks
    {
        int head_size = config.head_dim > 0 ? config.head_dim : config.dim / config.n_heads;
        int q_dim = config.n_heads * head_size;
        int max_input = config.dim;
        if (q_dim > max_input) max_input = q_dim;
        if (config.hidden_dim > max_input) max_input = config.hidden_dim;
        int q8_buf_size = (max_input + QK8_0 - 1) / QK8_0;
        q8_buf = (block_q8_0*)calloc(q8_buf_size, sizeof(block_q8_0));
    }

    // Determine Gemma3-specific stop tokens
    int gemma3_end_turn = -1;
    if (config.is_gemma3) {
        gemma3_end_turn = find_special_token("<end_of_turn>");
        if (gemma3_end_turn < 0) gemma3_end_turn = GEMMA3_END_TURN;
    }

    prompt_tokens = (int*)malloc(config.seq_len * sizeof(int));

    if (chat_history_file != NULL) {
        // Read chat history from file
        FILE* f = fopen(chat_history_file, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open chat history file: %s\n", chat_history_file);
            exit_code = 1; goto cleanup;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* json = (char*)malloc(fsize + 1);
        fread(json, 1, fsize, f);
        fclose(f);
        json[fsize] = '\0';

        ChatMessage* messages = NULL;
        int msg_count = 0;
        if (!parse_chat_history(json, &messages, &msg_count) || msg_count == 0) {
            fprintf(stderr, "Failed to parse chat history from file: %s\n", chat_history_file);
            free(json);
            exit_code = 1; goto cleanup;
        }
        free(json);

        if (config.is_gemma3) {
            encode_gemma3_chat_history(messages, msg_count, system_prompt, prompt_tokens, &num_prompt_tokens);
        } else {
            encode_llama3_chat_history(messages, msg_count, system_prompt, prompt_tokens, &num_prompt_tokens);
        }
        free_chat_history(messages, msg_count);

        token = prompt_tokens[0];
    } else if (prompt != NULL) {
        // Use chat template for instruct models
        if (use_chat_template) {
            if (config.is_gemma3) {
                // Use Gemma3 chat template
                encode_gemma3_chat(prompt, system_prompt, prompt_tokens, &num_prompt_tokens);
            } else {
                // Use Llama3 chat template
                encode_llama3_chat(prompt, system_prompt, prompt_tokens, &num_prompt_tokens);
            }
        } else {
            bpe_encode(prompt, tokenizer.vocab, tokenizer.vocab_scores, tokenizer.vocab_size,
                       tokenizer.max_token_length, prompt_tokens, &num_prompt_tokens);
        }

        // Use the first prompt token as the initial token
        token = prompt_tokens[0];
    }

    long start = 0;

    while (pos < max_tokens) {
        generate_token();
        if (start == 0) {
            start = time_in_ms();
        }

        // Only stop on EOS/EOT during generation (after prompt is consumed)
        if (pos >= num_prompt_tokens - 1) {
            // Stop on EOS token
            if (token == tokenizer.eos_token) {
                break;
            }
            // Stop on EOT token (Llama3)
            if (token == tokenizer.eot_token) {
                break;
            }
            // Stop on end_of_turn token (Gemma3)
            if (config.is_gemma3 && token == gemma3_end_turn) {
                break;
            }
            // Stop if the model is stuck repeating text
            if (repetition_detected) {
                break;
            }
        }
    }

    long end = time_in_ms();
    if (debug_mode && pos > 1) {
        printf("\nachieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    } else {
        printf("\n");  // Ensure newline after response
    }

cleanup:
    // Cleanup (safe to call on NULL - all globals are zero-initialized)
    free_run_state(&state);
    free(q8_buf);
    free_tokenizer();
    free(sorted_vocab);

    free(prompt_tokens);

    // Free float weight allocations (embeddings, norms, etc.)
    for (int i = 0; i < num_weight_allocations; i++) {
        free(weight_allocations[i]);
    }
    free(weight_allocations);

    // Free QuantizedTensor arrays (the arrays themselves, not the data - data is mmap'd)
    free(weights.wq);
    free(weights.wk);
    free(weights.wv);
    free(weights.wo);
    free(weights.w1);
    free(weights.w2);
    free(weights.w3);

    // Free Gemma3-specific arrays if allocated
    free(weights.attn_q_norm);
    free(weights.attn_k_norm);
    free(weights.attn_post_norm);
    free(weights.ffn_post_norm);

    // Free norm weight arrays
    free(weights.rms_att_weight);
    free(weights.rms_ffn_weight);

    if (mapped_data && mapped_data != MAP_FAILED) {
        munmap(mapped_data, mapped_size);
    }

    if (gguf_file) {
        free_gguf_file(gguf_file);
    }

    return exit_code;
}
