
#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <cmath>

namespace nnet {

struct layernorm_config {
    typedef float bias_t;
    typedef float scale_t;
    typedef float accum_t;
    typedef float table_t;
    static const unsigned n_in = 20;
    static const unsigned seq_len = 4;
    static const unsigned axis = 2;
    static const unsigned epsilon_power_of_10 = 3;
    static const unsigned table_range_power2 = 0;
    static const unsigned table_size = 1024;
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// ── Inverse-sqrt lookup table (synthesis) ──────────────────────────
// Covers variance range [eps, INVSQRT_MAX_VAR] with CONFIG_T::table_size entries.
// INVSQRT_MAX_VAR = 64.0 is sufficient for Dense outputs of typical wake-word models.
static const float INVSQRT_MAX_VAR = 64.0f;

template <typename CONFIG_T>
void init_invsqrt_table(typename CONFIG_T::table_t table_out[CONFIG_T::table_size]) {
    float eps = 1.0f;
    for (unsigned p = 0; p < CONFIG_T::epsilon_power_of_10; ++p) eps *= 0.1f;
    for (unsigned i = 0; i < CONFIG_T::table_size; ++i) {
        float x = eps + (INVSQRT_MAX_VAR - eps) * ((float)i / (float)(CONFIG_T::table_size - 1));
        float val = 1.0f / sqrtf(x);
        table_out[i] = (typename CONFIG_T::table_t)val;
    }
}

template <typename CONFIG_T>
typename CONFIG_T::table_t lookup_invsqrt(typename CONFIG_T::accum_t var) {
    // Compute epsilon at compile time
    float eps = 1.0f;
    for (unsigned p = 0; p < CONFIG_T::epsilon_power_of_10; ++p) {
        #pragma HLS UNROLL
        eps *= 0.1f;
    }
    // Initialize table (only once, stored in BRAM)
    static bool initialized = false;
    static typename CONFIG_T::table_t invsqrt_table[CONFIG_T::table_size];
    if (!initialized) {
        init_invsqrt_table<CONFIG_T>(invsqrt_table);
        initialized = true;
    }
    #pragma HLS RESOURCE variable=invsqrt_table core=ROM_nP_LUTRAM
    // Clamp and compute index
    float var_f = (float)var;
    if (var_f < eps) var_f = eps;
    if (var_f > INVSQRT_MAX_VAR) var_f = INVSQRT_MAX_VAR;
    unsigned index = (unsigned)(((var_f - eps) / (INVSQRT_MAX_VAR - eps)) * (CONFIG_T::table_size - 1));
    if (index >= CONFIG_T::table_size) index = CONFIG_T::table_size - 1;
    return invsqrt_table[index];
}

// ── LayerNorm core ─────────────────────────────────────────────────
template <class data_T, class res_T, typename CONFIG_T>
void layernorm_1d(data_T data[CONFIG_T::n_in / CONFIG_T::seq_len],
                  res_T res[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    typename CONFIG_T::accum_t sum_cache = 0;
    typename CONFIG_T::accum_t sum_cache2 = 0;
    typename CONFIG_T::accum_t var, mean, diff;
    typename CONFIG_T::accum_t data_diff[dim];
    typename CONFIG_T::accum_t deno_inver;

    #pragma HLS ARRAY_PARTITION variable=data_diff complete

    const typename CONFIG_T::accum_t k_inv = 1.0 / dim;

LAYERNORM_1D_SUM:
    for (int i = 0; i < dim; ++i) {
        sum_cache += static_cast<typename CONFIG_T::accum_t>(data[i]);
    }
    mean = CONFIG_T::template product<typename CONFIG_T::accum_t,
           typename CONFIG_T::accum_t>::product(sum_cache, k_inv);

LAYERNORM_1D_VAR:
    for (int i = 0; i < dim; ++i) {
        data_diff[i] = static_cast<typename CONFIG_T::accum_t>(data[i]) - mean;
        diff = data_diff[i] * data_diff[i];
        sum_cache2 += diff;
    }
    var = CONFIG_T::template product<typename CONFIG_T::accum_t,
          typename CONFIG_T::accum_t>::product(sum_cache2, k_inv);

    // Inverse sqrt of variance
#ifndef __SYNTHESIS__
    // C simulation: direct computation for accuracy
    {
        float eps_f = 1.0f;
        for (unsigned p = 0; p < CONFIG_T::epsilon_power_of_10; ++p) eps_f *= 0.1f;
        float var_f = (float)var;
        if (var_f < eps_f) var_f = eps_f;
        deno_inver = (typename CONFIG_T::accum_t)(1.0f / sqrtf(var_f));
    }
#else
    // HLS synthesis: lookup table (no floating-point sqrt hardware needed)
    deno_inver = (typename CONFIG_T::accum_t)lookup_invsqrt<CONFIG_T>(var);
#endif

LAYERNORM_1D_RESULT:
    for (int i = 0; i < dim; ++i) {
        res[i] = data_diff[i] * deno_inver * scale[i] + bias[i];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    data_T in_val[dim];
    res_T outval[dim];

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete
    #pragma HLS ARRAY_PARTITION variable=in_val complete
    #pragma HLS ARRAY_PARTITION variable=outval complete

LAYERNORM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE
    LAYERNORM_LOAD:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_val[i] = data[j * dim + i];
        }
        layernorm_1d<data_T, res_T, CONFIG_T>(in_val, outval, scale, bias);
    LAYERNORM_STORE:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            res[j * dim + i] = outval[i];
        }
    }
}

} // namespace nnet
#endif
