
#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

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

// Patched layernorm_1d: uses direct 1/sqrtf(var) instead of lookup table.
// The original table only covers variance in [eps, 1.0] which is too narrow
// for models with large feature dimensions. Direct sqrtf is accurate for C
// simulation. For HLS synthesis, replace with hls::rsqrt() or a wider-range table.
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

    // Direct 1/sqrt(var) computation
    {
        float eps_f = powf(10.0f, -(int)CONFIG_T::epsilon_power_of_10);
        float var_f = (float)var;
        if (var_f < eps_f) var_f = eps_f;
        deno_inver = (typename CONFIG_T::accum_t)(1.0f / sqrtf(var_f));
    }

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
