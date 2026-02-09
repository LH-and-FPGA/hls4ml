#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_layernorm.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/s3.h"
#include "weights/b3.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/s6.h"
#include "weights/b6.h"
#include "weights/w8.h"
#include "weights/b8.h"


// hls-fpga-machine-learning insert layer-config
// fc1
struct config2 : nnet::dense_config {
    static const unsigned n_in = 1536;
    static const unsigned n_out = 64;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 98304;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef fc1_bias_t bias_t;
    typedef fc1_weight_t weight_t;
    typedef layer2_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// ln1
struct config3 : nnet::layernorm_config {
    static const unsigned n_in = 64;
    static const unsigned seq_len = 1;
    static const unsigned axis = 1;
    static const unsigned epsilon_power_of_10 = 5;
    static const unsigned table_range_power2 = 0;
    static const unsigned table_size = 8192;
    typedef ln1_accum_t accum_t;
    typedef ln1_bias_t bias_t;
    typedef ln1_scale_t scale_t;
    typedef ln1_table_t table_t;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// relu1
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    typedef relu1_table_t table_t;
};

// fc2
struct config5 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 64;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 4096;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef fc2_bias_t bias_t;
    typedef fc2_weight_t weight_t;
    typedef layer5_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// ln2
struct config6 : nnet::layernorm_config {
    static const unsigned n_in = 64;
    static const unsigned seq_len = 1;
    static const unsigned axis = 1;
    static const unsigned epsilon_power_of_10 = 5;
    static const unsigned table_range_power2 = 0;
    static const unsigned table_size = 8192;
    typedef ln2_accum_t accum_t;
    typedef ln2_bias_t bias_t;
    typedef ln2_scale_t scale_t;
    typedef ln2_table_t table_t;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// relu2
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    typedef relu2_table_t table_t;
};

// fc3
struct config8 : nnet::dense_config {
    static const unsigned n_in = 64;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 64;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 64;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef fc3_bias_t bias_t;
    typedef fc3_weight_t weight_t;
    typedef layer8_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseResource_rf_leq_nin<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// sigmoid
struct sigmoid_config9 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 64;
    typedef sigmoid_table_t table_t;
};



#endif
