#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,10> input_t;
typedef ap_fixed<16,10> model_default_t;
typedef ap_fixed<44,32> fc1_result_t;
typedef ap_fixed<16,10> fc1_weight_t;
typedef ap_fixed<16,10> fc1_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<48,24> ln1_accum_t;
typedef ap_fixed<48,24> ln1_table_t;
typedef ap_fixed<16,10> layer3_t;
typedef ap_fixed<16,10> ln1_scale_t;
typedef ap_fixed<16,10> ln1_bias_t;
typedef ap_fixed<16,10> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<40,28> fc2_result_t;
typedef ap_fixed<16,10> fc2_weight_t;
typedef ap_fixed<16,10> fc2_bias_t;
typedef ap_uint<1> layer5_index;
typedef ap_fixed<48,24> ln2_accum_t;
typedef ap_fixed<48,24> ln2_table_t;
typedef ap_fixed<16,10> layer6_t;
typedef ap_fixed<16,10> ln2_scale_t;
typedef ap_fixed<16,10> ln2_bias_t;
typedef ap_fixed<16,10> layer7_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<40,28> fc3_result_t;
typedef ap_fixed<16,10> fc3_weight_t;
typedef ap_fixed<16,10> fc3_bias_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<16,10> result_t;
typedef ap_fixed<18,8> sigmoid_table_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
