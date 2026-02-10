#include <iostream>

#include "hey_jarvis_verifier.h"
#include "parameters.h"


void hey_jarvis_verifier(
    input_t x[1536],
    result_t layer9_out[1]
) {

    // hls-fpga-machine-learning insert IO
        #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    #pragma HLS INTERFACE ap_none port=x
    #pragma HLS INTERFACE ap_vld port=layer9_out
    #pragma HLS INTERFACE ap_ctrl_hs port=return 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<fc1_weight_t, 98304>(w2, "w2.txt");
        nnet::load_weights_from_txt<fc1_bias_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<ln1_scale_t, 64>(s3, "s3.txt");
        nnet::load_weights_from_txt<ln1_bias_t, 64>(b3, "b3.txt");
        nnet::load_weights_from_txt<fc2_weight_t, 4096>(w5, "w5.txt");
        nnet::load_weights_from_txt<fc2_bias_t, 64>(b5, "b5.txt");
        nnet::load_weights_from_txt<ln2_scale_t, 64>(s6, "s6.txt");
        nnet::load_weights_from_txt<ln2_bias_t, 64>(b6, "b6.txt");
        nnet::load_weights_from_txt<fc3_weight_t, 64>(w8, "w8.txt");
        nnet::load_weights_from_txt<fc3_bias_t, 1>(b8, "b8.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    fc1_result_t layer2_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    fc2_result_t layer5_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    fc3_result_t layer8_out[1];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    nnet::dense<input_t, fc1_result_t, config2>(x, layer2_out, w2, b2); // fc1

    nnet::layernormalize<fc1_result_t, layer3_t, config3>(layer2_out, layer3_out, s3, b3); // ln1

    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out); // relu1

    nnet::dense<layer4_t, fc2_result_t, config5>(layer4_out, layer5_out, w5, b5); // fc2

    nnet::layernormalize<fc2_result_t, layer6_t, config6>(layer5_out, layer6_out, s6, b6); // ln2

    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // relu2

    nnet::dense<layer7_t, fc3_result_t, config8>(layer7_out, layer8_out, w8, b8); // fc3

    nnet::sigmoid<fc3_result_t, result_t, sigmoid_config9>(layer8_out, layer9_out); // sigmoid

}

