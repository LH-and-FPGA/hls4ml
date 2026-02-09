#ifndef HEY_JARVIS_VERIFIER_H_
#define HEY_JARVIS_VERIFIER_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void hey_jarvis_verifier(
    input_t x[1536],
    result_t layer9_out[1]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
