#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "enlight_network.h"

#ifdef BARE_METAL_FW_DEV
    extern int _printf(const char *format, ...);
#   define  ENLIGHT_CUSTOM_PRINT  _printf
#else
#   define  ENLIGHT_CUSTOM_PRINT   printf
#endif

#define enlight_custom_err(...)  do {ENLIGHT_CUSTOM_PRINT(__VA_ARGS__); } while(0)
#define enlight_custom_log(...)  do {ENLIGHT_CUSTOM_PRINT(__VA_ARGS__); } while(0)

#ifdef ENLIGHT_CUSTOM_DEBUG
#   define enlight_custom_dbg(...)  do {ENLIGHT_CUSTOM_PRINT(__VA_ARGS__); } while(0)
#else
#   define enlight_custom_dbg(...)  do {} while(0)
#endif

void custom_postproc_init()
{
    //
}

/* Below code lines is just REFERENCE code              */
/*   in order to show how to acess output tensor        */
/*                                                      */
/* User can implement user own post-processing algorithm*/
/*   by updating this function                          */
int custom_postproc_run(
    int num_output,
    enlight_act_tensor_t** output_tensors     /**< output tensors          */
)
{
    int dims[4];
    int i, j, k, c, w, h, t, result;
    float data;
    enlight_act_tensor_t* tensor;

    enlight_custom_log("tensor num = %d\n", num_output);

    for (t = 0; t < num_output; t++) { 

        tensor = output_tensors[t];

        enlight_get_tensor_dimensions(tensor, dims);

        c = dims[1]; 
        h = dims[2];
        w = dims[3]; 

        enlight_custom_log("\ntensor HxWxC = %dx%dx%x\n", h, w, c);

        for (j = 0; j < h; j++) {
            for (i = 0; i < w; i++) {
                // print only first spatial position tensor for debug
                if (i == 0 && j == 0) {
                    enlight_custom_log("\n[%3d %3d, 0:%d]: ", j, i, c);
                }

                for (k = 0; k < c; k++) {

                    float data = enlight_get_tensor_data_by_off(tensor, j, i, k);

                    // print only first spatial position tensor for debug
                    if (i == 0 && j == 0) {
                        // enlight_custom_log(" %f", data);
                        enlight_custom_log(" %d", (int)data*1024);
                    }
                }
            }
        }
    }

    result = 0;

    return result; 
}

