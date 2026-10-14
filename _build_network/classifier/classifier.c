#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "enlight_network.h"

#ifdef BARE_METAL_FW_DEV
    extern int _printf(const char *format, ...);
#   define  ENLIGHT_CLS_PRINT  _printf
#else
#   define  ENLIGHT_CLS_PRINT   printf
#endif

#define enlight_cls_err(...)  do {ENLIGHT_CLS_PRINT(__VA_ARGS__); } while(0)
#define enlight_cls_log(...)  do {ENLIGHT_CLS_PRINT(__VA_ARGS__); } while(0)

#ifdef ENLIGHT_CLS_DEBUG
#   define enlight_cls_dbg(...)  do {ENLIGHT_CLS_PRINT(__VA_ARGS__); } while(0)
#else
#   define enlight_cls_dbg(...)  do {} while(0)
#endif

void classifier_init()
{
    //
}

/**
 * @brief Return the classified class id.
 * 
 * @param tensor The output tensor address.
 * @param num_class The total class number of model.
 * @return The classified class ID.
 */
int classifier_run(
    enlight_act_tensor_t* tensor,
    int num_class
)
{
    int dims[4];
    int y, x, c;
    float max_val;
    int class_id;

    enlight_get_tensor_dimensions(tensor, dims);

    if ((num_class < dims[1]) || (dims[2] != 1) || (dims[3] != 1)) {
        enlight_cls_err("output dims error for classification: %d %d %d\n", dims[1], dims[2], dims[3]);
        return -1;
    }

    max_val = enlight_get_tensor_data_by_off(tensor, 0, 0, 0);
    class_id = 0;
    for (c = 1; c < num_class; c++) {
        float data = enlight_get_tensor_data_by_off(tensor, 0, 0, c);
        if (data > max_val) {
            max_val = data;
            class_id = c;
        }
    }

    enlight_cls_dbg("Top1 class: %d with %f\n", class_id, max_val);
    
    return class_id; 
}

