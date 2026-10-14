

#ifndef __NETWORK_H__
#define __NETWORK_H__

/*
    Openedges Enlight Network Compiler
*/

#include "enlight_data_type.h"
#include "enlight_network.h"
#ifdef BARE_METAL_FW_DEV
#    include "npu_cmd.h"
#endif


static enlight_act_tensor_t Input_0 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    150528,
    3,
    4,
    {1, 3, 224, 224},
    128.0000,
    0,
    0,
    "Input_0",
};

static enlight_act_tensor_t Conv2d_5 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x18800,
    100352,
    2,
    4,
    {1, 24, 56, 56},
    22.0003,
    0,
    0,
    "Conv2d_5",
};

static enlight_act_tensor_t Add_0 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    100352,
    2,
    4,
    {1, 24, 56, 56},
    16.8819,
    0,
    0,
    "Add_0",
};

static enlight_act_tensor_t Conv2d_11 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x18800,
    25088,
    2,
    4,
    {1, 32, 28, 28},
    28.9096,
    0,
    0,
    "Conv2d_11",
};

static enlight_act_tensor_t Add_1 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    25088,
    2,
    4,
    {1, 32, 28, 28},
    24.7894,
    0,
    0,
    "Add_1",
};

static enlight_act_tensor_t Add_2 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x6200,
    25088,
    2,
    4,
    {1, 32, 28, 28},
    19.1896,
    0,
    0,
    "Add_2",
};

static enlight_act_tensor_t Conv2d_20 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    12544,
    2,
    4,
    {1, 64, 14, 14},
    36.0232,
    0,
    0,
    "Conv2d_20",
};

static enlight_act_tensor_t Add_3 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x3100,
    12544,
    2,
    4,
    {1, 64, 14, 14},
    32.7791,
    0,
    0,
    "Add_3",
};

static enlight_act_tensor_t Add_4 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    12544,
    2,
    4,
    {1, 64, 14, 14},
    28.3519,
    0,
    0,
    "Add_4",
};

static enlight_act_tensor_t Add_5 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x4980,
    12544,
    2,
    4,
    {1, 64, 14, 14},
    24.0422,
    0,
    0,
    "Add_5",
};

static enlight_act_tensor_t Conv2d_32 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    18816,
    2,
    4,
    {1, 96, 14, 14},
    38.2584,
    0,
    0,
    "Conv2d_32",
};

static enlight_act_tensor_t Add_6 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x4980,
    18816,
    2,
    4,
    {1, 96, 14, 14},
    26.4337,
    0,
    0,
    "Add_6",
};

static enlight_act_tensor_t Add_7 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    18816,
    2,
    4,
    {1, 96, 14, 14},
    21.7493,
    0,
    0,
    "Add_7",
};

static enlight_act_tensor_t Conv2d_41 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x4980,
    7840,
    2,
    4,
    {1, 160, 7, 7},
    46.6881,
    0,
    0,
    "Conv2d_41",
};

static enlight_act_tensor_t Add_8 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    7840,
    2,
    4,
    {1, 160, 7, 7},
    34.1725,
    0,
    0,
    "Add_8",
};

static enlight_act_tensor_t Add_9 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0xf500,
    7840,
    2,
    4,
    {1, 160, 7, 7},
    17.5937,
    0,
    0,
    "Add_9",
};

static enlight_act_tensor_t Clip_54 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    62720,
    2,
    4,
    {1, 1280, 7, 7},
    21.1667,
    0,
    0,
    "Clip_54",
};

static enlight_act_tensor_t AdaptiveAvgPool2d_0 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0xf500,
    1280,
    2,
    4,
    {1, 1280, 1, 1},
    21.1667,
    0,
    0,
    "AdaptiveAvgPool2d_0",
};

static enlight_act_tensor_t Linear_0 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    1024,
    4,
    4,
    {1, 1000, 1, 1},
    4.7265,
    0,
    1,
    "Linear_0",
};

/* Buf description of Activation and Output*/
static enlight_act_tensor_t* all_output_tensors[19] = {
    &Input_0,
    &Conv2d_5,
    &Add_0,
    &Conv2d_11,
    &Add_1,
    &Add_2,
    &Conv2d_20,
    &Add_3,
    &Add_4,
    &Add_5,
    &Conv2d_32,
    &Add_6,
    &Add_7,
    &Conv2d_41,
    &Add_8,
    &Add_9,
    &Clip_54,
    &AdaptiveAvgPool2d_0,
    &Linear_0,
};


static enlight_classification_t post_ext_param = {
    1000,
    {&Linear_0,},
    1,
};

static enlight_buf_size_t network_buf_size = {
    //code_buf_size
    0x5400, 
    //wght_buf_size
    0x364200, 
    //input_buf_size
    0x24c00, 
    //output_buf_size
    0x400, 
    //work_buf_size
    0x31000, 
};

enlight_network_t network_instance = {
    //network_name
    "mobilenetv2-10_quantized.enlight",
    //post_proc
    ENLIGHT_POST_CLASSIFIER,

    //batch_size
    1,

    //input_num
    1,
    //img_sizes
    {{3, 224, 224},},
    //input_tensors
    {&Input_0,},

    //buf_sizes
    &network_buf_size,
    //post_proc_extension
    (void*)&post_ext_param,

    //output_tensors_num
    1,
    //output_tensors
    {&Linear_0,},
    //Conv MAC num
    319153024,

    //For Debug only
    //all_tensors_num,
    289,
    //all_tensors
    all_output_tensors,

    // For toolkit developer
#ifdef BARE_METAL_FW_DEV
    //npu_cmd_codes
    npu_cmd,
#endif
};

#endif /*__NETWORK_H__*/