

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
    0x3100,
    100352,
    2,
    4,
    {1, 24, 56, 56},
    21.9993,
    0,
    0,
    "Conv2d_5",
};

static enlight_act_tensor_t Conv2d_20 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0x0,
    12544,
    2,
    4,
    {1, 64, 14, 14},
    36.0277,
    0,
    0,
    "Conv2d_20",
};

static enlight_act_tensor_t Conv2d_41 = {
    ENLIGHT_DTYPE_INT8,
    (void*)0x0,
    0xf500,
    7840,
    2,
    4,
    {1, 160, 7, 7},
    46.8004,
    0,
    0,
    "Conv2d_41",
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
    4.7251,
    0,
    1,
    "Linear_0",
};

/* Buf description of Activation and Output*/
static enlight_act_tensor_t* all_output_tensors[7] = {
    &Input_0,
    &Conv2d_5,
    &Conv2d_20,
    &Conv2d_41,
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
    0xa400, 
    //wght_buf_size
    0x362a00, 
    //input_buf_size
    0x24c00, 
    //output_buf_size
    0x400, 
    //work_buf_size
    0x1b900, 
};

enlight_network_t network_instance = {
    //network_name
    "mobilenetv2_10_quantized.enlight",
    //post_proc
    ENLIGHT_POST_CLASSIFIER,
    //batch_size
    1,
    //img_sizes
    {3, 224, 224},
    //buf_sizes
    &network_buf_size,
    //post_proc_extension
    (void*)&post_ext_param,
    //output_tensors_num
    1,
    //output_tensors
    {&Linear_0,},
    //Conv MAC num
    394375360,

    //For Debug only
    //all_tensors_num,
    109,
    //all_tensors
    all_output_tensors,

    // For toolkit developer
#ifdef BARE_METAL_FW_DEV
    //npu_cmd_codes
    npu_cmd,
#endif
};

#endif /*__NETWORK_H__*/