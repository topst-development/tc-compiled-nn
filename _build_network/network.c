#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "enlight_data_type.h"
#include "enlight_network.h"
#include "network.h"

extern int run_post_process(
    void*   net_inst,
    void*   output_base,
    int     num_input,
    void*   reserved);

struct enlight_net{
    int     fd;
    int     cmd_size;
    int     weight_size;
    int     work_size;
    int     input_size;
    int     output_size;
    int     post_type;
    int*    img_size;
    long long conv_mac_num;
    int     batch_size;
    char    name[128];
    
    int     (*get_buffer_size)(int idx);
    int*    (*get_image_size)(void);
    char*   (*get_network_name)(void);
    int     (*get_post_type)(void);
    int     (*run_post)(void* output_base,int batch_size, void* detected_objects);

    int     (*get_output_tensor_num)(void);
    enlight_act_tensor_t*
            (*get_output_tensor)(int idx);
    int     (*get_tensor_size)(int idx);
    int     (*get_tensor_offset)(int idx);
    char*   (*get_tensor_name)(int idx);
};


int get_buffer_size(int idx)
{
    return enlight_get_buf_size(&network_instance, idx);
}

int* get_image_size(void)
{
    return enlight_get_img_sizes(&network_instance);
}

char* get_network_name(void)
{
    return enlight_get_name(&network_instance);
}

int get_post_type(void)
{
    return  enlight_get_post_type(&network_instance);
}

int run_post(void* output_base, int batch_size, void* detected_objects)
{
   return run_post_process((void*)&network_instance, (void*)output_base, batch_size, (void *)detected_objects);
}

int get_output_tensor_num(void)
{
    return enlight_get_output_tensor_num(&network_instance);
}

enlight_act_tensor_t * get_output_tensor(int idx)
{
   return  enlight_get_output_tensor(&network_instance,idx);
}

int get_output_tensor_size(int idx)
{
    enlight_act_tensor_t* t;
    t = enlight_get_output_tensor(&network_instance, idx);
    return enlight_get_tensor_size(t);
}

int get_tensor_offset(int idx)
{
    enlight_act_tensor_t* t;
    t = enlight_get_output_tensor(&network_instance, idx);
    return enlight_get_tensor_offset(t);
}

char* get_tensor_name(int idx)
{
    enlight_act_tensor_t* t;
    t = enlight_get_output_tensor(&network_instance, idx);

    return enlight_get_tensor_name(t);
}

int init_network(struct enlight_net* net)
{
    net->cmd_size    = enlight_get_buf_size(&network_instance, ENLIGHT_BUF_CODE);
    net->weight_size = enlight_get_buf_size(&network_instance, ENLIGHT_BUF_WGHT);
    net->work_size   = enlight_get_buf_size(&network_instance, ENLIGHT_BUF_WORK);
    net->input_size  = enlight_get_buf_size(&network_instance, ENLIGHT_BUF_INPUT);
    net->output_size = enlight_get_buf_size(&network_instance, ENLIGHT_BUF_OUTPUT);

    net->conv_mac_num = enlight_get_conv_mac_num(&network_instance);
    net->img_size     = enlight_get_img_sizes(&network_instance);
    net->batch_size   = enlight_get_batch_size(&network_instance);
    strcpy(net->name, enlight_get_name(&network_instance));
    net->post_type    = enlight_get_post_type(&network_instance);

    net->get_buffer_size  = get_buffer_size;
    net->get_image_size   = get_image_size;
    net->get_network_name = get_network_name;
    net->get_post_type    = get_post_type;

    net->run_post = run_post;

    net->get_output_tensor_num = get_output_tensor_num;
    net->get_output_tensor     = get_output_tensor;
    net->get_tensor_size       = get_output_tensor_size;
    net->get_tensor_offset     = get_tensor_offset;
    net->get_tensor_name       = get_tensor_name;

    return 0;
}
