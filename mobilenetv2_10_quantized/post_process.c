
/*
    Openedges Enlight Network Compiler
    Classification main
*/

#include "stdint.h"
#include "enlight_network.h"

#ifdef __i386__
#   include <stdio.h>
#   define post_process_log(...) do {printf(__VA_ARGS__);} while(0)
#else
#   define post_process_log(...) do {_printf(__VA_ARGS__);} while(0)
#endif

extern void classifier_init();

extern int classifier_run(
    enlight_act_tensor_t*   outputs,           /**< output conf tensor number   */
    int                     num_class          /**< class num                   */
    );


/** @brief Run Classification post-processing and return top-1 class ID.
 *
 *      Classification post-processing, classifier_run() finds and returns
 *      top-1 class ID from network output tensor data.
 *
 *  @param net_inst         instance Network instance.
 *  @param output_base      output tensor buffer base
 *  @param reserved 
 *
 *  @return number of class
 */
int run_post_process(
    void *net_inst,
    void *output_base,
    int  num_input,
    enlight_batch_cls_t *batch_cls_ids)
{
    enlight_network_t *inst;
    enlight_classification_t* cls_param;
    int num_class;

    int i, num_output, cls;

    inst = (enlight_network_t *)net_inst;
    cls_param = (enlight_classification_t*)inst->post_proc_extension;

    num_output = inst->output_tensors_num;

    for (i = 0; i < num_output; i++)
        inst->output_tensors[i]->base = output_base;

    num_class = enlight_class_get_num_class(cls_param);

    classifier_init();

    for (i = 0; i < num_input; i++) {
        cls = classifier_run(inst->output_tensors[i], num_class);
        batch_cls_ids->class_ids[i] = cls;
        // post_process_log("class[%d]: %d\n", i, cls);
    }

    return num_input;
}
