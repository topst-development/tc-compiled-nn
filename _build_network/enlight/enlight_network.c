/*
    Openedges Enlight Network Compiler

    Network APIs functions.
*/

#include "enlight_network.h"

char* enlight_get_name(enlight_network_t *inst)
{
    return inst->network_name;
}

enlight_postproc_t enlight_get_post_type(enlight_network_t *inst)
{
    return inst->post_proc;
}

int enlight_get_num_inputs(enlight_network_t *inst)
{
    return inst->input_tensor_num;
}

int* enlight_get_img_sizes(enlight_network_t *inst, int idx)
{
    return inst->img_sizes[idx];
}

long long enlight_get_conv_mac_num(enlight_network_t *inst)
{
    return inst->conv_mac_num;
}

int enlight_get_batch_size(enlight_network_t *inst)
{
    return inst->batch_size;
}

int enlight_get_output_tensor_num(enlight_network_t *inst)
{
    return inst->output_tensors_num;
}

enlight_act_tensor_t* enlight_get_output_tensor(enlight_network_t *inst, int idx)
{
    return inst->output_tensors[idx];
}

char* enlight_get_tensor_name(enlight_act_tensor_t *t)
{
    return t->name;
}

void enlight_get_tensor_dimensions(enlight_act_tensor_t* t, int dims[4])
{
    dims[0] = t->dimensions[0];
    dims[1] = t->dimensions[1];
    dims[2] = t->dimensions[2];
    dims[3] = t->dimensions[3];
}

int enlight_get_tensor_size(enlight_act_tensor_t* t)
{
    return t->size;
}

char* enlight_get_tensor_addr(enlight_act_tensor_t *t)
{
    return &t->base[t->buf];
}

int enlight_get_tensor_offset(enlight_act_tensor_t *t)
{
    return t->buf;
}

void enlight_get_tensor_qdata(enlight_act_tensor_t *t, int8_t *dst)
{
    int i;

    enlight_network_dbg("tensor %08x\n", t);
    enlight_network_dbg("base   %08x\n", t->base);
    enlight_network_dbg("off    %08x\n", t->buf);

    int8_t* src = (int8_t*)(&t->base[t->buf]);

    enlight_network_dbg("src %08x\n", src);

    if (t->output != 0) {
        for (i = 0; i < t->size; i++) {
            dst[i] = src[i];
        }
    }
    else {
        const int WORD_SIZE = 32; 

        int ii;
        int jj;
        int kk;
        int c;
        int h;
        int w;

        c = t->dimensions[1];
        w = t->dimensions[2];
        h = t->dimensions[3];

        for(ii = 0; ii < h; ii++) {
            for(jj = 0; jj < w; jj++) {
                for(kk = 0; kk < c; kk++) {

                    i = (kk % WORD_SIZE);
                    i += (WORD_SIZE * jj);
                    i += (WORD_SIZE * w * ii);
                    i += ((kk / WORD_SIZE) * h * w); 

                    dst[i] = src[i];
                }
            }
        }
    }
}

void enlight_get_tensor_data(enlight_act_tensor_t* t, float* dst)
{
    int i;

    enlight_network_dbg("tensor %08x\n", t);
    enlight_network_dbg("base   %08x\n", t->base);
    enlight_network_dbg("off    %08x\n", t->buf);

    int8_t* src = (int8_t*)(&t->base[t->buf]);

    enlight_network_dbg("src %08x\n", src);

    if (t->output != 0) {
        for (i = 0 ; i < t->size; i++) {
            dst[i] = (float)src[i] / t->scale;
        }
    }
    else {
        const int WORD_SIZE = 32; 

        int ii;
        int jj;
        int kk;
        int c;
        int h;
        int w;

        c = t->dimensions[1];
        w = t->dimensions[2];
        h = t->dimensions[3];

        for (ii = 0; ii < h; ii++) {
            for (jj = 0; jj < w; jj++) {
                for (kk = 0; kk < c; kk++) {

                    i = (kk % WORD_SIZE);
                    i += (WORD_SIZE * jj);
                    i += (WORD_SIZE * w * ii);
                    i += ((kk / WORD_SIZE) * h * w); 

                    dst[i] = (float)src[i] / t->scale;
                }
            }
        }
    }
}

float enlight_get_tensor_scale(enlight_act_tensor_t* t)
{
    return t->scale;
}

int8_t enlight_get_tensor_qdata_by_off(enlight_act_tensor_t* t, int y_off, int x_off, int c_off)
{
    const int WORD_SIZE = 32; 
    int idx;

    int8_t* src = (int8_t*)(&t->base[t->buf]);

    int* dims = t->dimensions;

    int c = (dims[1] + (WORD_SIZE - 1)) & 0xFFE0;
    int h = dims[2];
    int w = dims[3];

    if (t->output != 0) {
        idx = c_off + (c * x_off) + (c * w * y_off); 
    }
    else {
        idx = (c_off % WORD_SIZE) + (WORD_SIZE * x_off) + (WORD_SIZE * w * y_off) + ((c_off / WORD_SIZE) * h * w); 
    }

    enlight_network_dbg("output %3d\n", t->output);
    enlight_network_dbg("dims %3d %3d %3d %3d\n", dims[0], dims[1], dims[2], dims[3]);
    enlight_network_dbg("hwc  %3d %3d %3d\n", h, w, c);
    enlight_network_dbg("off y x c  %3d %3d %3d\n", y_off, x_off, c_off);
    enlight_network_dbg("idx %d\n", idx);
    enlight_network_dbg("base offset 0x%08x + 0x%08x = 0x%08x\n", src, (unsigned long)tensor->buf, &src[idx]);

    return src[idx];
}

float enlight_get_tensor_data_by_off(enlight_act_tensor_t* tensor, int y_off, int x_off, int c_off)
{
    float src;
    src = (float)enlight_get_tensor_qdata_by_off(tensor, y_off, x_off, c_off);
    return src / tensor->scale;
}

int enlight_get_tensor_quant_data_type(enlight_act_tensor_t* tensor)
{
    return tensor->data_type;
}

int enlight_get_buf_size(enlight_network_t *inst, enlight_buf_base_t buf_type)
{
    int size;

    switch (buf_type) {
    case ENLIGHT_BUF_CODE:
        size = inst->buf_sizes->code_buf_size;
        break;
    case ENLIGHT_BUF_WGHT:
        size = inst->buf_sizes->wght_buf_size;
        break;
    case ENLIGHT_BUF_WORK:
        size =  inst->buf_sizes->work_buf_size;
        break;
    case ENLIGHT_BUF_INPUT:
        size = inst->buf_sizes->input_buf_size;
        break;
    case ENLIGHT_BUF_OUTPUT:
        size = inst->buf_sizes->output_buf_size;
        break;
    default:
        enlight_network_err("wrong buf type:%d\n", buf_type);
        size = 0;
        break;
    }

    return size;
}

#ifdef BARE_METAL_FW_DEV
// APIs only for ENLIGHT toolkit developer
uint8_t* enlight_get_code_base(enlight_network_t *inst)
{
    return (uint8_t*)inst->npu_cmd_codes;
}
#endif

/******************************************************************************/
/*  SSD post process extension APIs                                           */
/******************************************************************************/
float enlight_ssd_get_thres_conf(enlight_ssd_detector_t* p)
{
    return p->th_conf;
}

float enlight_ssd_get_thres_iou(enlight_ssd_detector_t* p)
{
    return p->th_iou;
}

int enlight_ssd_get_num_class(enlight_ssd_detector_t* p)
{
    return p->num_class;
}

int enlight_ssd_get_num_box(enlight_ssd_detector_t* p)
{
    return p->num_box;
}

float* enlight_ssd_get_default_box_buf_addr(enlight_ssd_detector_t* p)
{
    return p->default_box;
}

void enlight_ssd_get_float_default_boxes(enlight_ssd_detector_t *p, float* buf)
{
    int i;

    for (i = 0; i < p->num_box; i++) {
        buf[i] = p->default_box[i];
    }
}

float enlight_ssd_get_conf_scales(enlight_ssd_detector_t *p, int idx)
{
    float scl;
    if (idx >= p->conf_tensors_num) {
        enlight_network_err("Wrong conf idx %d\n", idx);
        scl = 0.;
    }
    else {
        enlight_act_tensor_t* out_tensor = p->conf_tensors[idx];
        scl = out_tensor->scale;
    }
    return scl;
}

float enlight_ssd_get_loc_scales(enlight_ssd_detector_t *p, int idx)
{
    float scl;

    if (idx >= p->loc_tensors_num) {
        enlight_network_err("Wrong loc idx %d\n", idx);
        scl = 0.;
    }
    else {
        enlight_act_tensor_t* out_tensor = p->loc_tensors[idx];
        scl = out_tensor->scale;
    }
    return scl;
}

float* enlight_ssd_get_conf_logistic_table(enlight_ssd_detector_t* p, int output_index)
{
    return &p->conf_logistic_tables[output_index * 256];
}

enlight_objs_t*   enlight_ssd_get_objs_buf_base(enlight_ssd_detector_t *p)
{
    return &p->detected_objects;
}

float* enlight_ssd_get_loc_exp_table(enlight_ssd_detector_t *p, int output_index)
{
    return &p->loc_exp_tables[output_index * 256];
}

int enlight_ssd_get_loc_tensors(enlight_ssd_detector_t *p, enlight_act_tensor_t** tensor_buf)
{
    for (int i = 0; i < MAX_NUM_LOC; i++) {
        tensor_buf[i] = p->loc_tensors[i];
    }

    return p->loc_tensors_num;
}

int enlight_ssd_get_conf_tensors(enlight_ssd_detector_t *p, enlight_act_tensor_t** tensor_buf)
{
    for (int i = 0; i < MAX_NUM_CONF; i++) {
        tensor_buf[i] = p->conf_tensors[i];
    }
    return p->conf_tensors_num;
}

int enlight_ssd_get_bg_in_cls(enlight_ssd_detector_t *p)
{
    return p->bg_in_cls;
}

enlight_logistic_t enlight_ssd_get_logistic(enlight_ssd_detector_t *p)
{
    return p->logistic;
}

enlight_nms_t enlight_ssd_get_nms_method(enlight_ssd_detector_t *p)
{
    return p->nms_method;
}

int enlight_ssd_get_max_detection(enlight_ssd_detector_t *p)
{
    return p->max_detection;
}

/******************************************************************************/
/*  YOLO post process extension APIs                                           */
/******************************************************************************/
int enlight_yolo_get_softmax_use(enlight_yolo_detector_t* param)
{
    return param->softmax_use;
}

int enlight_yolo_get_loc_tensors(enlight_yolo_detector_t* param, enlight_act_tensor_t **tensor_buf)
{
    for (int i = 0; i < MAX_NUM_LOC; i++) {
        tensor_buf[i] = param->loc_tensors[i];
    }
    return param->loc_tensors_num;
}

int enlight_yolo_get_conf_tensors(enlight_yolo_detector_t* param, enlight_act_tensor_t **tensor_buf)
{
    for (int i = 0; i < MAX_NUM_CONF; i++) {
        tensor_buf[i] = param->conf_tensors[i];
    }
    return param->conf_tensors_num;
}

int enlight_yolo_get_obj_tensors(enlight_yolo_detector_t* param, enlight_act_tensor_t **tensor_buf)
{
    for (int i = 0; i < MAX_NUM_OBJ; i++) {
        tensor_buf[i] = param->obj_tensors[i];
    }
    return param->obj_tensors_num;
}

float enlight_yolo_get_thres_conf(enlight_yolo_detector_t *p)
{
    return p->th_conf;
}

float enlight_yolo_get_thres_iou(enlight_yolo_detector_t *p)
{
    return p->th_iou;
}

int enlight_yolo_get_num_class(enlight_yolo_detector_t *p)
{
    return p->num_class;
}

int* enlight_yolo_get_num_grids(enlight_yolo_detector_t *p)
{
    return p->num_grids;
}

int enlight_yolo_get_num_anchor(enlight_yolo_detector_t *p)
{
    return p->num_anchor;
}

float* enlight_yolo_get_default_box_buf_addr(enlight_yolo_detector_t *p)
{
    return p->default_box;
}

void enlight_yolo_get_float_default_boxes(enlight_yolo_detector_t *p, float* buf)
{
    int i;
    int num_box;

    num_box = 0;
    for (i = 0; i < p->loc_tensors_num; i++) {
        num_box += p->num_grids[i * 2] * p->num_grids[(i * 2) + 1];
    }

    for (i=0; i < num_box; i++) {
        buf[i] = p->default_box[i];
    }
}

float* enlight_yolo_get_output_exp_table(enlight_yolo_detector_t *p, int output_index)
{
    return &p->exp_tables[output_index * 256];
}

float* enlight_yolo_get_output_sig_table(enlight_yolo_detector_t *p, int output_index)
{
    return &p->sig_tables[output_index * 256];
}

enlight_objs_t*   enlight_yolo_get_objs_buf_base(enlight_yolo_detector_t *p)
{
    return &p->detected_objects;
}

int enlight_yolo_get_sigmoid_applied(enlight_yolo_detector_t *p)
{
    return p->sigmoid_applied;
}

int enlight_yolo_get_version(enlight_yolo_detector_t *p)
{
    return p->yolo_version;
}

int enlight_yolo_get_max_detection(enlight_yolo_detector_t *p)
{
    return p->max_detection;
}

enlight_nms_t enlight_yolo_get_nms_method(enlight_yolo_detector_t *p)
{
    return p->nms_method;
}

int enlight_yolo_get_dfl_reg_max(enlight_yolo_detector_t* param)
{
    return param->dfl_reg_max;
}

int enlight_class_get_output_tensors(enlight_classification_t* p, enlight_act_tensor_t **tensor_buf)
{
    for (int i = 0; i < MAX_NUM_OUTPUT; i++) {
        tensor_buf[i] = p->output_tensors[i];
    }
    return p->output_tensors_num;
}

int enlight_class_get_num_class(enlight_classification_t* p)
{
    return p->num_class;
}

int enlight_custom_get_output_tensors(enlight_custom_postproc_t* p, enlight_act_tensor_t **tensor_buf)
{
    for (int i = 0; i < MAX_NUM_OUTPUT; i++) {
        tensor_buf[i] = p->output_tensors[i];
    }
    return p->output_tensors_num;
}
