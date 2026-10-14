#include "enlight_data_type.h"
#include "enlight_network.h"
#include "detector.h"

#ifdef DUMP
#   include <stdio.h>
#endif

#ifdef GDB_FW_DEBUG
extern enlight_objs_t gdb_fw_objects;
#endif

#ifdef DUMP
/**
 * @brief Debug function to dump regression result data for YOLO.
 * 
 * @param obj Object tensor data.
 * @param conf Confidence tensor array.
 * @param output_idx Output tensor index.
 * @param num_box Number of boxes.
 * @param sigmoid_applied Sigmoid applied flag.
 * @param num_class Number of classes.
 */
static void debug_regression_results(tensor_t* obj, tensor_t* conf[],
                                     int output_idx, int num_box,
                                     int sigmoid_applied, int num_class)
{
    //enlight_detector_dbg_fin();
    int i = output_idx;
    if (obj->dims[2]) {
        int8_t *obj_buf = (int8_t*)obj[i].buf;
        {
            char dump_name[256];
            sprintf(dump_name, "cand_obj_%d.bin", i);
            int num_e = num_box;
            float *d_arr = (float*)enlight_malloc(sizeof(float) * num_e);

            int j;
            for(j = 0; j < num_box; j++) {
                float *sig_tbl = obj[i].sig_tbl;
                float obj_val;
                if (sigmoid_applied)
                    obj_val = (float)obj_buf[j] / ENLIGHT_NPU_SIG_OUT_SCL;
                else
                    obj_val = sig_tbl[obj_buf[j]];

                d_arr[j] = obj_val;
            }
            enlight_dump_data_to_bin(dump_name, d_arr, 4, num_e);
            enlight_free(d_arr);
        }
    }

    {
        char dump_name[256];
        sprintf(dump_name, "cand_regression_%d.bin", i);
        int num_e = num_box * num_class;
        float *buf = conf[i]->buf;
        enlight_dump_data_to_bin(dump_name, buf, 4, num_e);
    }
    //enlight_detector_dbg_fout();
}

/**
 * @brief Debug function to dump regression object results.
 * 
 * @param conf Confidence tensor array.
 * @param output_idx Output tensor index.
 * @param num_class Number of classes.
 * @param num_box Number of boxes.
 */
static void debug_regression_obj_results(tensor_t* conf[], int output_idx, int num_class,
                                         int num_box)
{
    //enlight_detector_dbg_fin();
    int i = output_idx;
    char dump_name[256];
    sprintf(dump_name, "cand_regression_obj_%d.bin", i);
    int num_e = num_box * num_class;
    enlight_dump_data_to_bin(dump_name, conf[i]->buf, 4, num_e);
}

/**
 * @brief Debug function to dump NMS input data for specific class.
 * 
 * @param cand Candidate structure.
 * @param c Class index.
 */
static void debug_nms_input(cand_f_t* cand, int c)
{
    //enlight_detector_dbg_fin();
    int num_cand = cand->n;
    int i;
    if (num_cand) {
        float *dump_box = (float*)enlight_malloc(sizeof(float) * num_cand * 4);
        float *p = dump_box;
        for (i = 0 ; i < num_cand; i++){
            box_f_t *b = cand->box + i;

            *p++ = b->x_min;
            *p++ = b->y_min;
            *p++ = b->x_max;
            *p++ = b->y_max;
        }
        char dump_name[256];
        sprintf(dump_name, "cand_loc_%d.bin", c);
        enlight_dump_data_to_bin(dump_name, dump_box, 4, num_cand * 4);
        enlight_free(dump_box);
    }
    //enlight_detector_dbg_fout();
}

/**
 * @brief Debug function to dump detection results.
 * 
 * @param r1 Detection result structure.
 * @param nms_method NMS method type.
 */
static void debug_detection_results(detection_f_t* r1, enlight_nms_t nms_method)
{
    //enlight_detector_dbg_fin();
    int num_obj = r1->n;
    if (num_obj) {
        box_f_t *box_arr = r1->box;
        float   *score_arr = r1->score;
        uint16_t *detected_cls = r1->cls;

        char dump_name[256];

        int i = 0;
        int j = 0;
        if (nms_method == NMS_DEFAULT) {


            while (i < num_obj) {
                uint16_t pop_cls = detected_cls[i];
                int num_cls = 1;

                i++;
                while (i < num_obj && (pop_cls == detected_cls[i])) {
                    num_cls++;
                    i++;
                }

                sprintf(dump_name, "detection_loc_result_%d.bin", pop_cls);
                float *dump_loc = (float*)enlight_malloc(sizeof(float) * num_cls * 4);
                float *p = dump_loc;

                for (j = 0; j < num_cls; j++)
                {
                    box_f_t *b = box_arr + j;

                    *p++ = b->x_min;
                    *p++ = b->y_min;
                    *p++ = b->x_max;
                    *p++ = b->y_max;
                }

                enlight_dump_data_to_bin(dump_name, dump_loc, 4, num_cls * 4);
                enlight_free(dump_loc);

                sprintf(dump_name, "detection_scr_result_%d.bin", pop_cls);
                enlight_dump_data_to_bin(dump_name, score_arr, 4, num_cls);

                score_arr += num_cls;
                box_arr += num_cls;
            }
        }
        else {
            sprintf(dump_name, "detection_loc_result_0.bin");
            float *dump_loc = (float*)enlight_malloc(sizeof(float) * num_obj * 4);
            float *p = dump_loc;

            for (j = 0; j < num_obj; j++)
            {
                box_f_t *b = box_arr + j;

                *p++ = b->x_min;
                *p++ = b->y_min;
                *p++ = b->x_max;
                *p++ = b->y_max;
            }
            enlight_dump_data_to_bin(dump_name, dump_loc, 4, num_obj * 4);
            enlight_free(dump_loc);

            sprintf(dump_name, "detection_scr_result_0.bin");
            enlight_dump_data_to_bin(dump_name, score_arr, 4, num_obj);

        }
    }
    //enlight_detector_dbg_fout();
}

/**
 * @brief Debug function to dump FastNMS input data.
 * 
 * @param cand Candidate structure.
 */
static void debug_fastnms_inputs(cand_f_t* cand)
{
    //enlight_detector_dbg_fin();
    int i;
    {
        int num_cand = cand->n;
        if (num_cand) {
            float *dump_box = cand-> score;
            char dump_name[256];
            sprintf(dump_name, "cand_scr_0.bin");
            enlight_dump_data_to_bin(dump_name, dump_box, 4, num_cand);
        }
    }

    {
        int num_cand = cand->n;
        if (num_cand) {
            float *dump_box = (float*)enlight_malloc(sizeof(float) * num_cand * 4);
            float *p = dump_box;
            for (i = 0 ; i < num_cand; i++){
                box_f_t *b = cand->box + i;

                *p++ = b->x_min;
                *p++ = b->y_min;
                *p++ = b->x_max;
                *p++ = b->y_max;
            }
            char dump_name[256];
            sprintf(dump_name, "cand_loc_0.bin");
            enlight_dump_data_to_bin(dump_name, dump_box, 4, num_cand * 4);
            enlight_free(dump_box);
        }
    }
    //enlight_detector_dbg_fout();
}

/**
 * @brief Debug function to dump DFL output data.
 * 
 * @param i Output index.
 * @param smax_tensor Softmax tensor.
 * @param conv_tensor Convolution tensor.
 */
static void debug_dfl_outputs(int i, tensor_t* smax_tensor, tensor_t* conv_tensor)
{
    tensor_t* target = &smax_tensor[i];
    char dump_name[256];
    sprintf(dump_name, "dfl_softmax_%d.bin", i);
    float* sbuf = (float*)target -> buf;
    enlight_dump_data_to_bin(dump_name, sbuf, 4, target->num_ele);

    target = &conv_tensor[i];
    sprintf(dump_name, "dfl_conv1x1_%d.bin", i);
    float *cbuf = (float*)target -> buf;
    enlight_dump_data_to_bin(dump_name, cbuf, 4, target->num_ele);
}
#endif

/**
 * @brief Build bounding box information for YOLO detection.
 * 
 * @param box Output bounding box structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param idx Index of the box to process.
 * @param num_grid Grid dimensions.
 * @param sigmoid_applied Sigmoid applied flag.
 */
static void build_box_info(
    box_f_t*    box,
    tensor_t*   loc,
    tensor_t*   prior,
    int         idx,
    int*        num_grid,
    int         sigmoid_applied)
{
    //enlight_detector_dbg_fin();

    int loc_idx = idx * loc->dims[2];
    int box_idx = idx * prior->dims[1];

    int8_t* _loc = (int8_t*)loc->buf + loc_idx;
    float* _prior = (float*)prior->buf + box_idx;

    float* sig_tbl = loc->sig_tbl;
    float* exp_tbl = loc->exp_tbl;

    float x, y, w, h, anchor_x, anchor_y, anchor_w, anchor_h;

    if (!sigmoid_applied) {
        x = sig_tbl[_loc[0]];
        y = sig_tbl[_loc[1]];
    }
    else{
        x = (float)_loc[0] / ENLIGHT_NPU_SIG_OUT_SCL;
        y = (float)_loc[1] / ENLIGHT_NPU_SIG_OUT_SCL;
    }

    // (x / num_grid)
    // (y / num_grid)
    x /= num_grid[1];
    y /= num_grid[0];

    anchor_x = _prior[0];
    anchor_y = _prior[1];
    anchor_w = _prior[2];
    anchor_h = _prior[3];

    x = x + anchor_x;
    y = y + anchor_y;

    w =  exp_tbl[_loc[2]] * anchor_w;
    h =  exp_tbl[_loc[3]] * anchor_h;

    w /= 2.;
    h /= 2.;

    box->x_min = x - w;
    box->x_max = x + w;
    box->y_min = y - h;
    box->y_max = y + h;

    box->area = 4 * w * h;

    //enlight_detector_dbg_fout();
}

/**
 * @brief Build bounding box information for YOLOv5.
 * 
 * @param box Output bounding box structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param idx Index of the box to process.
 * @param num_grid Grid dimensions.
 */
static void build_box_info_v5(
    box_f_t*    box,
    tensor_t*   loc,
    tensor_t*   prior,
    int         idx,
    int*        num_grid)

{

    //enlight_detector_dbg_fin();

    int loc_idx = idx * loc->dims[2];
    int box_idx = idx * prior->dims[1];

    int8_t* _loc = (int8_t*)loc->buf + loc_idx;
    float* _prior = (float*)prior->buf + box_idx;

    float* sig_tbl = loc->sig_tbl;

    float x, y, w, h, anchor_x, anchor_y, anchor_w, anchor_h;

    x = sig_tbl[_loc[0]];
    y = sig_tbl[_loc[1]];
    w = sig_tbl[_loc[2]];
    h = sig_tbl[_loc[3]];

    // (x / num_grid)
    // (y / num_grid)
    x *= 2;
    y *= 2;
    x /= num_grid[1];
    y /= num_grid[0];

    anchor_x = _prior[0];
    anchor_y = _prior[1];
    anchor_w = _prior[2];
    anchor_h = _prior[3];

    x = x + anchor_x;
    y = y + anchor_y;

    w = 4 * w * w * anchor_w;
    h = 4 * h * h * anchor_h;

    w /= 2.;
    h /= 2.;

    box->x_min = x - w;
    box->x_max = x + w;
    box->y_min = y - h;
    box->y_max = y + h;

    box->area = 4 * w * h;

    //enlight_detector_dbg_fout();
}

/**
 * @brief Build bounding box information for YOLOv6.
 * 
 * @param box Output bounding box structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param idx Index of the box to process.
 * @param num_grid Grid dimensions.
 * @param dfl_applied DFL applied flag.
 */
static void build_box_info_v6(
    box_f_t*    box,
    tensor_t*   loc,
    tensor_t*   prior,
    int         idx,
    int*        num_grid,
    int         dfl_applied)

{

    //enlight_detector_dbg_fin();

    int loc_idx = idx * loc->dims[2];
    int box_idx = idx * prior->dims[1];

    float* _prior = (float*)prior->buf + box_idx;

    float x1, y1, x2, y2, grid_cx, grid_cy;
    float x, y, w, h;

    if (dfl_applied) {
        float* _loc = (float*)loc->buf + loc_idx;
        x1 = _loc[0];
        y1 = _loc[1];
        x2 = _loc[2];
        y2 = _loc[3];
    }
    else {
        int8_t* _loc = (int8_t*)loc->buf + loc_idx;
        x1 = (float)_loc[0] / loc->scl;
        y1 = (float)_loc[1] / loc->scl;
        x2 = (float)_loc[2] / loc->scl;
        y2 = (float)_loc[3] / loc->scl;
    }

    grid_cx = _prior[0];
    grid_cy = _prior[1];

    x = ((x2 / num_grid[1]) - (x1 / num_grid[1])) / 2 + grid_cx;
    y = ((y2 / num_grid[0]) - (y1 / num_grid[0])) / 2 + grid_cy;
    w = ((x2 / num_grid[1]) + (x1 / num_grid[1]));
    h = ((y2 / num_grid[0]) + (y1 / num_grid[0]));

    //enlight_detector_dbg("x: %f, y: %f, w: %f, h: %f\n", x, y, w, h);
    //enlight_detector_dbg("grid_cx: %f, grid_cy: %f\n", grid_cx, grid_cy);

    w /= 2.;
    h /= 2.;

    box->x_min = x - w;
    box->x_max = x + w;
    box->y_min = y - h;
    box->y_max = y + h;

    box->area = 4 * w * h;

    //enlight_detector_dbg_fout();
}

/**
 * @brief Build bounding box information for YOLOX.
 * 
 * @param box Output bounding box structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param idx Index of the box to process.
 * @param num_grid Grid dimensions.
 */
static void build_box_info_x(
    box_f_t*    box,
    tensor_t*   loc,
    tensor_t*   prior,
    int         idx,
    int*        num_grid
    )
{
    //enlight_detector_dbg_fin();

    int loc_idx = idx * loc->dims[2];
    int box_idx = idx * prior->dims[1];

    int8_t* _loc = (int8_t*)loc->buf + loc_idx;
    float* _prior = (float*)prior->buf + box_idx;

    float* exp_tbl = loc->exp_tbl;

    float x, y, w, h, grid_cx, grid_cy;

    grid_cx = _prior[0];
    grid_cy = _prior[1];

    x = (float)_loc[0] / (num_grid[1] * loc->scl) + grid_cx;
    y = (float)_loc[1] / (num_grid[0] * loc->scl) + grid_cy;
    w = *((float*)&exp_tbl[_loc[2]]) / num_grid[1];
    h = *((float*)&exp_tbl[_loc[3]]) / num_grid[0];

    //enlight_detector_dbg("x: %f, y: %f, w: %f, h: %f\n", x, y, w, h);
    //enlight_detector_dbg("grid_cx: %f, grid_cy: %f\n", grid_cx, grid_cy);

    w /= 2.;
    h /= 2.;

    box->x_min = x - w;
    box->x_max = x + w;
    box->y_min = y - h;
    box->y_max = y + h;

    box->area = 4 * w * h;

    //enlight_detector_dbg_fout();
}

/**
 * @brief Apply thresholding for specific class in YOLO.
 *
 * input: { batch, boxes, clses }
 * output: { boxes { coordinate, score } }
 * 
 * @param score Score tensor array.
 * @param cand Output candidate structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param cls Target class index.
 * @param th Confidence threshold.
 * @param num_grids Grid dimensions array.
 * @param num_output Number of outputs.
 * @param sigmoid_applied Sigmoid applied flag.
 * @param yolo_version YOLO version.
 * @param dfl_applied DFL applied flag.
 */
static void thresholding(tensor_t** score, cand_f_t* cand,
                         tensor_t* loc, tensor_t* prior,
                         int cls, float th, int* num_grids, int num_output,
                         int sigmoid_applied, int yolo_version,
                         int dfl_applied)
{
    //enlight_detector_dbg_fin();

    float* _score;
    int num_box;
    int num_class = score[0]->dims[2];

    int i, box;
    int n = 0;

    //enlight_detector_dbg("yolo_version: %d\n", yolo_version);

    for (i = 0; i < num_output; i++) {
        _score = (float*)score[i]->buf + cls;
        num_box = score[i]->dims[1];

        for (box = 0; box < num_box; box++, _score += num_class) {
            if (*_score > th) {
                //enlight_detector_dbg("[output_idx: %d] threshold: %f, score: %f,  cls: %d\n", i, th, *_score, cls);

                if (yolo_version < 5)
                    build_box_info(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2], sigmoid_applied);
                else if (yolo_version == 5 || yolo_version == 7)
                    build_box_info_v5(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2]);
                else if (yolo_version == 6 || yolo_version == 8)
                    build_box_info_v6(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2], dfl_applied);
                else if (yolo_version == 10)
                    build_box_info_x(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2]);
                cand->cls[n] = cls;
                cand->score[n] = *_score;
                n++;
            }
        }
    }
    cand->n = n;

    //enlight_detector_dbg_fout();
}

/**
 * @brief Apply thresholding by default box method in YOLO.
 * 
 * @param score Score tensor array.
 * @param cand Output candidate structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param th Confidence threshold.
 * @param num_grids Grid dimensions array.
 * @param num_output Number of outputs.
 * @param sigmoid_applied Sigmoid applied flag.
 * @param yolo_version YOLO version.
 * @param dfl_applied DFL applied flag.
 */
static void thresholding_by_default_box(tensor_t** score, cand_f_t* cand,
                                        tensor_t* loc, tensor_t* prior,
                                        float th, int *num_grids, int num_output,
                                        int sigmoid_applied, int yolo_version,
                                        int dfl_applied)
{
    float* _score;

    int num_class = score[0]->dims[2];
    int num_box;

    int i, box, n, cls;

    for (i = 0, n = 0; i < num_output; i++) {
        _score = (float*)score[i]->buf;
        num_box = score[i]->dims[1];

        for (box = 0; box < num_box; box++, _score += num_class) {
            int max_idx = 0;
            float max_val = _score[0];
            for (cls = 1; cls < num_class; cls++){
                if (max_val < _score[cls]) {
                    max_val = _score[cls];
                    max_idx = cls;
                }
            }

            if (max_val >= th) {
                //enlight_detector_dbg("[output_idx:%d] box_idx:%d, max_val:%f\n", i, box, max_val);
                if (yolo_version < 5)
                    build_box_info(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2], sigmoid_applied);
                else if (yolo_version == 5 || yolo_version == 7)
                    build_box_info_v5(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2]);
                else if (yolo_version == 6 || yolo_version == 8)
                    build_box_info_v6(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2], dfl_applied);
                else if (yolo_version == 10)
                    build_box_info_x(&cand->box[n], &loc[i], &prior[i], box, &num_grids[i*2]);
                cand->cls[n] = max_idx;
                cand->score[n] = max_val;
                n++;
            }
        }
    }
    cand->n = n;
}

/**
 * @brief YOLO post-processing pipeline for object detection.
 * 
 * @param score Score tensor data.
 * @param loc Location tensor data.
 * @param pri Prior anchor box data.
 * @param obj Object tensor data.
 * @param th_conf Confidence threshold.
 * @param th_nms NMS threshold.
 * @param num_grids Grid dimensions array.
 * @param num_output Number of outputs.
 * @param softmax_use Softmax usage flag.
 * @param sigmoid_applied Sigmoid applied flag.
 * @param yolo_version YOLO version.
 * @param nms_method NMS method type.
 * @param dfl_applied DFL applied flag.
 * @return Pointer to detection results.
 */
static detection_f_t* yolo_post_process(
    tensor_t*   score,
    tensor_t*   loc,
    tensor_t*   pri,
    tensor_t*   obj,
    float       th_conf,
    float       th_nms,
    int*        num_grids,
    int         num_output,
    int         softmax_use,
    int         sigmoid_applied,
    int         yolo_version,
    enlight_nms_t nms_method,
    int         dfl_applied
)
{
    //enlight_detector_dbg_fin();

    detection_f_t* detection;
    int i, j, c;
    int num_class, num_box;
    int total_num_box = 0;
    float *conf_buf;
    int8_t *obj_buf;
    tensor_t* conf[MAX_NUM_OUTPUT];

    int num_ele = 0;
    for(i = 0; i < num_output; i++)
        num_ele += score[i].num_ele;

    detection = alloc_detection_float(num_ele);
    num_class = score[0].dims[2];

    int has_obj = obj[0].dims[2] ? 1 : 0;

    for(i =0; i < num_output; i++) {
        num_box = score[i].dims[1];
        total_num_box += num_box;

        conf[i] = alloc_tensor(ENLIGHT_DTYPE_FP32, score[i].num_dim, score[i].dims, 0);

        if (!num_box)
            continue;

        // Sigmoid already applied in HW
        if (sigmoid_applied) {
            float* dst = (float*)conf[i]->buf;
            uint8_t* src = (uint8_t*)score[i].buf;
            for(j = 0; j < score[i].num_ele; j++)
                dst[j] = (float)src[j] / ENLIGHT_NPU_SIG_OUT_SCL;
        }
        else {
            if (softmax_use)
                softmax_float(&score[i], conf[i], score[i].dims[2]);
            else
                sigmoid_float(&score[i], conf[i], score[i].dims[2]);
        }

        conf_buf = (float*)conf[i]->buf;
        obj_buf = (int8_t*)obj[i].buf;

#ifdef DUMP
        debug_regression_results(obj, conf, i, num_box, sigmoid_applied, num_class);
#endif

        // conf = conf * objecness if objectess exists
        if (has_obj) {
            for(j = 0; j < num_box; j++) {
                float *sig_tbl = obj[i].sig_tbl;
                float obj_val;
                if (sigmoid_applied)
                    obj_val = (float)obj_buf[j] / ENLIGHT_NPU_SIG_OUT_SCL;
                else
                    obj_val = sig_tbl[obj_buf[j]];

                for(c = 0; c < num_class; c++) {
                    float conf_val = *conf_buf;
                    float conf_obj = conf_val * obj_val;
                    *conf_buf++ = conf_obj;
                }
            }
#ifdef DUMP
            debug_regression_obj_results(conf, i, num_class, num_box);
#endif
        }
    }

    if (nms_method == NMS_DEFAULT) {
        for (c = 0; c < num_class; c++) {
            cand_f_t* cand = (cand_f_t*)alloc_candidate_float(total_num_box);

            thresholding(conf, cand, loc, pri, c, th_conf, num_grids,
                         num_output, sigmoid_applied, yolo_version,
                         dfl_applied);

#ifdef DUMP
            debug_nms_input(cand, c);
#endif

            nms_float(detection, cand, c, th_nms);

            free_candidate_float(cand);
        }
    }
    else {
        cand_f_t* cand = (cand_f_t*)alloc_candidate_float(total_num_box);

        thresholding_by_default_box(conf, cand, loc, pri, th_conf,
                                    num_grids, num_output,
                                    sigmoid_applied, yolo_version,
                                    dfl_applied);

#ifdef DUMP
        debug_fastnms_inputs(cand);
#endif

        fastnms_float(detection, cand, th_nms);
        free_candidate_float(cand);

    }


    for(i = 0; i < num_output; i++)
        free_tensor(conf[i]);

    //enlight_detector_dbg_fout();

    return detection;
}

/**
 * @brief Get candidate boxes from YOLO output tensors.
 * 
 * @param loc_outputs Location output tensors.
 * @param conf_outputs Confidence output tensors.
 * @param obj_outputs Object output tensors.
 * @param cand_bufs Candidate buffer arrays.
 * @param cand_nums Candidate counts.
 * @param sig_tbls Sigmoid lookup tables.
 * @param num_output Number of outputs.
 * @param num_grids Grid dimensions.
 * @param num_anchor Number of anchors.
 * @param num_class Number of classes.
 * @param th_conf Confidence threshold.
 * @param out_tensor_divided Tensor division flag.
 * @param sigmoid_applied Sigmoid applied flag.
 * @param len_obj Object length.
 * @param len_loc Location length.
 */
static void get_candidate(
    enlight_act_tensor_t**  loc_outputs, /* loc_outputs if loc and conf tensor buffer
                                            if out_tensor_divided is = 0*/
    enlight_act_tensor_t**  conf_outputs,
    enlight_act_tensor_t**  obj_outputs,
    int**           cand_bufs,
    int*            cand_nums,
    float**         sig_tbls,
    int             num_output,
    int*            num_grids,
    int             num_anchor,
    int             num_class,
    float           th_conf,
    int             out_tensor_divided,
    int             sigmoid_applied,
    int             len_obj,
    int             len_loc
    )
{
    //enlight_detector_dbg_fin();

    int box_data_len;
    int cand_cnt;
    int output_idx, conf_output_idx;

    box_data_len = len_loc + len_obj + num_class;

    if (out_tensor_divided == 1)
        box_data_len = len_obj + num_class;
    else if (out_tensor_divided == 2)
        box_data_len = len_obj;

    for(output_idx = 0; output_idx < num_output; output_idx++) {
        float *sig_tbl;

        int *cand_buf;
        int grid_h, grid_w, gi, gj, k;
        cand_cnt = 0;

        sig_tbl = sig_tbls[output_idx];
        cand_buf = cand_bufs[output_idx];

        conf_output_idx = output_idx;

        enlight_act_tensor_t* _output;

        if (out_tensor_divided == 1)
            _output = conf_outputs[conf_output_idx];
        else if (out_tensor_divided == 2)
            _output = obj_outputs[conf_output_idx];
        else
            _output = loc_outputs[conf_output_idx];

        if (_output->output == 0) {
            enlight_network_err("enlight act data format not supported\n");
        }

        int8_t* output_base = (int8_t*)(&_output->base[_output->buf]);
        int* output_dims = _output->dimensions;
        int output_c = (output_dims[1] + (32 - 1)) & 0xFFE0;
        int output_w = output_dims[3];

        grid_h = num_grids[2 * output_idx];
        grid_w = num_grids[(2 * output_idx) + 1];

        for (gi = 0; gi < grid_h; gi++) {
            for (gj = 0; gj < grid_w; gj++) {
                int cand_pos = num_anchor * ((grid_w * gi) + gj);
                
                for (k = 0; k < num_anchor; k++) {
                    float obj;
                    int x_off = gj;
                    int y_off = gi;
                    int c_off;
                    int8_t src;

                    if (out_tensor_divided >= 1)
                        c_off = box_data_len * k;
                    else
                        c_off = (box_data_len * k) + len_loc;

                    if (len_obj) {
                        int8_t* output_buf = output_base + c_off + (output_c * x_off) + (output_c * output_w * y_off); 
                        src = *output_buf;
                        //src = enlight_get_tensor_qdata_by_off(_output, y_off, x_off, c_off);

                        if (sigmoid_applied)
                            obj = (float)src / ENLIGHT_NPU_SIG_OUT_SCL;
                        else
                            obj = sig_tbl[src];

                        if (obj >= th_conf) {
                            cand_buf[cand_cnt++] = cand_pos + k;
                        }
                    }
                    else { //objectness not included
                        int i;
                        int8_t* output_buf = output_base + c_off + (output_c * x_off) + (output_c * output_w * y_off); 
                        for (i = 0; i < num_class; i++) {
                            src = output_buf[i];
                            //src = enlight_get_tensor_qdata_by_off(_output, y_off, x_off, c_off + i);

                            if (sigmoid_applied)
                                obj = (float)src / ENLIGHT_NPU_SIG_OUT_SCL;
                            else
                                obj = *((float*)&sig_tbl[src]);

                            if (obj >= th_conf) {
                                cand_buf[cand_cnt++] = cand_pos + k;
                                break;
                            }
                        }
                    }
                }
            }
        }

        cand_nums[output_idx] = cand_cnt;
    }

    //enlight_detector_dbg_fout();
}

/**
 * @brief Decompose YOLO output tensors into location, confidence, and prior data.
 * 
 * @param loc_outputs Location output tensors.
 * @param conf_outputs Confidence output tensors.
 * @param obj_outputs Object output tensors.
 * @param def_box_addr Default box addresses.
 * @param num_anchor Number of anchors.
 * @param num_class Number of classes.
 * @param num_output Number of outputs.
 * @param num_grids Grid dimensions.
 * @param cand_buf_addr Candidate buffer addresses.
 * @param cand_nums Candidate counts.
 * @param out_tensor_divided Tensor division flag.
 * @param loc Location tensors.
 * @param conf Confidence tensors.
 * @param prior Prior tensors.
 * @param obj Object tensors.
 * @param dfl_reg_max DFL regression maximum.
 */
static void decompose_output(
    enlight_act_tensor_t**  loc_outputs,
    enlight_act_tensor_t**  conf_outputs,
    enlight_act_tensor_t**  obj_outputs,
    float**                 def_box_addr,
    int                     num_anchor,
    int                     num_class,
    int                     num_output,
    int*                    num_grids,
    int**                   cand_buf_addr,
    int*                    cand_nums,
    int                     out_tensor_divided,
    tensor_t*               loc,
    tensor_t*               conf,
    tensor_t*               prior,
    tensor_t*               obj,
    int                     dfl_reg_max
)
{
    //enlight_detector_dbg_fin();

    int i, j, k;

    //  ouput tensor shape
    //      output[h * w][box_data_size * 5 + 0's padding] = ouput[13 * 13][128]
    const int len_coord = 4;
    const int len_obj = obj[0].dims[2];
    const int len_pri = prior[0].dims[1];
    int len_loc = loc[0].dims[2]; //xywh except yolov6, yolov8
    if (dfl_reg_max)
        len_loc = dfl_reg_max * len_coord;

    // select default box, which conf > threshould_conf
    for(i = 0 ; i < num_output; i++) {
        float* prior_dst = prior[i].buf;
        int cand_num = cand_nums[i];
        int* cand_buf = cand_buf_addr[i];

        for(j = 0; j < cand_num; j++) {
            float *src = def_box_addr[i] + (len_pri * cand_buf[j]);

            for(k = 0; k < len_pri; k++) {
                float val = *src++;
                *prior_dst++ = val;
            }
        }
    }

    // declare temporal loc variable for dfl in yolov6, yolov8
    tensor_t* loc_temp = loc;
    tensor_t* loc_smax = NULL;
    void* buf_temp = NULL;
    if (dfl_reg_max) {
        loc_temp = (tensor_t*)enlight_malloc(sizeof(tensor_t) * num_output * 2);
        loc_smax = loc_temp + num_output;

        int total_ele = 0;
        for (i = 0; i < num_output; i++) {
            int temp_loc_dims[3] = {1, loc[i].dims[1] * len_coord, dfl_reg_max};
            init_tensor(&loc_temp[i], ENLIGHT_DTYPE_INT8, 3, temp_loc_dims, loc[i].scl,
                        loc[i].exp_tbl, loc[i].sig_tbl, NULL);
            init_tensor(&loc_smax[i], ENLIGHT_DTYPE_FP32, 3, temp_loc_dims, 1,
                        NULL, NULL, NULL);

            total_ele += loc_temp[i].num_ele;
        }
        buf_temp = enlight_malloc((ENLIGHT_DTYPE_FP32 + ENLIGHT_DTYPE_INT8) * total_ele);
        void* ip = buf_temp;
        void* fp = (int8_t*)buf_temp + total_ele;

        for (i = 0; i < num_output; i++) {
            loc_temp[i].buf = ip;
            loc_smax[i].buf = fp;

            ip = (int8_t*)ip + loc_temp[i].num_ele;
            fp = (float*)fp + loc_smax[i].num_ele;
        }
    }

    // copy loc, obj, cls conf
    int loc_box_len = len_loc + len_obj + num_class;
    int conf_box_len = len_loc + len_obj + num_class;
    int obj_box_len = len_loc + len_obj + num_class;

    if (out_tensor_divided == 1) {
        loc_box_len = len_loc;
        conf_box_len = len_obj + num_class;
        obj_box_len = len_obj + num_class;
    }
    else if (out_tensor_divided == 2) {
        loc_box_len = len_loc;
        conf_box_len = num_class;
        obj_box_len = len_obj;
    }


    for(i = 0; i < num_output; i++) {

        int8_t* coor_dst = loc_temp[i].buf;
        int8_t* obj_dst = obj[i].buf;
        int8_t* conf_dst = conf[i].buf;

        int cand_num = cand_nums[i];
        int* cand_buf = cand_buf_addr[i];

        enlight_act_tensor_t* loc_output;
        enlight_act_tensor_t* conf_output;
        enlight_act_tensor_t* obj_output;

        if (out_tensor_divided == 1) {
            conf_output = conf_outputs[i];
            obj_output = conf_outputs[i];
            loc_output = loc_outputs[i];
        }
        else if (out_tensor_divided == 2) {
            conf_output = conf_outputs[i];
            obj_output = obj_outputs[i];
            loc_output = loc_outputs[i];
        }
        else {
            conf_output = loc_outputs[i];
            obj_output = loc_outputs[i];
            loc_output = loc_outputs[i];
        }

        for(j = 0; j < cand_num; j++) {
            int grid_idx, box_idx;

            grid_idx = cand_buf[j] / num_anchor;
            box_idx = cand_buf[j] % num_anchor;

            // local
            for(k = 0; k < len_loc; k++) {
                int y_off = grid_idx / num_grids[2 * i + 1];
                int x_off = grid_idx % num_grids[2 * i + 1];
                int c_off = loc_box_len * box_idx + k;

                int8_t val = enlight_get_tensor_qdata_by_off(loc_output, y_off, x_off, c_off);

                *coor_dst++ = val;
            }

            // objectness
            for(k = 0; k < len_obj; k++) {
                int y_off = grid_idx / num_grids[2 * i + 1];
                int x_off = grid_idx % num_grids[2 * i + 1];
                int c_off = obj_box_len * box_idx + len_coord + k;
                if (out_tensor_divided >= 1)
                    c_off = obj_box_len * box_idx + k;

                int val = enlight_get_tensor_qdata_by_off(obj_output, y_off, x_off, c_off);

                *obj_dst++ = val;
            }

            //cls confidence
            for(k = 0; k < num_class; k++) {
                int y_off = grid_idx / num_grids[2 * i + 1];
                int x_off = grid_idx % num_grids[2 * i + 1];
                int c_off = conf_box_len * box_idx + len_loc + len_obj + k;
                if (out_tensor_divided == 1)
                    c_off = conf_box_len * box_idx + len_obj + k;
                else if (out_tensor_divided == 2)
                    c_off = conf_box_len * box_idx + k;

                int8_t val = enlight_get_tensor_qdata_by_off(conf_output, y_off, x_off, c_off);

                *conf_dst++ = val;
            }
        }

    }

    //apply dfl projection
    if (dfl_reg_max) {
        float *coord_src = NULL;
        float *coord_dst = NULL;
        for(i = 0; i < num_output; i++) {
            // apply softmax
            if (!loc_temp[i].num_ele)
                continue;

            softmax_float(&loc_temp[i], &loc_smax[i], dfl_reg_max);

            // apply conv1x1
            coord_src = (float*)loc_smax[i].buf;
            coord_dst = (float*)loc[i].buf;

            for (j = 0; j < loc_smax[i].dims[1]; j++) {

                float sum = 0;
                for (k = 0; k < dfl_reg_max; k++)
                    sum += (coord_src[k] * k);

                *coord_dst++ = sum;
                coord_src += dfl_reg_max;
            }

#ifdef DUMP
            debug_dfl_outputs(i, loc_smax, loc);
#endif

        }
        free_tensor(loc_temp);
        enlight_free(buf_temp);
    }

    //enlight_detector_dbg_fout();
}

/**
 * @brief YOLO detector wrapper function for float precision.
 * 
 * @param softmax_use Softmax usage flag(for yolov2).
 * @param sigmoid_applied Sigmoid applied flag.
 * @param num_loc Number of location tensors.
 * @param num_conf Number of confidence tensors.
 * @param num_obj Number of object tensors.
 * @param loc_outputs Location output tensors.
 * @param conf_outputs Confidence output tensors.
 * @param obj_outputs Object output tensors.
 * @param def_box_addr Default box addresses.
 * @param exp_tbl_base Exponential lookup tables.
 * @param sig_tbl_base Sigmoid lookup tables.
 * @param num_class Number of classes.
 * @param num_grids Grid dimensions.
 * @param num_anchor Number of anchors per a grid.
 * @param img_sizes Input image dimensions(C, H, W).
 * @param th_conf Confidence threshold.
 * @param th_iou IOU threshold.
 * @param detected_objs Output detected objects.
 * @param yolo_version YOLO version.
 * @param max_detection Maximum detections.
 * @param nms_method NMS method type.
 * @param dfl_reg_max DFL regression maximum.
 * @return Status code (0 for success).
 */
static int detect_object_float(
    int             softmax_use,
    int             sigmoid_applied,
    int             num_loc,
    int             num_conf,
    int             num_obj,
    enlight_act_tensor_t**
                    loc_outputs,
    enlight_act_tensor_t**
                    conf_outputs,
    enlight_act_tensor_t**
                    obj_outputs,
    float*          def_box_addr,
    float*          exp_tbl_base,
    float*          sig_tbl_base,
    int             num_class,
    int*            num_grids,
    int             num_anchor,
    int*            img_sizes,
    float           th_conf,
    float           th_iou,
    enlight_objs_t* detected_objs,
    int             yolo_version,
    int             max_detection,
    enlight_nms_t   nms_method,
    int             dfl_reg_max
    )
{
    //enlight_detector_dbg_fin();

    int i;
#ifdef POST_PROCESS_REPORT_CYCLE
    volatile unsigned long cycle0, cycle1, cycle0_h, cycle1_h;
#endif //POST_PROCESS_REPORT_CYCLE

    tensor_t scr[MAX_NUM_OUTPUT];
    tensor_t loc[MAX_NUM_OUTPUT];
    tensor_t pri[MAX_NUM_OUTPUT];
    tensor_t obj[MAX_NUM_OUTPUT];

    int cand_size = 0;
    int cand_nums[MAX_NUM_OUTPUT];
    int* cand_bufs_addr[MAX_NUM_OUTPUT];
    int* cand_buf;
    int max_cand_num = 0;

    float* loc_sig_tbls_addr[MAX_NUM_OUTPUT];
    float* conf_sig_tbls_addr[MAX_NUM_OUTPUT];
    float* obj_sig_tbls_addr[MAX_NUM_OUTPUT];

    float* loc_exp_tbls_addr[MAX_NUM_OUTPUT];
    float* conf_exp_tbls_addr[MAX_NUM_OUTPUT];
    float* obj_exp_tbls_addr[MAX_NUM_OUTPUT];

    float* def_boxs_addr[MAX_NUM_OUTPUT];

    int src_loc_len = 4;
    int src_pri_len = 4;
    int src_obj_len = 1;

    int out_tensor_divided;
    int num_output = num_loc;

#ifdef POST_PROCESS_REPORT_CYCLE
#ifdef __i386__
    asm volatile ("rdtsc" : "=a"(cycle0), "=d"(cycle0_h));
#else
    asm volatile ("rdcycle %0" : "=r" (cycle0));
#endif
#endif //POST_PROCESS_REPORT_CYCLE

    if (yolo_version == 6 || yolo_version == 8)
        src_obj_len = 0;

    if (yolo_version == 6 || yolo_version == 8 || yolo_version == 10)
        src_pri_len = 2;

    if (dfl_reg_max)
        src_loc_len = 4 * dfl_reg_max;


    if (num_loc == num_conf) {
        // Tensors divided into loc, conf
        out_tensor_divided = 1;
        if (num_conf == num_obj) {
            // Tensors Divided into loc, conf, obj
           out_tensor_divided = 2;
        }
    }
    else { //Not divided
        out_tensor_divided = 0;
    }

    if (!src_obj_len && out_tensor_divided == 2) {
        enlight_network_err("Objectness tensor is given but,"
                             "size of objectness is 0\n");
        return 1;
    }

#ifdef GDB_FW_DEBUG
        gdb_fw_objects.cnt = 0;
#endif

    for(i = 0; i < num_output; i++)
        max_cand_num += (num_grids[2 * i] * num_grids[2 * i + 1]);
    max_cand_num *= num_anchor * sizeof(int);

    cand_buf = (int*)enlight_malloc(max_cand_num);

    cand_size = 0;
    for(i = 0; i < num_output; i++) {
        cand_bufs_addr[i] = &cand_buf[cand_size];
        cand_size += (num_grids[i * 2] * num_grids[(i * 2) + 1]) * num_anchor;
    }

    cand_size = 0;
    for(i = 0; i < num_output; i++) {
        def_boxs_addr[i] = &def_box_addr[cand_size];
        cand_size += num_grids[i * 2] * num_grids[(i * 2) + 1] * num_anchor * src_pri_len;
    }

    for(i = 0; i < num_output; i++) {
        conf_sig_tbls_addr[i] = &sig_tbl_base[i * 256 + 128];
        loc_sig_tbls_addr[i] = &sig_tbl_base[i * 256 + 128];
        obj_sig_tbls_addr[i] = &sig_tbl_base[i * 256 + 128];
        conf_exp_tbls_addr[i] = &exp_tbl_base[i * 256 + 128];
        loc_exp_tbls_addr[i] = &exp_tbl_base[i * 256 + 128];
        obj_exp_tbls_addr[i] = &sig_tbl_base[i * 256 + 128];

        if (out_tensor_divided == 1) {
            conf_sig_tbls_addr[i] = &sig_tbl_base[(2*i+0) * 256 + 128];
            obj_sig_tbls_addr[i] = &sig_tbl_base[(2*i+0) * 256 + 128];
            loc_sig_tbls_addr[i] = &sig_tbl_base[(2*i+1) * 256 + 128];
            conf_exp_tbls_addr[i] = &exp_tbl_base[(2*i+0) * 256 + 128];
            obj_exp_tbls_addr[i] = &exp_tbl_base[(2*i+0) * 256 + 128];
            loc_exp_tbls_addr[i] = &exp_tbl_base[(2*i+1) * 256 + 128];
        }
        else if (out_tensor_divided == 2) {
            conf_sig_tbls_addr[i] = &sig_tbl_base[(3*i+0) * 256 + 128];
            loc_sig_tbls_addr[i] = &sig_tbl_base[(3*i+1) * 256 + 128];
            obj_sig_tbls_addr[i] = &sig_tbl_base[(3*i+2) * 256 + 128];
            conf_exp_tbls_addr[i] = &exp_tbl_base[(3*i+0) * 256 + 128];
            loc_exp_tbls_addr[i] = &exp_tbl_base[(3*i+1) * 256 + 128];
            obj_exp_tbls_addr[i] = &exp_tbl_base[(3*i+2) * 256 + 128];
        }
    }

    get_candidate(
        loc_outputs,
        conf_outputs,
        obj_outputs,
        cand_bufs_addr,
        cand_nums,
        obj_sig_tbls_addr,
        num_output, num_grids,
        num_anchor, num_class,
        th_conf,
        out_tensor_divided,
        sigmoid_applied,
        src_obj_len,
        src_loc_len);


    void* output_work_buf_base[4];

    for(i = 0 ; i < num_output; i++) {
        int cand_num = cand_nums[i];

        enlight_detector_dbg("cand_num : %d\n", cand_num);

        int dst_obj_len = sizeof(int8_t) * cand_num * src_obj_len;
        dst_obj_len = (dst_obj_len + (WORD_SIZE - 1)) & ~(WORD_SIZE - 1);
        int dst_pri_len = sizeof(float) * cand_num * src_pri_len;
        int dst_loc_len = sizeof(int8_t) * cand_num * 4;
        if (dfl_reg_max)
            dst_loc_len = sizeof(float) * cand_num * 4;
        int dst_con_len = sizeof(int8_t) * cand_num * num_class;

        output_work_buf_base[i] = enlight_malloc(dst_obj_len +
                                                 dst_pri_len +
                                                 dst_loc_len +
                                                 dst_con_len);

        uint8_t* obj_buf = output_work_buf_base[i];
        uint8_t* pri_buf = obj_buf + dst_obj_len;
        uint8_t* loc_buf = pri_buf + dst_pri_len;
        uint8_t* con_buf = loc_buf + dst_loc_len;

        obj[i].buf      = obj_buf;
        obj[i].dtype    = ENLIGHT_DTYPE_INT8;
        obj[i].num_ele  = 1 * cand_num;
        obj[i].num_dim  = 3;
        obj[i].dims[0]  = 1;
        obj[i].dims[1]  = cand_num;
        obj[i].dims[2]  = src_obj_len;
        obj[i].exp_tbl  = obj_exp_tbls_addr[i];
        obj[i].sig_tbl  = obj_sig_tbls_addr[i];

        pri[i].buf      = pri_buf;
        pri[i].dtype    = ENLIGHT_DTYPE_FP32;
        pri[i].num_ele  = cand_num * src_pri_len;
        pri[i].num_dim  = 2;
        pri[i].dims[0]  = cand_num;
        pri[i].dims[1]  = src_pri_len;
        pri[i].exp_tbl  = NULL;
        pri[i].sig_tbl  = NULL;

        loc[i].buf      = loc_buf;
        loc[i].dtype    = ENLIGHT_DTYPE_INT8;
        if (dfl_reg_max)
            loc[i].dtype= ENLIGHT_DTYPE_FP32;
        loc[i].num_ele  = cand_num * 4;
        loc[i].num_dim  = 3;
        loc[i].dims[0]  = 1;
        loc[i].dims[1]  = cand_num;
        loc[i].dims[2]  = 4;
        loc[i].exp_tbl  = loc_exp_tbls_addr[i];
        loc[i].sig_tbl  = loc_sig_tbls_addr[i];
        loc[i].scl      = enlight_get_tensor_scale(loc_outputs[i]);

        scr[i].buf      = con_buf;
        scr[i].dtype    = ENLIGHT_DTYPE_INT8;
        scr[i].num_ele  = cand_num * num_class;
        scr[i].num_dim  = 3;
        scr[i].dims[0]  = 1;
        scr[i].dims[1]  = cand_num;
        scr[i].dims[2]  = num_class;
        scr[i].exp_tbl  = conf_exp_tbls_addr[i];
        scr[i].sig_tbl  = conf_sig_tbls_addr[i];
    }

    decompose_output(loc_outputs, conf_outputs, obj_outputs,
                     def_boxs_addr,
                     num_anchor, num_class, num_output, num_grids,
                     cand_bufs_addr, cand_nums, out_tensor_divided,
                     loc, scr, pri, obj,
                     dfl_reg_max);

    enlight_free(cand_buf);


    int dfl_applied = dfl_reg_max ? 1 : 0;
    detection_f_t* r1 = yolo_post_process(scr, loc, pri, obj,
                                          th_conf, th_iou, num_grids,
                                          num_output, softmax_use,
                                          sigmoid_applied,
                                          yolo_version, nms_method,
                                          dfl_applied);

#ifdef POST_PROCESS_REPORT_CYCLE
#ifdef __i386__
    asm volatile ("rdtsc" : "=a"(cycle1), "=d"(cycle1_h));
#else
    asm volatile ("rdcycle %0" : "=r" (cycle1));
#endif
    enlight_detector_log("[Postproc cycles] %ld: %ld ~ %ld \n", cycle1 - cycle0, cycle0, cycle1);
#endif //POST_PROCESS_REPORT_CYCLE

    enlight_detector_dbg("%d object is detected\n", r1->n);


#ifdef DUMP
    debug_detection_results(r1, nms_method);
#endif

    if (max_detection > 0)
        r1->n = max_detection;

    if (r1->n > MAX_DETECT_BOX) {
        
        r1->n = MAX_DETECT_BOX;
    }

    detected_objs->cnt = r1->n;

    for (i = 0; i < r1->n; i++) {
        box_f_t* b = r1->box + i;

        

#ifdef GDB_FW_DEBUG
        enlight_obj_t *dbg_obj = &gdb_fw_objects.obj[i];
        dbg_obj->x_min = img_sizes[2]*b->x_min;
        dbg_obj->x_max = img_sizes[2]*b->x_max;
        dbg_obj->y_min = img_sizes[1]*b->y_min;
        dbg_obj->y_max = img_sizes[1]*b->y_max;
        dbg_obj->score = r1->score[i];
        dbg_obj->cls = r1->cls[i] + 1; // + 1 for background
        dbg_obj->img_w = img_sizes[2];
        dbg_obj->img_h = img_sizes[1];
        gdb_fw_objects.cnt = r1->n;
#endif
        enlight_obj_t *ret_obj = &detected_objs->obj[i];
        ret_obj->x_min = img_sizes[2]*b->x_min;
        ret_obj->x_max = img_sizes[2]*b->x_max;
        ret_obj->y_min = img_sizes[1]*b->y_min;
        ret_obj->y_max = img_sizes[1]*b->y_max;
        ret_obj->score = r1->score[i];
        ret_obj->cls = r1->cls[i] + 1; // + 1 for background
        ret_obj->img_w = img_sizes[2];
        ret_obj->img_h = img_sizes[1];
    }

    enlight_free(r1);

    for(i = 0 ; i < num_output; i++)
        enlight_free(output_work_buf_base[i]);

    //enlight_detector_dbg_fout();

    return 0;
}

/**
 * @brief Initialize YOLO detector.
 */
void yolo_init()
{
    //enlight_detector_dbg_fin();

#ifdef BARE_METAL_FW_DEV
    extern uint8_t* detector_work_buf;
    enlight_init_mem(detector_work_buf);
#endif

    //enlight_detector_dbg_fout();
}

/**
 * @brief YOLO detector wrapper function for object detection.
 *
 * @param softmax_use Softmax usage flag(for yolov2).
 * @param sigmoid_applied Sigmoid applied flag.
 * @param num_loc Number of location tensors.
 * @param num_conf Number of confidence tensors.
 * @param num_obj Number of object tensors.
 * @param loc_outputs Location output tensors.
 * @param conf_outputs Confidence output tensors.
 * @param obj_outputs Object output tensors.
 * @param def_box_addr Default box addresses.
 * @param exp_tbl_base Exponential lookup tables.
 * @param sig_tbl_base Sigmoid lookup tables.
 * @param num_class Number of classes.
 * @param num_grids Grid dimensions.
 * @param num_anchor Number of anchors per a grid.
 * @param img_sizes Input image dimensions(C, H, W).
 * @param th_conf Confidence threshold.
 * @param th_iou IOU threshold.
 * @param detected_objs Output detected objects.
 * @param yolo_version YOLO version.
 * @param max_detection Maximum detections.
 * @param nms_method NMS method type.
 * @param dfl_reg_max DFL regression maximum.
 * @return Status code (0 for success).
 */
int yolo_detect_object(
    int             softmax_use,
    int             sigmoid_applied,
    int             num_loc,
    int             num_conf,
    int             num_obj,
    enlight_act_tensor_t**
                    loc_outputs,
    enlight_act_tensor_t**
                    conf_outputs,
    enlight_act_tensor_t**
                    obj_outputs,
    float*          def_box_addr,
    float*          exp_tbl_base,
    float*          sig_tbl_base,
    int             num_class,
    int*            num_grids,
    int             num_anchor,
    int*            img_sizes,
    float           th_conf,
    float           th_iou,
    enlight_objs_t* detected_objs,
    int             yolo_version,
    int             max_detection,
    enlight_nms_t   nms_method,
    int             dfl_reg_max
    )
{
    //enlight_detector_dbg_fin();

    int res = detect_object_float(
        softmax_use,
        sigmoid_applied,
        num_loc,
        num_conf,
        num_obj,
        loc_outputs,
        conf_outputs,
        obj_outputs,
        def_box_addr,
        exp_tbl_base,
        sig_tbl_base,
        num_class,
        num_grids,
        num_anchor,
        img_sizes,
        th_conf,
        th_iou,
        detected_objs,
        yolo_version,
        max_detection,
        nms_method,
        dfl_reg_max
    );

    //enlight_detector_dbg_fout();

    return res;
}
