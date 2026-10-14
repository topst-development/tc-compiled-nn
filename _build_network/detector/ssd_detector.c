#include "enlight_data_type.h"
#include "enlight_network.h"
#include "detector.h"

#ifdef DUMP
#   include <stdio.h>
#endif

#ifdef GDB_FW_DEBUG
enlight_objs_t gdb_fw_objects;
#endif

#ifdef DUMP
/**
 * @brief Debug function to dump regression result data.
 * 
 * @param num_cand_box Number of candidate boxes.
 * @param num_class Number of classes.
 * @param cand_conf Candidate confidence tensor.
 * @param bg_in_cls Background in class flag.
 * @param th_conf Confidence threshold.
 */
static void debug_regression_result(int num_cand_box, int num_class,
                                    tensor_t* cand_conf, int bg_in_cls,
                                    float th_conf)
{
    char dump_name[256];
    sprintf(dump_name, "regression_result.bin");

    int i, j;
    float val;

    int num_e = num_cand_box * num_class;
    float *cand_conf_deb = (float*)cand_conf->buf;
    float *d_arr = (float*)enlight_malloc(sizeof(float) * num_e);
    for (i = 0; i < num_cand_box; i++) {
        for (j = 0; j < num_class; j++) {
            val = cand_conf_deb[i*num_class + j];

            if (j == 0 && bg_in_cls) 
                d_arr[i*num_class] = 0;

            if (val < th_conf)
                d_arr[i*num_class + j] = 0;
            else {
                d_arr[i*num_class + j] = val;
            }
        }
    }
    enlight_dump_data_to_bin(dump_name, d_arr, 4, num_e);
}

/**
 * @brief Debug function to dump NMS input data for specific class.
 * 
 * @param cand Candidate structure.
 * @param cls Class index.
 */
static void debug_nms_input(cand_f_t* cand, int cls)
{
    int i;
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
        sprintf(dump_name, "cand_loc_%d.bin", cls);
        enlight_dump_data_to_bin(dump_name, dump_box, 4, num_cand * 4);
        enlight_free(dump_box);
    }
}

/**
 * @brief Debug function to dump FastNMS input data.
 * 
 * @param cand Candidate structure.
 */
static void debug_fastnms_inputs(cand_f_t* cand)
{
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

}

/**
 * @brief Debug function to dump detection results.
 * 
 * @param r1 Detection result structure.
 * @param nms_method NMS method type.
 */
static void debug_detection_results(detection_f_t* r1, enlight_nms_t nms_method)
{
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

}

#endif

/**
 * @brief Build bounding box information from location and prior tensors.
 * 
 * @param box Output bounding box structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param idx Index of the box to process.
 */
static void build_box_info(box_f_t* box, tensor_t* loc, tensor_t* prior, int idx)
{
    //enlight_detector_dbg_fin();

    int box_idx = idx << 2;

    int8_t xywh[4];
    float* _prior = (float*)prior->buf + box_idx;

    uint32_t* table = (uint32_t*)loc->exp_tbl;

    float x, y, w, h;
    float anchor_x, anchor_y, anchor_w, anchor_h;

    int y_off = idx;
    int x_off = 0;
    int c_off = 0;

    xywh[0] = enlight_get_tensor_qdata_by_off(loc->output_tensor, y_off, x_off, c_off + 0);
    xywh[1] = enlight_get_tensor_qdata_by_off(loc->output_tensor, y_off, x_off, c_off + 1);
    xywh[2] = enlight_get_tensor_qdata_by_off(loc->output_tensor, y_off, x_off, c_off + 2);
    xywh[3] = enlight_get_tensor_qdata_by_off(loc->output_tensor, y_off, x_off, c_off + 3);

    //enlight_detector_dbg("%d: xywh[0]: %d, xywh[1]: %d, xywh[2]: %d, xywh[3]: %d\n",
    //                     idx, xywh[0], xywh[1], xywh[2], xywh[3]);

    x = xywh[0] / loc->scl;
    y = xywh[1] / loc->scl;
    w = *((float*)&table[xywh[2]]);
    h = *((float*)&table[xywh[3]]);

    //enlight_detector_dbg("%d: x: %f, y: %f, w: %f, h: %f\n",
    //                     idx, x, y, w, h);

    anchor_x = _prior[0];
    anchor_y = _prior[1];
    anchor_w = _prior[2];
    anchor_h = _prior[3];

    x = x * anchor_w + anchor_x;
    y = y * anchor_h + anchor_y;
    w = w * anchor_w;
    h = h * anchor_h;

    box->x_min = x - w/2;
    box->x_max = x + w/2;
    box->y_min = y - h/2;
    box->y_max = y + h/2;
    box->area  = w * h;

    //enlight_detector_dbg_fout();
}

/**
 * @brief Apply thresholding to select best boxes by default method.
 * 
 * @param scr Score tensor data.
 * @param cand Output candidate structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param cand_box Candidate box indices.
 * @param num_cand_box Number of candidate boxes.
 * @param th Confidence threshold.
 * @param has_bg Background class flag.
 */
static void thresholding_by_default_box(tensor_t* scr, cand_f_t* cand,
                                        tensor_t* loc, tensor_t* prior,
                                        int *cand_box, int num_cand_box,
                                        float th, int has_bg)
{
    float* _scr = (float*)scr->buf + has_bg;

    int num_class = scr->dims[2];

    int box, n, cls;
    int candidate_box_idx;

    for (box = 0, n = 0 ; box < num_cand_box; box++, _scr += num_class) {
        int max_idx = 0;
        float max_val = _scr[0];
        for (cls = 1; cls < num_class - has_bg; cls++){
            if (max_val < _scr[cls]) {
                max_val = _scr[cls];
                max_idx = cls;
            }
        }

        candidate_box_idx = cand_box[box];

        if (max_val >= th) {
            build_box_info(&cand->box[n], loc, prior, candidate_box_idx);
            cand->cls[n] = max_idx;
            cand->score[n] = max_val;
            n++;
        }
    }
    cand->n = n;
}

/**
 * @brief Apply thresholding with specific candidate class.
 * 
 * @param score Score tensor data.
 * @param cand Output candidate structure.
 * @param loc Location tensor data.
 * @param prior Prior anchor box data.
 * @param cand_box Candidate box indices.
 * @param num_cand_box Number of candidate boxes.
 * @param cls Target class index.
 * @param threshold Confidence threshold.
 */
static void thresholding_with_candidate(
    tensor_t* score,
    cand_f_t* cand,
    tensor_t* loc,
    tensor_t* prior,
    int *cand_box,
    int num_cand_box,
    int cls,
    float threshold)
{
    //enlight_detector_dbg_fin();

    float* score_buf_base = (float*)score->buf;
    float* score_buf = &score_buf_base[cls];

    int num_class = score->dims[2];

    int i, box_idx;
    int n = 0;

    for (i = 0; i < num_cand_box; i++) {
        box_idx = cand_box[i];

        if (score_buf[num_class * i] >= threshold) {
            build_box_info(&cand->box[n], loc, prior, box_idx);
            cand->cls[n] = cls;
            cand->score[n] = score_buf[num_class * i];
            n++;
        }
    }
    cand->n = n;

    //enlight_detector_dbg_fout();
}

/**
 * @brief SSD post-processing pipeline for object detection.
 * 
 * @param score Score tensor data.
 * @param loc Location tensor data.
 * @param pri Prior anchor box data.
 * @param th_conf Confidence threshold.
 * @param th_nms NMS threshold.
 * @param word_size Word alignment size.
 * @param bg_in_cls Background in class flag.
 * @param logistic Logistic function type.
 * @param nms_method NMS method type.
 * @return Pointer to detection results.
 */
detection_f_t* ssd_post_process(tensor_t* score, tensor_t* loc, tensor_t* pri,
                              float th_conf, float th_nms, const int word_size,
                              int bg_in_cls, enlight_logistic_t logistic,
                              enlight_nms_t nms_method)
{
    //enlight_detector_dbg_fin();

    detection_f_t* detection = alloc_detection_float(score->num_ele);

    // batches, boxes, clses
    int num_box = score->dims[1];
    int num_class = score->dims[2];

    int conf_stride = (num_class + (word_size - 1)) & ~ (word_size - 1);

    int cls;

    int num_cand_box;
    int *cand_box = (int*)enlight_malloc(4 * num_box);

    tensor_t* cand_conf;
    int dst_dims[3];
    int i;

    int max_nms_cand = num_box;

    for (i = 0; i < score->num_dim; i++)
        dst_dims[i] = score->dims[i];
    dst_dims[1] = max_nms_cand;

    cand_conf = alloc_tensor(ENLIGHT_DTYPE_FP32, score->num_dim, dst_dims, 0);

    if (logistic == LOGISTIC_NONE) {
        // sigmoid was applied already when inferencing network
        num_cand_box = sigmoid_with_threshold_float(score, cand_conf, cand_box,
                                              conf_stride, th_conf, bg_in_cls, max_nms_cand, 1);
    }
    else if (logistic == LOGISTIC_SIGMOID) {
        num_cand_box = sigmoid_with_threshold_float(score, cand_conf, cand_box,
                                              conf_stride, th_conf, bg_in_cls, max_nms_cand, 0);
    }
    else {
        num_cand_box = softmax_with_threshold_float(score, cand_conf, cand_box,
                               conf_stride, th_conf, bg_in_cls, max_nms_cand);
    }

#ifdef DUMP
    debug_regression_result(num_cand_box, num_class, cand_conf, bg_in_cls, th_conf);
#endif

    if (nms_method == NMS_DEFAULT) {
        for (cls = bg_in_cls; cls < num_class; cls++)
        {
            int num_box = cand_conf->dims[1];
            cand_f_t* cand = (cand_f_t*)alloc_candidate_float(num_box);

            thresholding_with_candidate(cand_conf, cand, loc, pri, cand_box,
                                        num_cand_box, cls, th_conf);

#ifdef DUMP
            debug_nms_input(cand, cls);
#endif

            nms_float(detection, cand, cls, th_nms);
            free_candidate_float(cand);
        }
    }
    else {
        cand_f_t* cand = (cand_f_t*)alloc_candidate_float(num_box);

        thresholding_by_default_box(cand_conf, cand, loc, pri,
                                    cand_box, num_cand_box,
                                    th_conf, bg_in_cls);

#ifdef DUMP
        debug_fastnms_inputs(cand);
#endif

        fastnms_float(detection, cand, th_nms);
        free_candidate_float(cand);
    }

    enlight_free(cand_box);
    free_tensor(cand_conf);

    //enlight_detector_dbg_fout();
    return detection;
}

/**
 * @brief SSD detector wrapper function for float precision.
 * 
 * @param num_loc Output loc tensor number.
 * @param num_conf Output conf tensor number.
 * @param loc_tensors Loc output tensors.
 * @param conf_tensors Conf output tensors.
 * @param default_box_addrs Default box base addresses.
 * @param loc_exp_tables Loc output exp(x) tables.
 * @param con_logistic_tables Conf output logistic tables.
 * @param num_class Number of classes.
 * @param num_box Number of boxes.
 * @param img_sizes Input image H, W.
 * @param th_conf Confidence threshold.
 * @param th_iou IOU threshold.
 * @param bg_in_cls Background in class flag.
 * @param logistic Logistic type.
 * @param nms_method NMS method type.
 * @param detected_objs Return objects buffer base.
 * @param max_detection Max detection results.
 * @return Status code (0 for success).
 */
static int detect_object_float(
    int                    num_loc,
    int                    num_conf,
    enlight_act_tensor_t** loc_tensors,
    enlight_act_tensor_t** conf_tensors,
    float*                 default_box_addrs,
    float*                 loc_exp_tables,
    float*                 con_logistic_tables,
    int                    num_class,
    int                    num_box,
    int*                   img_sizes,
    float                  th_conf,
    float                  th_iou,
    int                    bg_in_cls,
    enlight_logistic_t     logistic,
    enlight_nms_t          nms_method,
    enlight_objs_t*        detected_objs,
    int                    max_detection
)
{
    //enlight_detector_dbg_fin();

    const int word_size = 32;
    int i;

#ifdef POST_PROCESS_REPORT_CYCLE
    volatile unsigned long cycle0, cycle1, cycle0_h, cycle1_h;
#endif //POST_PROCESS_REPORT_CYCLE

    tensor_t scr, loc, pri;

    pri.output_tensor = NULL;
    pri.buf      = default_box_addrs;
    pri.dtype    = ENLIGHT_DTYPE_UINT32;
    pri.num_ele  = num_box*4;
    pri.num_dim  = 2;
    pri.dims[0]  = num_box;
    pri.dims[1]  = 4;
    pri.exp_tbl  = NULL;

    loc.output_tensor = (void*)loc_tensors[0];
    loc.buf     = NULL;
    loc.scl     = enlight_get_tensor_scale(loc_tensors[0]);
    loc.dtype   = ENLIGHT_DTYPE_INT8;
    loc.num_ele = 1*num_box*4;
    loc.num_dim = 3;
    loc.dims[0] = 1;
    loc.dims[1] = num_box;
    loc.dims[2] = 4;
    loc.exp_tbl = &loc_exp_tables[128];

    scr.output_tensor = (void*)conf_tensors[0];
    scr.buf     = NULL;
    scr.dtype   = ENLIGHT_DTYPE_INT8;
    scr.num_ele = 1*num_box*num_class;
    scr.num_dim = 3;
    scr.dims[0] = 1;
    scr.dims[1] = num_box;
    scr.dims[2] = num_class + bg_in_cls;

    switch (logistic) {
        case LOGISTIC_NONE:
            break;
        case LOGISTIC_SIGMOID:
            scr.sig_tbl = &con_logistic_tables[128];
            break;
        case LOGISTIC_SOFTMAX:
            scr.exp_tbl = &con_logistic_tables[128];
            break;
        default:
            enlight_detector_err("Unknown logistic function\n");
            break;
    }

#ifdef POST_PROCESS_REPORT_CYCLE
#ifdef __i386__
    asm volatile ("rdtsc" : "=a"(cycle0), "=d"(cycle0_h));
#else
    asm volatile ("rdcycle %0" : "=r" (cycle0));
#endif
#endif //POST_PROCESS_REPORT_CYCLE

    detection_f_t* r1 = ssd_post_process(&scr, &loc, &pri, th_conf, th_iou, word_size,
                                       bg_in_cls, logistic, nms_method);

#ifdef POST_PROCESS_REPORT_CYCLE
#ifdef __i386__
    asm volatile ("rdtsc" : "=a"(cycle1), "=d"(cycle1_h));
#else
    asm volatile ("rdcycle %0" : "=r" (cycle1));
#endif
    enlight_detector_log("[Postproc cycles] %ld \n", cycle1 - cycle0);
#endif //POST_PROCESS_REPORT_CYCLE

    enlight_detector_dbg("%d object is detected\n", r1->n);

    detected_objs->cnt = r1->n;

    if (max_detection > 0)
        r1->n = max_detection;

    if (r1->n > MAX_DETECT_BOX) {
        enlight_detector_log("detection result %d is limited to MAX VALUE: %d\n",
                             r1->n, MAX_DETECT_BOX); 
        r1->n = MAX_DETECT_BOX;
    }

    detected_objs->cnt = r1->n;

#ifdef DUMP
    debug_detection_results(r1, nms_method);
#endif

    for (i = 0; i < r1->n ; i++) {
        box_f_t* b = r1->box + i;

        enlight_detector_dbg("x_min: %.4f, x_max: %.4f, y_min: %.4f, y_max: %.4f, score: %.4f, cls: %d\n",
                img_sizes[2]*b->x_min,
                img_sizes[2]*b->x_max,
                img_sizes[1]*b->y_min,
                img_sizes[1]*b->y_max,
                r1->score[i],
                r1->cls[i]);

#ifdef GDB_FW_DEBUG
        enlight_obj_t *dbg_obj = &gdb_fw_objects.obj[i];
        dbg_obj->x_min = img_sizes[2]*b->x_min;
        dbg_obj->x_max = img_sizes[2]*b->x_max;
        dbg_obj->y_min = img_sizes[1]*b->y_min;
        dbg_obj->y_max = img_sizes[1]*b->y_max;
        dbg_obj->score = r1->score[i];
        dbg_obj->cls   = r1->cls[i];
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
        ret_obj->cls   = r1->cls[i];
        ret_obj->img_w = img_sizes[2];
        ret_obj->img_h = img_sizes[1];
    }

    enlight_free(r1);

    //enlight_detector_dbg_fout();
    return 0;
}

/**
 * @brief Initialize SSD detector.
 */
void ssd_init()
{
#ifdef BARE_METAL_FW_DEV
    extern uint8_t* detector_work_buf;
    enlight_init_mem(detector_work_buf);
#endif
}

/**
 * @brief SSD detector wrapper function for object detection.
 *
 * @param num_loc Output loc tensor number.
 * @param num_conf Output conf tensor number.
 * @param loc_tensors Loc output tensors.
 * @param conf_tensors Conf output tensors.
 * @param default_box_addrs Default box base addresses.
 * @param loc_exp_tables Loc output exp(x) tables.
 * @param con_logistic_tables Conf output logistic tables.
 * @param num_class Number of classes.
 * @param num_box Number of boxes.
 * @param img_sizes Input image H, W.
 * @param th_conf Confidence threshold.
 * @param th_iou IOU threshold.
 * @param bg_in_cls Background in class flag.
 * @param logistic Logistic type.
 * @param nms_method NMS method type.
 * @param detected_objs Return objects buffer base.
 * @param max_detection Max detection results.
 * @return Status code (0 for success).
 */
int ssd_detect_object(
    int                    num_loc,
    int                    num_conf,
    enlight_act_tensor_t** loc_tensors,
    enlight_act_tensor_t** conf_tensors,
    float*                 default_box_addrs,
    float*                 loc_exp_tables,
    float*                 con_logistic_tables,
    int                    num_class,
    int                    num_box,
    int*                   img_sizes,
    float                  th_conf,
    float                  th_iou,
    int                    bg_in_cls,
    enlight_logistic_t     logistic,
    enlight_nms_t          nms_method,
    enlight_objs_t*        detected_objs,
    int                    max_detection
)
{

    int res = detect_object_float(
        num_loc,
        num_conf,
        loc_tensors,
        conf_tensors,
        default_box_addrs,
        loc_exp_tables,
        con_logistic_tables,
        num_class,
        num_box,
        img_sizes,
        th_conf,
        th_iou,
        bg_in_cls,
        logistic,
        nms_method,
        detected_objs,
        max_detection);

    return res;
}
