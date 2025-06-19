#ifndef __ENLIGHT_DETECTOR_H__
#define __ENLIGHT_DETECTOR_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef MIN
#   define MIN(A, B) (((A)>(B))?(B):(A))
#endif
#ifndef MAX
#   define MAX(A, B) (((A)>(B))?(A):(B))
#endif

#ifdef BARE_METAL_FW_DEV
    extern int _printf(const char *format, ...);
#   define  ENLIGHT_DETECT_PRINT  _printf
#else
#   define  ENLIGHT_DETECT_PRINT   printf
#endif

#define enlight_detector_err(...)  do {ENLIGHT_DETECT_PRINT(__VA_ARGS__); } while(0)
#define enlight_detector_log(...)  do {ENLIGHT_DETECT_PRINT(__VA_ARGS__); } while(0)

#ifdef ENLIGHT_DETECTOR_DEBUG
#   define enlight_detector_dbg(...)  do {ENLIGHT_DETECT_PRINT(__VA_ARGS__); } while(0)
#else
#   define enlight_detector_dbg(...)  do {} while(0)
#endif

#define enlight_detector_dbg_fin()  do {\
                                            enlight_detector_dbg("\n>>>>");\
                                            enlight_detector_dbg(__FUNCTION__);\
                                            enlight_detector_dbg("\n");\
                                       } while(0)

#define enlight_detector_dbg_fout()  do {\
                                            enlight_detector_dbg("\n<<<<");\
                                            enlight_detector_dbg(__FUNCTION__);\
                                            enlight_detector_dbg("\n");\
                                       } while(0)
    

#ifndef NULL
#   define  NULL    0
#endif

#define MAX_STACK_SIZE          64    //Max recursion to Max_CANDIDATE_BOX 2*log2(16384) * element(2)
#define INSERTION_SORT_SIZE     15
#define SMALL_MERGESORT_SIZE   20

#define alloc_candidate(max_candidate) alloc_detection(max_candidate)
#define alloc_candidate_float(max_candidate) alloc_detection_float(max_candidate)

#define free_candidate(candidate) free_detection(candidate)
#define free_candidate_float(candidate) free_detection_float(candidate)

#define ENLIGHT_NPU_SIG_OUT_SCL (64)

#define WORD_SIZE               4

typedef struct {
    enlight_act_tensor_t* output_tensor;
    void* buf;

    float scl;

    int dtype;
    int num_ele;
    int num_dim;

    int dims[4];

    float* sig_tbl;
    float* exp_tbl;
} tensor_t;

typedef struct {
    int16_t x_min;
    int16_t x_max;
    int16_t y_min;
    int16_t y_max;
    uint32_t area;
} box_t;

typedef struct {
    box_t* box;

    uint16_t* cls;
    uint16_t* score;

    int n;
} detection_t;

typedef struct
{
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float area;
} box_f_t;

typedef struct
{
    box_f_t* box;

    uint16_t* cls;
    float* score;

    int n;
} detection_f_t;

typedef detection_t cand_t;
typedef detection_f_t cand_f_t;

#ifdef BARE_METAL_FW_DEV
void enlight_init_mem(void *buf);
#endif

void* enlight_malloc(int size);
void enlight_free(void* buf);
void enlight_dump_data_to_bin(char *fn, void *buf, int size_of_dtype, int num_ele);
void init_tensor(tensor_t* tensor, int dtype, int num_dim, int *dims, float scl,
                 void* exp_tbl, void* sig_tbl, void* buf);
tensor_t* alloc_tensor(int dtype, int num_dim, int* dims, float scl);
void free_tensor(tensor_t* tensor);

detection_t* alloc_detection(int detect_num);
detection_f_t* alloc_detection_float(int detect_num);

void free_detection(detection_t* detection);
void free_detection_float(detection_f_t* detection);

void copy_tensor_info(tensor_t* dst, tensor_t* src);

void softmax(tensor_t* src, tensor_t* dst, int stride);
void softmax_float(tensor_t* src, tensor_t* dst, int stride);

int softmax_with_threshold(tensor_t* src,
                           tensor_t* dst,
                           int *cand_box,
                           int stride,
                           float threshold,
                           int has_bg,
                           int max_nms_cand);

int softmax_with_threshold_float(tensor_t* src,
                                 tensor_t* dst,
                                 int *cand_box_index_buf,
                                 int stride,
                                 float threshold,
                                 int has_bg,
                                 int max_cand_box);

float iou(box_t* l, box_t* r);
float iou_float(box_f_t* l, box_f_t* r);

void nms(detection_t* detection, cand_t* cand, int cls, float threshold);
void nms_float(detection_f_t* detection, cand_f_t* cand, int cls, float threshold);

void fastnms(detection_t* detection, cand_t* cand, float th);
void fastnms_float(detection_f_t* detection, cand_f_t* cand, float th);

void sigmoid(tensor_t *src, tensor_t *dst, int stride);
void sigmoid_float(tensor_t *src, tensor_t *dst, int stride);

int sigmoid_with_threshold(tensor_t *src, tensor_t *dst, int *cand_box,
                           int stride, float threshold,
                           int has_bg, int max_nms_cand, int sigmoid_applied);

int sigmoid_with_threshold_float(tensor_t *src, tensor_t *dst, int *cand_box,
                                 int stride, float threshold,
                                 int has_bg, int max_nms_cand, int sigmoid_applied);

void intro_sort(const uint16_t* unsorted, int* indices, int N);
void intro_sort_float(const float* unsorted, int* indices, int N);

void _mergesort(const uint16_t* unsorted, int *left, int *right, int *buffer);
void _mergesort_float(const float* unsorted, int *left, int *right, int *buffer);

void mergesort(const uint16_t* unsorted, int* indices, int N);
void mergesort_float(const float* unsorted, int* indices, int N);

#endif /*__ENLIGHT_DETECTOR_H__*/
