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

/**
 * @brief Allocate memory of the input size. 
 * 
 * @param size The size to allocate memory.
 * @return void* The address that allocated.
 */
void* enlight_malloc(int size);

/**
 * @brief Free memory of the input address.
 * 
 * @param buf The address to free.
 */
void enlight_free(void* buf);

/**
 * @brief Dump of the input buffer. The buffer size to dump is [size_of_dtype * num_ele]. 
 * 
 * @param fn The file name to save dump result.
 * @param buf The address to dump.
 * @param size_of_dtype Size of each element(data type) to dump(ex. int 4, float 4).
 * @param num_ele Number of elements to dump.
 */
void enlight_dump_data_to_bin(char *fn, void *buf, int size_of_dtype, int num_ele);

/**
 * @brief Initialize the tensor.
 * 
 * @param tensor Tensor structure to initialize.
 * @param dtype Data type of the tensor.
 * @param num_dim Number of the tensor dimensions.
 * @param dims List of the tensor dimensions.
 * @param scl Scale value for quantized tensor.
 * @param exp_tbl Exponetial lookup table.
 * @param sig_tbl Sigmoid lookup table.
 * @param buf Data buffer of the tensor.
 */
void init_tensor(tensor_t* tensor, int dtype, int num_dim, int *dims, float scl,
                 void* exp_tbl, void* sig_tbl, void* buf);

/**
 * @brief Allocation the tensor.
 * 
 * @param dtype Data type for the tensor.
 * @param num_dim Number of the tensor dimensions.
 * @param dims List of the tensor dimensions.
 * @param scl Scale value for quantized tensor.
 * @return tensor_t* Address of the tensor that allocated.
 */
tensor_t* alloc_tensor(int dtype, int num_dim, int* dims, float scl);

/**
 * @brief Free the input tensor.
 * 
 * @param tensor Address of the tensor to free.
 */
void free_tensor(tensor_t* tensor);

/**
 * @brief Allocate list of the detection structure.
 * 
 * @param detect_num Number of the detection structure to allocate.
 * @return detection_t* List for the detection structure that allocated.
 */
detection_t* alloc_detection(int detect_num);

/**
 * @brief Allocate list of the float type detection structure.
 * 
 * @param detect_num Number of the detection structure to allocate.
 * @return detection_f_t* List for the float type detection structure that allocated.
 */
detection_f_t* alloc_detection_float(int detect_num);

/**
 * @brief Free the input detection list.
 * 
 * @param detection Detection structure to free.
 */
void free_detection(detection_t* detection);
void free_detection_float(detection_f_t* detection);

/**
 * @brief Copy the tensor information from source to destination.
 * 
 * @param dst Destination tensor address to copy information.
 * @param src Source tensor address to copy information.
 */
void copy_tensor_info(tensor_t* dst, tensor_t* src);

/**
 * @brief Apply softmax to input source tensor.
 * 
 * @param src Source tensor to apply the softmax function.
 * @param dst Destination tensor after softmax.
 * @param stride Stride between each tensor elements(Box).
 */
void softmax(tensor_t* src, tensor_t* dst, int stride);
void softmax_float(tensor_t* src, tensor_t* dst, int stride);

/**
 * @brief Apply softmax with threshold to input source tensor.
 * 
 * @param src Source tensor to apply the softmax function.
 * @param dst Destination tensor after softmax.
 * @param cand_box(cand_box_index_buf) Index list of candidate box.
 * @param stride Stride between each tensor elements(Box).
 * @param threshold Threshold to apply.
 * @param has_bg Including background in the class or not(0: not include, 1: include).
 * @param max_nms_cand Maximum number of candidate bounding box for NMS.
 * @return int Number of candidate bounding box after thresholding.
 */
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

/**
 * @brief Caculate IoU(Intersection over Union) between input bounding boxes.
 * 
 * @param l box to caculate IoU.
 * @param r box to caculate IoU.
 * @return float IoU that caculated.
 */
float iou(box_t* l, box_t* r);
float iou_float(box_f_t* l, box_f_t* r);

/**
 * @brief Apply NMS to detection structure.
 * 
 * @param detection Detection structure to apply NMS.
 * @param cand Candidate structure for NMS.
 * @param cls Class index to apply NMS.
 * @param threshold IoU threshold for NMS.
 */
void nms(detection_t* detection, cand_t* cand, int cls, float threshold);
void nms_float(detection_f_t* detection, cand_f_t* cand, int cls, float threshold);

/**
 * @brief Apply FastNMS to detection structure.
 * 
 * @param detection Detection structure to apply FastNMS.
 * @param cand Candidate structure for FastNMS.
 * @param th IoU threshold for NMS.
 */
void fastnms(detection_t* detection, cand_t* cand, float th);
void fastnms_float(detection_f_t* detection, cand_f_t* cand, float th);

/**
 * @brief Apply sigmoid to input source tensor(integer domain).
 * 
 * @param src Source tensor to apply the sigmoid function.
 * @param dst Destination tensor after sigmoid.
 * @param stride Stride between each tensor elements(Box).
 */
void sigmoid(tensor_t *src, tensor_t *dst, int stride);
void sigmoid_float(tensor_t *src, tensor_t *dst, int stride);

/**
 * @brief Apply sigmoid with threshold to input source tensor.
 * 
 * @param src Source tensor to apply the sigmoid function.
 * @param dst Destination tensor after sigmoid.
 * @param cand_box Index list of candidate box.
 * @param stride Stride between each tensor elements(Box).
 * @param threshold Threshold to apply.
 * @param has_bg Including background in the class or not(0: not include, 1: include).
 * @param max_nms_cand Maximum number of candidate bounding box for NMS.
 * @param sigmoid_applied Whether sigmoid is already applied(0: not applied, 1: applied).
 * @return int Number of candidate bounding box after thresholding.
 */
int sigmoid_with_threshold(tensor_t *src, tensor_t *dst, int *cand_box,
                           int stride, float threshold,
                           int has_bg, int max_nms_cand, int sigmoid_applied);
int sigmoid_with_threshold_float(tensor_t *src, tensor_t *dst, int *cand_box,
                                 int stride, float threshold,
                                 int has_bg, int max_nms_cand, int sigmoid_applied);

/**
 * @brief Apply intro_sort to input list.
 * 
 * @param unsorted Input list to sort.
 * @param indices Indices of input list.
 * @param N Number of input list elements.
 */
void intro_sort(const uint16_t* unsorted, int* indices, int N);
void intro_sort_float(const float* unsorted, int* indices, int N);

/**
 * @brief Internal recursive function for mergesort.
 *
 * @param unsorted Input list to sort.
 * @param left Left boundary indices.
 * @param right Right boundary indices.
 * @param buffer Temporary buffer for merge operations.
 */
void _mergesort(const uint16_t* unsorted, int *left, int *right, int *buffer);
void _mergesort_float(const float* unsorted, int *left, int *right, int *buffer);

/**
 * @brief Apply mergesort to input list.
 * @param unsorted Input list to sort.
 * @param indices Indices of input list.
 * @param N Number of input list elements.
 */
void mergesort(const uint16_t* unsorted, int* indices, int N);
void mergesort_float(const float* unsorted, int* indices, int N);

#endif /*__ENLIGHT_DETECTOR_H__*/
