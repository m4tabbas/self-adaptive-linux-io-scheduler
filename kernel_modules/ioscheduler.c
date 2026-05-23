/*
 * Copyright (c) 2019-2021 Ibrahim Umit Akgun
 * Copyright (c) 2019-2021 Erez Zadok
 * Copyright (c) 2019-2021 Stony Brook University
 * Copyright (c) 2019-2021 The Research Foundation of SUNY
 *
 * You can redistribute it and/or modify it under the terms of the Apache
 * License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
 */

#include <linux/delay.h>
#include <linux/fcntl.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/module.h>
// for set_fs and get_fs
#include <asm/uaccess.h>
#include <kernel-interfaces/ioscheduler.h>
#include <linux/blkdev.h>
#include <linux/fs.h>

// trace headers included
#include <kml_lib.h>
#include <trace/events/block.h>
// #include <ioscheduler_net.h>
#include <decision_tree.h>
#include <ioscheduler_net_classification.h>
#include <utility.h>

MODULE_AUTHOR("Umit Akgun");
MODULE_LICENSE("GPL");

static struct task_struct *kml_ioscheduler_update_thread;
static struct task_struct *kml_perf_monitoring_thread;

enum kml_perf_stats {
  ops_interval = 0,
  ops_total = 1,
  ops_sec_interval = 2,
  ops_sec_total = 3,
  second_interval = 4,
  second_total = 5
} kml_perf_stats;

bool perf_monitoring_failed = false;
bool ioscheduler_update_failed = false;
bool module_exiting = false;

void ioscheduler_block_rq_insert(struct request *rq);
/*
void ioscheduler_add_to_page_cache(struct page *page);
void ioscheduler_mm_filemap_fsl_read(struct page *page);
void ioscheduler_fsl_writeback_dirty_page(struct page *page, struct inode *inode);
dev_t ioscheduler_get_tuning_device(void);
unsigned long ioscheduler_get_disk_ra_val(void);
void ioscheduler_create_per_file_structure(unsigned long ino,
                                         unsigned int ra_pages);
unsigned long ioscheduler_get_ra_pages_per_file(unsigned long ino);
*/
/*
static const int readrandom_ranking[] = {16,  32,  64,  8,   4,   2,   1,   128,
                                         192, 256, 320, 384, 448, 512, 576, 640,
                                         704, 768, 832, 896, 960, 1024};
static const int readreverse_ranking[] = {
    960, 832, 1024, 896, 768, 704, 576, 640, 512, 448, 384,
    320, 256, 192,  128, 64,  32,  16,  1,   2,   4,   8};
static const int readseq_ranking[] = {640, 1024, 960, 512, 576, 448, 768, 320,
                                      832, 704,  896, 384, 256, 192, 128, 64,
                                      32,  8,    4,   1,   16,  2};
static const int readrandomwriterandom_ranking[] = {
    16,  32,  64,  4,   1,   8,   2,   128, 192, 256, 320,
    384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024};
static const int *workload_rankigs[] = {readrandom_ranking,
                                        readrandomwriterandom_ranking,
                                        readseq_ranking, readreverse_ranking};
static const char workload_names[4][25] = {
    "readrandom", "readrandomwriterandom", "readseq", "readreverse"};
*/
static const char r1w1_read[10] = "2000000";
static const char r1w1_write[10] = "10000000";
static const char r1w32_read[10] = "2000000";
static const char r1w32_write[10] = "10000000";
static const char r32w1_read[10] = "2000000";
static const char r32w1_write[10] = "10000000";
static const char r32w32_read[10] = "2000000";
static const char r32w32_write[10] = "10000000";

static const char *workload_rankigs[] = {
  r1w1_read, r1w1_write, r1w32_read, r1w32_write,
  r32w1_read, r32w1_write, r32w32_read, r32w32_write};

static const char workload_names[4][25] = {
  "r1w1", "r1w32", "r32w1", "r32w32"};



static ioscheduler_class_net *ioscheduler = NULL;
static const int n_features = 2;
static u64 start_time = 0;
//static int current_ioscheduler_val = 0;

static u64 data_process_total = 0;
static atomic_int data_process_count;
static u64 inference_timing_total = 0;
static atomic_int inference_count;

extern ioscheduler_class_net *build_ioscheduler_class_net(
    ioscheduler_model_config *config);

#define LAST_N_SEC_OP 5
static long last_n_sec_perf[LAST_N_SEC_OP] = {0};
static int last_n_sec_perf_idx = 0;

static dev_t tunning_device_number;
//static unsigned long disk_base_ioscheduler_val;

static kml_dt *dt;

// tuning device
#define DEVICE_NAME "/dev/sda6"  // nvme0n1p1, sdd1

#define NN_INFERENCE

int ioscheduler_update(void *data) {
  struct block_device *bdev = NULL;
  int class = 0;
  u64 inference_start = 0, inference_end = 0;
  int log_idx = 0;
  long average_ops_sec = 0;
  int seconds = 0;
  filep read_lat, write_lat;
  char read_buf[10] = {0};
  char write_buf[10] = {0};
  unsigned long long read_offset = 0;
  unsigned long long write_offset = 0;
  // per file
  //int file_traverse = 0;
  //struct hlist_head *head = NULL;
  //ioscheduler_per_file_data *per_file_node = NULL;
  //int per_file_class;
#ifndef NN_INFERENCE
  matrix *normalized_data;
#endif
  int prediction_bucket[4] = {0};

  //bdev = blkdev_get_by_path(DEVICE_NAME, FMODE_READ | FMODE_WRITE, NULL);
  //if (IS_ERR(bdev)) {
    //ioscheduler_update_failed = true;
    //printk("kml ioscheduler update thread could not open block device %s\n",
    //       DEVICE_NAME);
    //return 0;
  //}
  //tunning_device_number = bdev->bd_dev;

  printk("opened block device %p\n", DEVICE_NAME);
  //blkdev_ioctl(bdev, 0, BLKRAGET, (unsigned long)&current_ioscheduler_val);
  //disk_base_ioscheduler_val = current_ioscheduler_val;

  while (1) {
    msleep(1000);
    if (kthread_should_stop()) {
      printk("ioscheduler_update thread should stop\n");
      break;
    }
    printk("----------------------- %d -----------------------", ++seconds);

    kernel_fpu_begin();
    inference_start = kml_get_current_time();
#ifdef NN_INFERENCE
    class = predict_ioscheduler_class(ioscheduler);
    // prediction_bucket[0] = 0;
    // prediction_bucket[1] = 0;
    // prediction_bucket[2] = 0;
    // prediction_bucket[3] = 0;
    //
    // for (file_traverse = 0; file_traverse < 1024; ++file_traverse) {
    //   head = &ioscheduler->ioscheduler_per_file_data_hlist[file_traverse];
    //   hlist_for_each_entry(per_file_node, head, hlist) {
    //     per_file_class = predict_ioscheduler_class_per_file(
    //         ioscheduler, per_file_node->ra_pages, per_file_node);
    //     //printk("ino: %ld\t class: %s\n", per_file_node->ino,
    //     //       workload_names[per_file_class]);
    //     per_file_node->predicted_ra_pages =
    //     workload_rankigs[per_file_class][0];
    // prediction_bucket[per_file_class]++;
    //  }
    //}
    // printk("per file prediction classes readrandom: %d rwrandom: %d readseq:
    // %d readreverse: %d\n",
    //   prediction_bucket[0],prediction_bucket[1],prediction_bucket[2],prediction_bucket[3]);
#else
    normalized_data =
        get_normalized_ioscheduler_data(ioscheduler, disk_base_ioscheduler_val);
    class = predict_decision_tree(dt, normalized_data);
    free_matrix(normalized_data);
#endif
    inference_end = kml_get_current_time();
    kernel_fpu_end();

    //disk_base_ioscheduler_val = workload_rankigs[class][0];
    read_buf = workload_rankigs[2*class];
    write_buf = workload_rankigs[(2*class)+1];

    read_lat = kml_file_open(
        "/sys/block/sda/queue/iosched/read_lat_nsec",
        "w+", O_RDWR);
    write_lat = kml_file_open(
        "/sys/block/sda//queue/iosched/write_lat_nsec", 
        "w+", O_RDWR)

    kml_file_write(read_lat, read_buf, 10, &read_offset);
    kml_file_write(write_lat, write_buf, 10, &write_offset);

    kml_file_close(read_lat);
    kml_file_close(write_lat);

    //blkdev_ioctl(bdev, 0, BLKRASET, disk_base_ioscheduler_val);
    printk("ioscheduler read_val set:\t\t %s\n", read_buf);
    printk("ioscheduler write_val set:\t\t %s\n", write_buf);

    printk("predicted class:\t\t %s\n", workload_names[class]);
    inference_timing_total += kml_get_time_diff(inference_end, inference_start);
    kml_atomic_add(&inference_count, 1);
    printk("inference took in avg.:\t\t %lld\n",
           inference_timing_total / kml_atomic_int_read(&inference_count));

    if (kml_atomic_int_read(&data_process_count) > 0) {
      printk("data processing took in avg.:\t %lld\n",
             data_process_total / kml_atomic_int_read(&data_process_count));
    }

#ifdef LOG_LAST_N
    printk(KERN_CONT "perf [ ");
#endif
    average_ops_sec = 0;
    for (log_idx = 0; log_idx < LAST_N_SEC_OP; ++log_idx) {
#ifdef LOG_LAST_N
      printk(KERN_CONT "%ld ", last_n_sec_perf[log_idx]);
#endif
      average_ops_sec += last_n_sec_perf[log_idx];
    }
#ifdef LOG_LAST_N
    printk(KERN_CONT "] -> %ld\n", average_ops_sec / LAST_N_SEC_OP);
#else
    printk("ops/sec average:\t\t %ld\n", average_ops_sec / LAST_N_SEC_OP);
#endif
  }

  return 0;
}
/*
int perf_monitoring(void *data) {
  size_t read_size;
  loff_t offset = 0;
  char buffer[256] = {0};
  long ops_interval_stat = 0, ops_total_stat = 0, ops_sec_interval_stat = 0,
       ops_sec_total_stat = 0, second_interval_stat = 0, second_total_stat = 0;

  struct file *perf_file = kml_file_open("/tmp/detail.txt", NULL, O_LARGEFILE);

  if (perf_file == NULL) {
    perf_monitoring_failed = true;
    printk("kml ioscheduler perf thread could not find perf file\n");
    return 0;
  }

  while (1) {
    if ((read_size = kernel_read(perf_file, buffer, 256, &offset)) == 0) {
      if (read_size == 0) {
        msleep(1000);
      }
    } else {
      int idx = 0, result = 0;
      char *token = NULL, *long_token = NULL, *split = buffer, *separator = " ";
      while ((token = strsep(&split, separator)) != NULL) {
        // printk("%d %s %d\n", idx, token, result);
        switch (idx) {
          case ops_interval: {
            result |= kstrtol(token, 10, &ops_interval_stat);
            break;
          }
          case ops_total: {
            result |= kstrtol(token, 10, &ops_total_stat);
            break;
          }
          case ops_sec_interval: {
            long_token = strsep(&token, ".");
            result |= kstrtol(long_token, 10, &ops_sec_interval_stat);
            break;
          }
          case ops_sec_total: {
            long_token = strsep(&token, ".");
            result |= kstrtol(long_token, 10, &ops_sec_total_stat);
            break;
          }
          case second_interval: {
            long_token = strsep(&token, ".");
            result |= kstrtol(long_token, 10, &second_interval_stat);
            break;
          }
          case second_total: {
            long_token = strsep(&token, ".");
            result |= kstrtol(long_token, 10, &second_total_stat);
            break;
          }
        }
        idx++;
      }
      if (result == 0) {
        // printk("kml perf stat ops/sec: %ld\n", ops_sec_interval_stat);
        last_n_sec_perf[last_n_sec_perf_idx] = ops_sec_interval_stat;
        last_n_sec_perf_idx = (last_n_sec_perf_idx + 1) % 5;
      }
    }

    if (kthread_should_stop()) {
      printk("perf_monitoring thread should stop\n");
      break;
    }
  }

  kml_file_close(perf_file);
  return 0;
}

*/
void ioscheduler_block_rq_insert(struct request *rq){
u64 time_passed;
u64 data_process_start, data_process_end;
double rw;
double data[2];

time_passed = kml_get_current_time();
time_passed = kml_get_time_diff(time_passed, start_time);

if((rq->flags & 1) == 0){       //identify read operation
  rw = 0;
}

if((rq->flags & 1) == 1){       //identify write operation
  rw = 1;
}

if(module_exiting) return;


kernel_fpu_begin();
data[0] = time_passed;
data[1] = rw;
data_process_start = kml_get_current_time();
if(ioscheduler_data_processing(data, (ioscheduler_net *)ioscheduler,
                              true, false)){
}

data_process_end = kml_get_current_time();

data_process_total = kml_get_time_diff(data_process_end, data_process_start);
kml_atomic_add(&data_process_count, 1);

kernel_fpu_end();

}
/*
void ioscheduler_add_to_page_cache(struct page *page) {
  unsigned long index = 0, i_ino = 0;
  u64 time_passed;
  u64 data_process_start, data_process_end;
  dev_t blk_dev_no = 0;
  double data[3];

  if (page != NULL) {
    index = page->index;
    if (page->mapping != NULL && page->mapping->host != NULL) {
      i_ino = page->mapping->host->i_ino;
      blk_dev_no = page->mapping->host->i_sb ? page->mapping->host->i_sb->s_dev
                                             : page->mapping->host->i_rdev;
      if (blk_dev_no != tunning_device_number) {
        return;
      }
    }
  }

  time_passed = kml_get_current_time();
  time_passed = kml_get_time_diff(time_passed, start_time);

  if (module_exiting) return;

  kernel_fpu_begin();
  data[0] = time_passed;
  data[1] = i_ino;
  data[2] = index;

  data_process_start = kml_get_current_time();
  if (ioscheduler_data_processing(data, (ioscheduler_net *)ioscheduler,
                                disk_base_ioscheduler_val, true, false, i_ino)) {
  }
  data_process_end = kml_get_current_time();

  data_process_total += kml_get_time_diff(data_process_end, data_process_start);
  kml_atomic_add(&data_process_count, 1);

  kernel_fpu_end();
}

dev_t ioscheduler_get_tuning_device(void) { return tunning_device_number; }

unsigned long ioscheduler_get_disk_ra_val(void) {
  return disk_base_ioscheduler_val;
}

void ioscheduler_create_per_file_structure(unsigned long ino,
                                         unsigned int ra_pages) {
  ioscheduler_create_per_file_data(ioscheduler, ino, ra_pages);
}

unsigned long ioscheduler_get_ra_pages_per_file(unsigned long ino) {
  uint64_t hash = 0;
  struct hlist_head *head = NULL;
  ioscheduler_per_file_data *per_file_node = NULL;
  unsigned long predicted_ra_pages = 0;

  hash = hash_64(ino, 64);
  head = &ioscheduler->ioscheduler_per_file_data_hlist[hash & 1023];
  hlist_for_each_entry(per_file_node, head, hlist) {
    if (per_file_node->ino == ino) {
      predicted_ra_pages = (unsigned long)per_file_node->predicted_ra_pages;
    }
  }

  return predicted_ra_pages;
}
*/
static int __init kml_ioscheduler_init(void) {
  ioscheduler_model_config config;
  filep mean_file, stddev_file;
  matrix *mean_matrix, *stddev_matrix;
  kernel_fpu_begin();
  config.batch_size = 1;
  config.learning_rate = 0.01;
  config.momentum = 0.99;
  config.num_features = n_features;
  config.model_type = FLOAT;
  ioscheduler = build_ioscheduler_class_net(&config);
  // decision tree

  set_weights_biases_from_file(
      ioscheduler->layer_list->layer_list_head,
      "/home/abbas/kernel-ml/"
      "ml-models-analyses/ioscheduler/user_nn_arch_data/linear0_w.csv",
      "/home/abbas/kernel-ml/"
      "ml-models-analyses/ioscheduler/user_nn_arch_data/"
      "linear0_bias.csv");
  set_weights_biases_from_file(
      ioscheduler->layer_list->layer_list_head->next->next,
      "/home/abbas/kernel-ml/ml-models-analyses/ioscheduler/"
      "user_nn_arch_data/"
      "linear1_w.csv",
      "/home/abbas/kernel-ml/ml-models-analyses/ioscheduler/"
      "user_nn_arch_data/"
      "linear1_bias.csv");
  set_weights_biases_from_file(
      ioscheduler->layer_list->layer_list_tail,
      "/home/abbas/kernel-ml/"
      "ml-models-analyses/ioscheduler/user_nn_arch_data/linear2_w.csv",
      "/home/abbas/kernel-ml/"
      "ml-models-analyses/ioscheduler/user_nn_arch_data/"
      "linear2_bias.csv");

  mean_matrix = allocate_matrix(1, n_features, FLOAT);
  stddev_matrix = allocate_matrix(1, n_features, FLOAT);
  mean_file = kml_file_open(
      "/home/abbas/python/trainmean.csv",
      "r", O_RDONLY);
  stddev_file = kml_file_open(
      "/home/abbas/python/trainstddev",
      "r", O_RDONLY);
  load_matrix_from_file(mean_file, mean_matrix);
  load_matrix_from_file(stddev_file, stddev_matrix);
  kml_file_close(mean_file);
  kml_file_close(stddev_file);

  set_ioscheduler_data(&ioscheduler->norm_data_stat, mean_matrix, stddev_matrix,
                     1510);

  // initializing decision tree
  //dt = build_decision_tree_model_from_file(
  //    "/home/kml/ml-models-analyses/ioscheduler-per-disk/"
  //    "decision_tree_model/"
  //    "ioscheduler_model.dt");
  // print_kml_decision_tree(dt->root);
  // class = predict_decision_tree(dt, row);
  kernel_fpu_end();

  udelay(1000);

  kml_ioscheduler_update_thread =
      kthread_run(&ioscheduler_update, NULL, "kml_ioscheduler_update");
  //kml_perf_monitoring_thread =
  //    kthread_run(&perf_monitoring, NULL, "kml_perf_monitoring");

  udelay(1000);
/*
  set_trace_ioscheduler_add_to_page_cache_fptr(
      (void *)&ioscheduler_add_to_page_cache);
  set_trace_ioscheduler_mm_filemap_fsl_read_fptr(
      (void *)&ioscheduler_mm_filemap_fsl_read);
  set_trace_ioscheduler_fsl_writeback_dirty_page_fptr(
      (void *)&ioscheduler_fsl_writeback_dirty_page);
  set_trace_ioscheduler_get_tuning_device_fptr(
      (void *)&ioscheduler_get_tuning_device);
  set_trace_ioscheduler_get_disk_ra_val_fptr((void *)&ioscheduler_get_disk_ra_val);
  */
  // set_trace_ioscheduler_create_per_file_structure_fptr(
  //     (void *)&ioscheduler_create_per_file_structure);
  // set_trace_ioscheduler_get_ra_pages_per_file(
  //     (void *)&ioscheduler_get_ra_pages_per_file);

  set_trace_ioscheduler_block_rq_insert_fptr(
      (void *)&ioscheduler_block_rq_insert);

  kml_atomic_int_init(&data_process_count, 0);
  kml_atomic_int_init(&inference_count, 0);
  start_time = kml_get_current_time();

  printk(KERN_WARNING "KML fio ioscheduler_update started");
  return 0;
}

static void __exit kml_ioscheduler_exit(void) {
  if (!ioscheduler_update_failed && kml_ioscheduler_update_thread != NULL)
    kthread_stop(kml_ioscheduler_update_thread);
  printk("kml ioscheduler update thread stopped\n");
  //if (!perf_monitoring_failed && kml_perf_monitoring_thread != NULL)
  //  kthread_stop(kml_perf_monitoring_thread);
  //printk("kml readahed perf thread stopped\n");
  udelay(1000);
  module_exiting = true;
  udelay(1000);
  /*
  set_trace_ioscheduler_add_to_page_cache_fptr((void *)NULL);
  set_trace_ioscheduler_mm_filemap_fsl_read_fptr((void *)NULL);
  set_trace_ioscheduler_fsl_writeback_dirty_page_fptr((void *)NULL);
  set_trace_ioscheduler_get_tuning_device_fptr((void *)NULL);
  set_trace_ioscheduler_get_disk_ra_val_fptr((void *)NULL);
  set_trace_ioscheduler_create_per_file_structure_fptr((void *)NULL);
  set_trace_ioscheduler_get_ra_pages_per_file((void *)NULL);
  */
  set_trace_ioscheduler_block_rq_insert_fptr((void *)NULL);
  udelay(1000);
  kernel_fpu_begin();
  clean_ioscheduler_class_net(ioscheduler);
  kernel_fpu_end();
  printk(KERN_WARNING "KML fio ioscheduler_update ended\n");
}

module_init(kml_ioscheduler_init);
module_exit(kml_ioscheduler_exit);
