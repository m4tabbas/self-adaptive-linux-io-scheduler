/*
 * Copyright (c) 2019-2021 Ibrahim Umit Akgun
 * Copyright (c) 2019-2021 Erez Zadok
 * Copyright (c) 2019-2021 Stony Brook University
 * Copyright (c) 2019-2021 The Research Foundation of SUNY
 *
 * You can redistribute it and/or modify it under the terms of the Apache
 * License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
 */

#include <kml_lib.h>
#include <ioscheduler_net_data.h>
#include <utility.h>

void ioscheduler_normalized_online_data(ioscheduler_net *ioscheduler,
                                      bool apply) {
  val n_seconds, n_1_seconds;
  matrix *diff = allocate_matrix(1, ioscheduler->online_data->cols,
                                 ioscheduler->online_data->type);
  matrix *local_average =
      allocate_matrix(ioscheduler->norm_data_stat.average->rows,
                      ioscheduler->norm_data_stat.average->cols,
                      ioscheduler->norm_data_stat.average->type);
  matrix *local_std_dev =
      allocate_matrix(ioscheduler->norm_data_stat.std_dev->rows,
                      ioscheduler->norm_data_stat.std_dev->cols,
                      ioscheduler->norm_data_stat.std_dev->type);
  matrix *local_variance =
      allocate_matrix(ioscheduler->norm_data_stat.variance->rows,
                      ioscheduler->norm_data_stat.variance->cols,
                      ioscheduler->norm_data_stat.variance->type);

  // setting values
  set_matrix_with_matrix(ioscheduler->norm_data_stat.average, local_average);
  set_matrix_with_matrix(ioscheduler->norm_data_stat.std_dev, local_std_dev);
  set_matrix_with_matrix(ioscheduler->norm_data_stat.variance, local_variance);
  /*
  ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0,
                                           ioscheduler->online_data->cols - 1)] =
      ((double)ioscheduler_val) / 1024;
  */
  ioscheduler->norm_data_stat.n_seconds++;
  n_seconds.d = (double)ioscheduler->norm_data_stat.n_seconds;
  n_1_seconds.d = (double)(ioscheduler->norm_data_stat.n_seconds - 1);

  matrix_mult_constant(local_average, &n_1_seconds, diff);
  matrix_add(ioscheduler->online_data, diff, diff);
  matrix_div_constant(diff, &n_seconds, diff);
  set_matrix_with_matrix(diff, local_average);
  // print_matrix(ioscheduler->norm_data_stat.average);

  matrix_sub(ioscheduler->online_data, ioscheduler->norm_data_stat.last_values,
             diff);
  matrix_elementwise_mult(diff, diff, diff);
  matrix_mult_constant(local_variance, &n_1_seconds, local_variance);
  matrix_add(local_variance, diff, local_variance);
  matrix_div_constant(local_variance, &n_seconds, local_variance);
  // print_matrix(ioscheduler->norm_data_stat.variance);

  matrix_map(local_variance, NULL, fast_sqrt_d, local_std_dev);
  // print_matrix(ioscheduler->norm_data_stat.std_dev);

  matrix_sub(ioscheduler->online_data, local_average,
             ioscheduler->norm_online_data);
  matrix_elementwise_div(ioscheduler->norm_online_data, local_std_dev,
                         ioscheduler->norm_online_data);

  set_matrix_with_matrix(ioscheduler->online_data,
                         ioscheduler->norm_data_stat.last_values);

  if (apply) {
    set_matrix_with_matrix(local_average, ioscheduler->norm_data_stat.average);
    set_matrix_with_matrix(local_std_dev, ioscheduler->norm_data_stat.std_dev);
    set_matrix_with_matrix(local_variance, ioscheduler->norm_data_stat.variance);
  }

  free_matrix(diff);
  free_matrix(local_average);
  free_matrix(local_std_dev);
  free_matrix(local_variance);
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(ioscheduler_normalized_online_data);
#endif

#ifdef KML_KERNEL
void ioscheduler_normalized_online_data_per_file(
    ioscheduler_net *ioscheduler, int ioscheduler_val, bool apply,
    ioscheduler_per_file_data *per_file_data) {
  val n_seconds, n_1_seconds;
  matrix *diff, *local_average, *local_std_dev, *local_variance;

  if (per_file_data == NULL) {
    return;
  }

  diff = allocate_matrix(1, per_file_data->online_data->cols,
                         per_file_data->online_data->type);
  local_average = allocate_matrix(per_file_data->norm_data_stat.average->rows,
                                  per_file_data->norm_data_stat.average->cols,
                                  per_file_data->norm_data_stat.average->type);
  local_std_dev = allocate_matrix(per_file_data->norm_data_stat.std_dev->rows,
                                  per_file_data->norm_data_stat.std_dev->cols,
                                  per_file_data->norm_data_stat.std_dev->type);
  local_variance =
      allocate_matrix(per_file_data->norm_data_stat.variance->rows,
                      per_file_data->norm_data_stat.variance->cols,
                      per_file_data->norm_data_stat.variance->type);

  // setting values
  set_matrix_with_matrix(ioscheduler->norm_data_stat.average, local_average);
  set_matrix_with_matrix(ioscheduler->norm_data_stat.std_dev, local_std_dev);
  set_matrix_with_matrix(ioscheduler->norm_data_stat.variance, local_variance);

  per_file_data->online_data->vals.d[mat_index(
      per_file_data->online_data, 0, per_file_data->online_data->cols - 1)] =
      ((double)ioscheduler_val) / 1024;
  n_seconds.d = (double)ioscheduler->norm_data_stat.n_seconds;
  n_1_seconds.d = (double)(ioscheduler->norm_data_stat.n_seconds - 1);

  matrix_mult_constant(local_average, &n_1_seconds, diff);
  matrix_add(per_file_data->online_data, diff, diff);
  matrix_div_constant(diff, &n_seconds, diff);
  set_matrix_with_matrix(diff, local_average);
  // print_matrix(ioscheduler->norm_data_stat.average);

  matrix_sub(per_file_data->online_data,
             per_file_data->norm_data_stat.last_values, diff);
  matrix_elementwise_mult(diff, diff, diff);
  matrix_mult_constant(local_variance, &n_1_seconds, local_variance);
  matrix_add(local_variance, diff, local_variance);
  matrix_div_constant(local_variance, &n_seconds, local_variance);
  // print_matrix(ioscheduler->norm_data_stat.variance);

  matrix_map(local_variance, NULL, fast_sqrt_d, local_std_dev);
  // print_matrix(ioscheduler->norm_data_stat.std_dev);

  matrix_sub(per_file_data->online_data, local_average,
             per_file_data->norm_online_data);
  matrix_elementwise_div(per_file_data->norm_online_data, local_std_dev,
                         per_file_data->norm_online_data);

  set_matrix_with_matrix(per_file_data->online_data,
                         per_file_data->norm_data_stat.last_values);

  /* if (apply) { */
  /*   set_matrix_with_matrix(local_average, */
  /*                          per_file_data->norm_data_stat.average); */
  /*   set_matrix_with_matrix(local_std_dev, */
  /*                          per_file_data->norm_data_stat.std_dev); */
  /*   set_matrix_with_matrix(local_variance, */
  /*                          per_file_data->norm_data_stat.variance); */
  /* } */

  free_matrix(diff);
  free_matrix(local_average);
  free_matrix(local_std_dev);
  free_matrix(local_variance);
}
EXPORT_SYMBOL(ioscheduler_normalized_online_data_per_file);
#endif

void ioscheduler_online_data_update(double pg_idx, ioscheduler_net *ioscheduler) {
  double delta_pg_idx = pg_idx - ioscheduler->online_data_stat.moving_average;
  ioscheduler->online_data_stat.moving_average +=
      delta_pg_idx / ioscheduler->online_data_stat.n_transactions;
  ioscheduler->online_data_stat.moving_m2 +=
      delta_pg_idx * (pg_idx - ioscheduler->online_data_stat.moving_average);

  //ioscheduler->online_data_stat.diff_s = time_origin - ioscheduler->online_data_stat.last_event_time;
}

#ifdef KML_KERNEL
void ioscheduler_per_file_online_data_update(double pg_idx,
                                           ioscheduler_per_file_data *data) {
  double delta_pg_idx = pg_idx - data->online_data_stat.moving_average;
  data->online_data_stat.moving_average +=
      delta_pg_idx / data->online_data_stat.n_transactions;
  data->online_data_stat.moving_m2 +=
      delta_pg_idx * (pg_idx - data->online_data_stat.moving_average);
}
#endif

// data[] 0->time/nanosecond 1->ino 2->pg_idx
// return true -> finalized the second or not
bool ioscheduler_data_processing(double *data, ioscheduler_net *ioscheduler,
                               bool apply, bool reset)
                               {
  static double read_timing_starts = 0;
  static double write_timing_starts = 0;
  static double last_read_event_time = -1;
  static double last_write_event_time = -1;
  static double total_read_event_time_diffs = 0;
  static double total_write_event_time_diffs = 0;
  //static double min_event_time_norm = DBL_MAX , max_event_time_norm = DBL_MIN;
  //static int last_pg_idx = -1;
  //static double total_pg_idx_diffs = 0;
  //static int min_pg_idx_norm = INT_MAX, max_pg_idx_norm = INT_MIN;
#ifdef KML_KERNEL
  ioscheduler_per_file_data *per_file_data =
      ioscheduler_get_per_file_data((ioscheduler_class_net *)ioscheduler, ino);
#endif

  if (reset) {
    read_timing_starts = 0;
    write_timing_starts = 0;
#ifdef KML_KERNEL
    // per-file
    if (per_file_data != NULL) {
      per_file_data->timing_starts = 0;
    }
#endif
    return false;
  }

  if (ioscheduler->online_data_stat.read_n_transactions == 0 && data[1] == 0) {
    read_timing_starts = data[0];
  }
  if (ioscheduler->online_data_stat.write_n_transactions == 0 && data[1] == 1) {
    write_timing_starts = data[0];
  }

#ifdef KML_KERNEL
  // per-file
  if (per_file_data != NULL &&
      per_file_data->online_data_stat.n_transactions == 0) {
    per_file_data->timing_starts = data[0];
  }
#endif

  // eliminating superblock accesses
  //if (data[2] > 1e6) {
  //  return false;
  //}

  // number of transactions
  if(data[1] == 0){
    ioscheduler->online_data_stat.read_n_transactions++;
  }
  if(data[1] == 1){
    ioscheduler->online_data_stat.write_n_transactions++;
  }
#ifdef KML_KERNEL
  // per-file
  if (per_file_data != NULL) {
    per_file_data->online_data_stat.n_transactions++;
  }
#endif
  if(last_read_event_time != -1 && data[1] == 0){
    total_read_event_time_diffs += (data[0] - last_read_event_time);
  }
  if(last_write_event_time != -1 && data[1] == 1){
    total_write_event_time_diffs += (data[0] - last_write_event_time);
  }

  if(data[1] == 0){
    last_read_event_time = data[0];
  }
  if(data[1] == 1){
    last_write_event_time = data[0];
  }
  // pg_idx diffs
  //if (last_pg_idx != -1) {
  //  total_pg_idx_diffs += (abs((int)data[2] - last_pg_idx));
  //}
  //last_pg_idx = data[2];
/*
#ifdef KML_KERNEL
  // per-file
  if (per_file_data != NULL) {
    if (per_file_data->last_pg_idx != -1) {
      per_file_data->total_pg_idx_diffs +=
          (abs((int)data[2] - per_file_data->last_pg_idx));
    }
    per_file_data->last_pg_idx = data[2];
  }
#endif
*/
  //ioscheduler_online_data_update(data[0], ioscheduler);
/*
#ifdef KML_KERNEL
  if (per_file_data != NULL) {
    ioscheduler_per_file_online_data_update(data[2], per_file_data);
  }
#endif
*/
/*
  if (data[0] < min_event_time_norm) {
    min_event_time_norm = data[0];
  }
  if (data[0] > max_event_time_norm) {
    max_event_time_norm = data[0];
  }
*/
#ifdef KML_KERNEL
  if (per_file_data != NULL) {
    if (data[2] < per_file_data->min_pg_idx_norm) {
      per_file_data->min_pg_idx_norm = data[2];
    }
    if (data[2] > per_file_data->max_pg_idx_norm) {
      per_file_data->max_pg_idx_norm = data[2];
    }
  }
#endif

#ifdef KML_KERNEL
  if (per_file_data != NULL && data[0] - per_file_data->timing_starts > 1e9) {
    int normalized_range =
        per_file_data->max_pg_idx_norm - per_file_data->min_pg_idx_norm;
    per_file_data->online_data->vals
        .d[mat_index(per_file_data->online_data, 0, 0)] =
        per_file_data->online_data_stat.n_transactions;
    per_file_data->online_data->vals
        .d[mat_index(per_file_data->online_data, 0, 1)] =
        per_file_data->online_data_stat.moving_m2 /
        per_file_data->online_data_stat.n_transactions;
    per_file_data->online_data->vals
        .d[mat_index(per_file_data->online_data, 0, 1)] /= normalized_range;
    per_file_data->online_data->vals
        .d[mat_index(per_file_data->online_data, 0, 2)] =
        per_file_data->online_data_stat.moving_m2 /
        (per_file_data->online_data_stat.n_transactions - 1);
    per_file_data->online_data->vals
        .d[mat_index(per_file_data->online_data, 0, 2)] /= normalized_range;
    per_file_data->online_data->vals
        .d[mat_index(per_file_data->online_data, 0, 3)] =
        per_file_data->total_pg_idx_diffs /
        per_file_data->online_data_stat.n_transactions;

    // kml_debug("-------------------- per-file ------------------------\n");
    // printk("inode no: %ld ra pages: %d\n", ino, per_file_data->ra_pages * 8);
    ioscheduler_normalized_online_data_per_file(
        ioscheduler, per_file_data->ra_pages * 8, apply, per_file_data);
    // kml_debug("per file non-normalized data:\n");
    // print_matrix(matrix_float_conversion(per_file_data->online_data));
    // kml_debug("per file normalized data:\n");
    // print_matrix(matrix_float_conversion(per_file_data->norm_online_data));
    // kml_debug("------------------------------------------------------\n");

    per_file_data->online_data_stat.n_transactions = 0;
    per_file_data->online_data_stat.moving_average = 0;
    per_file_data->online_data_stat.moving_m2 = 0;
    per_file_data->min_pg_idx_norm = INT_MAX;
    per_file_data->max_pg_idx_norm = INT_MIN;
    per_file_data->timing_starts = data[0];
    per_file_data->total_pg_idx_diffs = 0;
    per_file_data->last_pg_idx = -1;
  }
#endif

  if (data[1] == 0 && data[0] - read_timing_starts > 1e9) {
  /*
    int normalized_range = max_event_time_norm - min_event_time_norm;
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 0)] =
        ioscheduler->online_data_stat.n_transactions;
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 1)] =
        ioscheduler->online_data_stat.moving_m2 /
        ioscheduler->online_data_stat.n_transactions;
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 1)] /=
        normalized_range;
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 2)] =
        ioscheduler->online_data_stat.moving_m2 /
        (ioscheduler->online_data_stat.n_transactions - 1);
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 2)] /=
        normalized_range;
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 3)] =
        total_pg_idx_diffs / ioscheduler->online_data_stat.n_transactions;
    */
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 0)] =
        total_read_event_time_diffs / ioscheduler->online_data_stat.read_n_transactions;
    ioscheduler->online_data->vals.d[mat_index(ioscheduler->online_data, 0, 1)] =
        total_write_event_time_diffs / ioscheduler->online_data_stat.write_n_transactions;

    // kml_debug("++++++++++++++++++++ per-disk ++++++++++++++++++++++++\n");
    ioscheduler_normalized_online_data(ioscheduler, apply);
    // kml_debug("non-normalized data:\n");
    // print_matrix(matrix_float_conversion(ioscheduler->online_data));
    // kml_debug("normalized data:\n");
    // print_matrix(matrix_float_conversion(ioscheduler->norm_online_data));
    // kml_debug("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    ioscheduler->online_data_stat.read_n_transactions = 0;
    ioscheduler->online_data_stat.write_n_transactions = 0;
    //ioscheduler->online_data_stat.moving_average = 0;
    //ioscheduler->online_data_stat.moving_m2 = 0;
    //min_pg_idx_norm = INT_MAX;
    //max_pg_idx_norm = INT_MIN;
    read_timing_starts = data[0];
    total_read_event_time_diffs = 0;
    total_write_event_time_diffs = 0;
    last_read_event_time = -1;
    last_write_event_time = -1;
    //total_pg_idx_diffs = 0;
    //last_pg_idx = -1;
    return true;
  }

  return false;
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(ioscheduler_data_processing);
#endif

#ifdef KML_KERNEL
void ioscheduler_create_per_file_data(ioscheduler_class_net *ioscheduler,
                                    unsigned long ino, unsigned int ra_pages) {
  uint64_t hash = 0;
  struct hlist_head *head = NULL;
  ioscheduler_per_file_data *per_file_node = NULL, *found_file_node = NULL;
  bool file_found = false;

  hash = hash_64(ino, 64);
  head = &ioscheduler->ioscheduler_per_file_data_hlist[hash & 1023];
  hlist_for_each_entry(per_file_node, head, hlist) {
    if (per_file_node->ino == ino) {
      found_file_node = per_file_node;
      file_found = true;
    }
  }

  if (!file_found) {
    per_file_node = kml_calloc(sizeof(ioscheduler_per_file_data), 1);
    if (per_file_node) {
      per_file_node->ino = ino;
      per_file_node->ra_pages = ra_pages;
      per_file_node->online_data = allocate_matrix(
          ioscheduler->online_data->rows, ioscheduler->online_data->cols,
          ioscheduler->online_data->type);
      per_file_node->norm_online_data = allocate_matrix(
          ioscheduler->norm_online_data->rows, ioscheduler->norm_online_data->cols,
          ioscheduler->norm_online_data->type);
      per_file_node->norm_data_stat.average =
          allocate_matrix(ioscheduler->norm_data_stat.average->rows,
                          ioscheduler->norm_data_stat.average->cols,
                          ioscheduler->norm_data_stat.average->type);
      per_file_node->norm_data_stat.std_dev =
          allocate_matrix(ioscheduler->norm_data_stat.std_dev->rows,
                          ioscheduler->norm_data_stat.std_dev->cols,
                          ioscheduler->norm_data_stat.std_dev->type);
      per_file_node->norm_data_stat.variance =
          allocate_matrix(ioscheduler->norm_data_stat.variance->rows,
                          ioscheduler->norm_data_stat.variance->cols,
                          ioscheduler->norm_data_stat.variance->type);
      per_file_node->norm_data_stat.last_values =
          allocate_matrix(ioscheduler->norm_data_stat.last_values->rows,
                          ioscheduler->norm_data_stat.last_values->cols,
                          ioscheduler->norm_data_stat.last_values->type);

      hlist_add_head_rcu(&per_file_node->hlist, head);
      // printk(KERN_INFO "pf ds created %ld %d\n", per_file_node->ino,
      //        per_file_node->ra_pages);
    }
  } else {
    found_file_node->ra_pages = ra_pages;
  }
}
EXPORT_SYMBOL(ioscheduler_create_per_file_data);

ioscheduler_per_file_data *ioscheduler_get_per_file_data(
    ioscheduler_class_net *ioscheduler, unsigned long ino) {
  uint64_t hash = 0;
  struct hlist_head *head = NULL;
  ioscheduler_per_file_data *per_file_node = NULL, *ret_node = NULL;

  hash = hash_64(ino, 64);
  head = &ioscheduler->ioscheduler_per_file_data_hlist[hash & 1023];
  hlist_for_each_entry(per_file_node, head, hlist) {
    if (per_file_node->ino == ino) {
      ret_node = per_file_node;
    }
  }

  return ret_node;
}
EXPORT_SYMBOL(ioscheduler_get_per_file_data);
#endif
