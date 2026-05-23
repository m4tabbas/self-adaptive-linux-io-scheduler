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
#include <ioscheduler_net_classification.h>
#include <utility.h>

#define N_FEATURES 2
#define N_SECONDS_TRAINING 1510
#define N_ITERATIONS 1000
static const int ioscheduler_test_list[12] = {8,   16,  32,  64,  128, 256,
                                            384, 512, 640, 768, 896, 1024};
static const char workload_names[4][25] = {
    "readrandom", "readrandomwriterandom", "readseq", "readreverse"};
static matrix *input_matrix;
static matrix *output_matrix;

void preparing_ml_xy(void) {
  filep processing_file = NULL;
  int feature_file_idx = 0, feature_workload_idx = 0;
  int seconds = 0, total_seconds = 0;
  float seconds_f = 0;
  char processing_file_name[255] = {0};
  int feature_idx;

  for (feature_workload_idx = 0; feature_workload_idx < 4;
       ++feature_workload_idx) {
    for (feature_file_idx = 0;
         feature_file_idx < sizeof(ioscheduler_test_list) / sizeof(int);
         feature_file_idx++) {
      int ret =
          snprintf(processing_file_name, 255,
                   "../ml-models-analyses/ioscheduler-per-disk/training_analysis/"
                   "nvme_features/feature_%s_%d.csv",
                   workload_names[feature_workload_idx],
                   ioscheduler_test_list[feature_file_idx]);

      if (!(ret > 0 && ret < 255)) {
        kml_assert(false);
      }

      processing_file = kml_file_open(processing_file_name, "r", O_RDONLY);
      kml_fscanf(processing_file, "%f", &seconds_f);
      seconds = (int)seconds_f;

      // printf("%s %d -> seconds %d\n", workload_names[feature_workload_idx],
      //        ioscheduler_test_list[feature_file_idx], seconds);
      //  printf("total seconds %d\n", total_seconds);

      int second_idx = 0;
      for (; second_idx < seconds; ++second_idx) {
        for (feature_idx = 0; feature_idx < N_FEATURES - 1; ++feature_idx) {
          float current_feature;
          kml_fscanf(processing_file, "%f", &current_feature);
          switch (input_matrix->type) {
            case FLOAT:
              input_matrix->vals.f[mat_index(
                  input_matrix, (total_seconds + second_idx), feature_idx)] =
                  current_feature;
              break;
            case DOUBLE:
              input_matrix->vals.d[mat_index(
                  input_matrix, (total_seconds + second_idx), feature_idx)] =
                  (double)current_feature;
              break;
            case INTEGER:
              kml_assert(false);  // not implemented
              break;
          }
          if (feature_workload_idx == 2 || feature_workload_idx == 3) {
            switch (input_matrix->type) {
              case FLOAT:
                input_matrix->vals.f[mat_index(
                    input_matrix, (total_seconds + seconds + second_idx),
                    feature_idx)] = current_feature;
                break;
              case DOUBLE:
                input_matrix->vals.d[mat_index(
                    input_matrix, (total_seconds + seconds + second_idx),
                    feature_idx)] = (double)current_feature;
                break;
              case INTEGER:
                kml_assert(false);  // not implemented
                break;
            }
          }
        }
        switch (input_matrix->type) {
          case FLOAT:
            input_matrix->vals.f[mat_index(
                input_matrix, (total_seconds + second_idx), feature_idx)] =
                (float)ioscheduler_test_list[feature_file_idx] / 1024;
            break;
          case DOUBLE:
            input_matrix->vals.d[mat_index(
                input_matrix, (total_seconds + second_idx), feature_idx)] =
                (double)ioscheduler_test_list[feature_file_idx] / 1024;
            break;
          case INTEGER:
            kml_assert(false);  // not implemented
            break;
        }
        output_matrix->vals
            .i[mat_index(output_matrix, (total_seconds + second_idx), 0)] =
            feature_workload_idx;
        if (feature_workload_idx == 2 || feature_workload_idx == 3) {
          switch (input_matrix->type) {
            case FLOAT:
              input_matrix->vals.f[mat_index(
                  input_matrix, (total_seconds + seconds + second_idx),
                  feature_idx)] =
                  (float)ioscheduler_test_list[feature_file_idx] / 1024;
              break;
            case DOUBLE:
              input_matrix->vals.d[mat_index(
                  input_matrix, (total_seconds + seconds + second_idx),
                  feature_idx)] =
                  (double)ioscheduler_test_list[feature_file_idx] / 1024;
              break;
            case INTEGER:
              kml_assert(false);  // not implemented
              break;
          }
          output_matrix->vals.i[mat_index(
              output_matrix, (total_seconds + seconds + second_idx), 0)] =
              feature_workload_idx;
        }
      }

      kml_file_close(processing_file);
      total_seconds += (feature_workload_idx == 2 || feature_workload_idx == 3)
                           ? (seconds * 2)
                           : seconds;
    }
  }

  printf("total seconds: %d\n", total_seconds);
  matrix *old_input_matrix = input_matrix;
  input_matrix = matrix_zscore(input_matrix, 1);
  free_matrix(old_input_matrix);
  // print_matrix(input_matrix);
}

int main(void) {
  int epoch = N_ITERATIONS, i;
  ioscheduler_model_config config;
  filep input_file, output_file, std_dev_file, mean_file;
  matrix *mean = NULL, *std_dev = NULL;
  val modula_f = {.f = 10};

  config.batch_size = N_SECONDS_TRAINING;
  config.learning_rate = 0.01;
  config.momentum = 0.99;
  config.num_features = N_FEATURES;
  config.model_type = FLOAT;
  ioscheduler_class_net *ioscheduler = build_ioscheduler_class_net(&config);
  ioscheduler->state.is_training = true;
  set_random_weights(ioscheduler->layer_list, modula_f);

  input_matrix =
      allocate_matrix(N_SECONDS_TRAINING, N_FEATURES, ioscheduler->type);
  output_matrix = allocate_matrix(N_SECONDS_TRAINING, 1, INTEGER);

  input_file = kml_file_open(
      "/home/abbas/python/traininput.csv",
      "r", O_RDONLY);
  output_file = kml_file_open(
      "/home/abbas/python/trainoutput.csv",
      "r", O_RDONLY);
  std_dev_file = kml_file_open(
      "/home/abbas/python/trainstddev.csv",
      "w+", O_RDWR);
  mean_file = kml_file_open(
      "/home/abbas/python/trainmean.csv",
      "w+", O_RDWR);

  load_matrix_from_file(input_file, input_matrix);
  load_matrix_from_file(output_file, output_matrix);

  mean = matrix_mean(input_matrix, 1);
  std_dev = matrix_stddev(input_matrix, mean, 1);
  save_matrix_to_file(mean_file, mean);
  save_matrix_to_file(std_dev_file, std_dev);

  kml_file_close(mean_file);
  kml_file_close(std_dev_file);
  kml_file_close(input_file);
  kml_file_close(output_file);

  input_matrix = matrix_zscore(input_matrix, 1);
  // preparing_ml_xy();

  ioscheduler->data.input = input_matrix;
  ioscheduler->data.output = output_matrix;
  set_ioscheduler_data(&ioscheduler->norm_data_stat, mean, std_dev,
                     input_matrix->rows);

  for (i = 0; i < epoch; i++) {
    ioscheduler_class_net_train(ioscheduler);
    if ((i % 1000) == 0) {
      printf("epoch: %d loss :%f\n", i, ioscheduler->current_loss);
    }
  }

  print_weigths(ioscheduler->layer_list);
  print_biases(ioscheduler->layer_list);

  save_weights_biases_to_file(ioscheduler->layer_list->layer_list_head,
                              "../ml-models-analyses/ioscheduler/"
                              "user_nn_arch_data/linear0_w.csv",
                              "../ml-models-analyses/ioscheduler/"
                              "user_nn_arch_data/linear0_bias.csv");
  save_weights_biases_to_file(
      ioscheduler->layer_list->layer_list_head->next->next,
      "../ml-models-analyses/ioscheduler/user_nn_arch_data/"
      "linear1_w.csv",
      "../ml-models-analyses/ioscheduler/user_nn_arch_data/"
      "linear1_bias.csv");
  save_weights_biases_to_file(ioscheduler->layer_list->layer_list_tail,
                              "../ml-models-analyses/ioscheduler/"
                              "user_nn_arch_data/linear2_w.csv",
                              "../ml-models-analyses/ioscheduler/"
                              "user_nn_arch_data/linear2_bias.csv");

  clean_ioscheduler_class_net(ioscheduler);
  free_matrix(input_matrix);
  free_matrix(output_matrix);
}
