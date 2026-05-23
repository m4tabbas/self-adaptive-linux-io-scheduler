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

matrix *ioscheduler_class_net_inference(matrix *input,
                                      ioscheduler_class_net *ioscheduler) {
  return autodiff_forward(ioscheduler->layer_list, input);
}

void ioscheduler_class_net_train(ioscheduler_class_net *ioscheduler) {
  matrix *prediction;
  cross_entropy_loss *cross_entropy_l;
  val *loss_result;

  //================================= forward =================================
  prediction = ioscheduler_class_net_inference(ioscheduler->data.input, ioscheduler);

  //================================= backward ================================
  cross_entropy_l = (cross_entropy_loss *)ioscheduler->loss->internal;
  set_cross_entropy_loss_parameters(cross_entropy_l, prediction,
                                    ioscheduler->data.output);
  cross_entropy_loss_functions.derivative(cross_entropy_l);
  autodiff_backward(ioscheduler->layer_list, cross_entropy_l->derivative);

  //============================== optimization ===============================
  sgd_optimize(ioscheduler->sgd, ioscheduler->batch_size);

//================================= debug ===================================
#ifdef ML_MODEL_DEBUG
  print_weigths(ioscheduler->layer_list);
  print_biases(ioscheduler->layer_list);
#endif
  loss_result = cross_entropy_loss_functions.compute(cross_entropy_l);
  switch (ioscheduler->type) {
    case FLOAT:
      ioscheduler->current_loss = loss_result->f / ioscheduler->batch_size;
      break;
    case DOUBLE:
      ioscheduler->current_loss = (float)loss_result->d / ioscheduler->batch_size;
      break;
    case INTEGER:
      kml_assert(false);
      break;
  }

  cleanup_autodiff(ioscheduler->layer_list);
  kml_free(loss_result);
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(ioscheduler_class_net_train);
#endif

int ioscheduler_class_net_test(ioscheduler_class_net *ioscheduler, matrix **result) {
  int correct_prediction = 0;
  int row_idx, col_idx;
  val y_hat_class, y_class;

  matrix *y_hat =
      ioscheduler_class_net_inference(ioscheduler->data.input, ioscheduler);

  foreach_mat(y_hat, rows, row_idx){foreach_mat(y_hat, cols, col_idx){
      y_hat_class.f = y_hat->vals.f[mat_index(y_hat, row_idx, col_idx)];
  y_class.f = ioscheduler->data.output->vals
                  .f[mat_index(ioscheduler->data.output, row_idx, col_idx)];
  if (ioscheduler->check_correctness(y_class, y_hat_class)) {
    correct_prediction++;
  }
}
}
*result = copy_matrix(y_hat);

cleanup_autodiff(ioscheduler->layer_list);
return correct_prediction;
}

thread_ret ioscheduler_class_net_train_inference(void *ioscheduler_reg) {
  matrix *result;
  ioscheduler_class_net *ioscheduler = (ioscheduler_class_net *)ioscheduler_reg;

  if (kml_atomic_bool_read(&(ioscheduler->state.is_training))) {
    ioscheduler_class_net_train(ioscheduler);
  } else {
    kml_atomic_add(&(ioscheduler->state.num_accurate_predictions),
                   ioscheduler_class_net_test(ioscheduler, &result));
    free_matrix(result);
  }

  return DEFAULT_THREAD_RET;
}

void set_ioscheduler_data_constant(ioscheduler_norm_data_stat *norm_data_stat) {
  norm_data_stat->average->vals.d[mat_index(norm_data_stat->average, 0, 0)] =
      51930.47353497164L;
  norm_data_stat->average->vals.d[mat_index(norm_data_stat->average, 0, 1)] =
      512.8280933468811L;
  // std_dev
  norm_data_stat->std_dev->vals.d[mat_index(norm_data_stat->std_dev, 0, 0)] =
      43471.872061385984L;
  norm_data_stat->std_dev->vals.d[mat_index(norm_data_stat->std_dev, 0, 1)] =
      80.55386095014184L;
  // variance
  matrix_elementwise_mult(norm_data_stat->std_dev, norm_data_stat->std_dev,
                          norm_data_stat->variance);
  // last values
  set_matrix_with_matrix(norm_data_stat->average, norm_data_stat->last_values);
  norm_data_stat->n_seconds = 2640;
}

void set_ioscheduler_data(ioscheduler_norm_data_stat *norm_data_stat, matrix *mean,
                        matrix *std_dev, int n_dataset_size) {
  switch (norm_data_stat->average->type) {
    case DOUBLE: {
      set_matrix_with_matrix(matrix_double_conversion(mean),
                             norm_data_stat->average);
      break;
    }
    case FLOAT: {
      set_matrix_with_matrix(matrix_float_conversion(mean),
                             norm_data_stat->average);
      break;
    }
    case INTEGER: {
      kml_assert(false);  // TODO not implemented
      break;
    }
  }

  switch (norm_data_stat->std_dev->type) {
    case DOUBLE: {
      set_matrix_with_matrix(matrix_double_conversion(std_dev),
                             norm_data_stat->std_dev);
      break;
    }
    case FLOAT: {
      set_matrix_with_matrix(matrix_float_conversion(std_dev),
                             norm_data_stat->std_dev);
      break;
    }
    case INTEGER: {
      kml_assert(false);  // TODO not implemented
      break;
    }
  }
  // variance
  matrix_elementwise_mult(norm_data_stat->std_dev, norm_data_stat->std_dev,
                          norm_data_stat->variance);
  // last values
  set_matrix_with_matrix(norm_data_stat->average, norm_data_stat->last_values);
  norm_data_stat->n_seconds = n_dataset_size;
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(set_ioscheduler_data);
#endif

ioscheduler_class_net *build_ioscheduler_class_net(ioscheduler_model_config *config) {
  ioscheduler_class_net *ioscheduler;
#ifdef USE_INTERNAL_MEMORY_ALLOCATOR
  memory_pool_init();
#endif
  ioscheduler = kml_calloc(1, sizeof(ioscheduler_class_net));

  ioscheduler->data.collect_input =
      allocate_matrix(config->batch_size, config->num_features, DOUBLE);
  ioscheduler->data.collect_output =
      allocate_matrix(config->batch_size, 1, DOUBLE);
  ioscheduler->online_data = allocate_matrix(1, config->num_features, DOUBLE);
  ioscheduler->norm_online_data =
      allocate_matrix(1, config->num_features, DOUBLE);
  ioscheduler->norm_data_stat.average =
      allocate_matrix(1, config->num_features, DOUBLE);
  ioscheduler->norm_data_stat.std_dev =
      allocate_matrix(1, config->num_features, DOUBLE);
  ioscheduler->norm_data_stat.variance =
      allocate_matrix(1, config->num_features, DOUBLE);
  ioscheduler->norm_data_stat.last_values =
      allocate_matrix(1, config->num_features, DOUBLE);

  // dataset initialization
  set_ioscheduler_data_constant(&(ioscheduler->norm_data_stat));

  ioscheduler->type = config->model_type;
  ioscheduler->batch_size = config->batch_size;
  kml_atomic_bool_init(&(ioscheduler->state.is_training), true);
  kml_atomic_int_init(&(ioscheduler->state.num_accurate_predictions), 0);
  ioscheduler->loss =
      build_loss(build_cross_entropy_loss(NULL, NULL), CROSS_ENTROPY_LOSS);

  ioscheduler->layer_list = allocate_layers();

  add_layer(ioscheduler->layer_list,
            allocate_layer(
                build_linear_layer(config->num_features, 4, config->model_type),
                LINEAR_LAYER));

  add_layer(ioscheduler->layer_list,
            allocate_layer(
                build_sigmoid_layer(config->num_features, config->num_features,
                                    config->model_type),
                SIGMOID_LAYER));

  add_layer(ioscheduler->layer_list,
            allocate_layer(
                build_linear_layer(config->num_features * 3,
                                   config->num_features, config->model_type),
                LINEAR_LAYER));

  add_layer(ioscheduler->layer_list,
            allocate_layer(build_sigmoid_layer(config->num_features * 3,
                                               config->num_features * 3,
                                               config->model_type),
                           SIGMOID_LAYER));

  add_layer(ioscheduler->layer_list,
            allocate_layer(build_linear_layer(config->num_features,
                                              config->num_features * 3,
                                              config->model_type),
                           LINEAR_LAYER));

  ioscheduler->sgd = build_sgd_optimizer(config->learning_rate, config->momentum,
                                       ioscheduler->layer_list, ioscheduler->loss);

  init_multithreading_execution(&(ioscheduler->multithreading),
                                config->batch_size, config->num_features);
  create_async_thread(&(ioscheduler->multithreading), &(ioscheduler->data),
                      ioscheduler_class_net_train_inference, ioscheduler);

  return ioscheduler;
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(build_ioscheduler_class_net);
#endif

void reset_ioscheduler_class_net(ioscheduler_class_net *ioscheduler) {
  layer *current_layer;

  reset_updates(ioscheduler->sgd->update_list);
  ioscheduler->sgd->current_loss.f = 0;
  ioscheduler->sgd->prev_loss.f = 0;
  kml_atomic_bool_init(&(ioscheduler->state.is_training), true);
  kml_atomic_int_init(&(ioscheduler->state.num_accurate_predictions), 0);
  traverse_layers_forward(ioscheduler->layer_list, current_layer) {
    switch (current_layer->type) {
      case LINEAR_LAYER: {
        reset_linear_layer((linear_layer *)current_layer->internal);
        break;
      }
      default:
        break;
    }
  }
}

void clean_ioscheduler_class_net(ioscheduler_class_net *ioscheduler) {
  layer *current_layer;

  free_matrix(ioscheduler->data.collect_input);
  free_matrix(ioscheduler->data.collect_output);
  free_matrix(ioscheduler->online_data);
  free_matrix(ioscheduler->norm_online_data);
  free_matrix(ioscheduler->norm_data_stat.std_dev);
  free_matrix(ioscheduler->norm_data_stat.average);
  free_matrix(ioscheduler->norm_data_stat.variance);
  free_matrix(ioscheduler->norm_data_stat.last_values);

  traverse_layers_forward(ioscheduler->layer_list, current_layer) {
    switch (current_layer->type) {
      case LINEAR_LAYER: {
        clean_linear_layer((linear_layer *)current_layer->internal);
        break;
      }
      case SIGMOID_LAYER: {
        clean_sigmoid_layer((sigmoid_layer *)current_layer->internal);
      }
      default:
        break;
    }
  }

  clean_multithreading_execution(&(ioscheduler->multithreading));
  cleanup_sgd_optimizer(ioscheduler->sgd);
  delete_layers(ioscheduler->layer_list);
  cross_entropy_loss_functions.cleanup(
      (cross_entropy_loss *)ioscheduler->loss->internal);
  kml_free(ioscheduler->loss);
  kml_free(ioscheduler);
#ifdef USE_INTERNAL_MEMORY_ALLOCATOR
  memory_pool_cleanup();
#endif
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(clean_ioscheduler_class_net);
#endif

matrix *get_normalized_ioscheduler_data(ioscheduler_class_net *ioscheduler)
{
  matrix *normalized_data = NULL;

  ioscheduler_normalized_online_data((ioscheduler_net *)ioscheduler,
                                    false);

  switch (ioscheduler->type) {
    case FLOAT:
      normalized_data = matrix_float_conversion(ioscheduler->norm_online_data);
      break;
    case DOUBLE:
      normalized_data = matrix_double_conversion(ioscheduler->norm_online_data);
      break;
    default:
      kml_assert(false);
      break;
  }

  return normalized_data;
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(get_normalized_ioscheduler_data);
#endif

#ifdef KML_KERNEL
matrix *get_normalized_ioscheduler_data_per_file(
    ioscheduler_class_net *ioscheduler, int current_ioscheduler_val,
    ioscheduler_per_file_data *ioscheduler_per_file_data) {
  matrix *normalized_data = NULL;

  ioscheduler_normalized_online_data_per_file((ioscheduler_net *)ioscheduler,
                                            current_ioscheduler_val, false,
                                            ioscheduler_per_file_data);

  switch (ioscheduler->type) {
    case FLOAT:
      normalized_data =
          matrix_float_conversion(ioscheduler_per_file_data->norm_online_data);
      break;
    case DOUBLE:
      normalized_data =
          matrix_double_conversion(ioscheduler_per_file_data->norm_online_data);
      break;
    default:
      kml_assert(false);
      break;
  }

  return normalized_data;
}
EXPORT_SYMBOL(get_normalized_ioscheduler_data_per_file);
#endif

int predict_ioscheduler_class(ioscheduler_class_net *ioscheduler)
{
  matrix *normalized_data = NULL, *indv_result = NULL;
  int class = 0;

  normalized_data =
      get_normalized_ioscheduler_data(ioscheduler);

  // kml_debug("normalized per-disk data:\n");
  // print_matrix(normalized_data);
  indv_result = ioscheduler_class_net_inference(normalized_data, ioscheduler);
  class = matrix_argmax(indv_result);

  cleanup_autodiff(ioscheduler->layer_list);
  free_matrix(normalized_data);

  return class;
}
#ifdef KML_KERNEL
EXPORT_SYMBOL(predict_ioscheduler_class);
#endif

#ifdef KML_KERNEL
int predict_ioscheduler_class_per_file(
    ioscheduler_class_net *ioscheduler, int current_ioscheduler_val,
    ioscheduler_per_file_data *ioscheduler_per_file_data) {
  matrix *normalized_data = NULL, *indv_result = NULL;
  int class = 0;

  normalized_data = get_normalized_ioscheduler_data_per_file(
      ioscheduler, current_ioscheduler_val, ioscheduler_per_file_data);

  // kml_debug("normalized per-file data:\n");
  // print_matrix(normalized_data);
  indv_result = ioscheduler_class_net_inference(normalized_data, ioscheduler);
  class = matrix_argmax(indv_result);

  cleanup_autodiff(ioscheduler->layer_list);
  free_matrix(normalized_data);

  return class;
}
EXPORT_SYMBOL(predict_ioscheduler_class_per_file);
#endif
