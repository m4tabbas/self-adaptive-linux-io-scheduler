/*
 * Copyright (c) 2019-2021 Ibrahim Umit Akgun
 * Copyright (c) 2019-2021 Erez Zadok
 * Copyright (c) 2019-2021 Stony Brook University
 * Copyright (c) 2019-2021 The Research Foundation of SUNY
 *
 * You can redistribute it and/or modify it under the terms of the Apache
 * License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
 */

#ifndef ioscheduler_H
#define ioscheduler_H

#include <linux/fs.h>
#include <linux/mm_types.h>
#include <linux/types.h>
#include <linux/blkdev.h>
/*
typedef enum ioscheduler_trace_types {
  ADD_TO_PAGE_CACHE = 0,
  FSL_WRITEBACK_DIRTY_PAGE,
  FILEMAP_FSL_READ
} ioscheduler_trace_types;
*/
/*
// ioscheduler-ml tracing for add to page cache
typedef void (*trace_ioscheduler_add_to_page_cache_fptr)(struct page *page);

// ioscheduler-ml tracing for filemap fsl read
typedef void (*trace_ioscheduler_mm_filemap_fsl_read_fptr)(struct page *page);

// ioscheduler-ml tracing for fsl writeback dirty page
typedef void (*trace_ioscheduler_fsl_writeback_dirty_page_fptr)(
    struct page *page, struct inode *inode);

// ioscheduler-ml tracing for tuning device
typedef dev_t (*trace_ioscheduler_get_tuning_device_fptr)(void);

// ioscheduler-ml tracing for ioscheduler value for tuning device
typedef unsigned long (*trace_ioscheduler_get_disk_ra_val_fptr)(void);

// ioscheduler-ml creating per-file data structure
typedef void (*trace_ioscheduler_create_per_file_structure_fptr)(
    unsigned long ino, unsigned int ra_pages);

// ioscheduler-ml getting per-file predicted ioscheduler values
typedef unsigned long (*trace_ioscheduler_get_ra_pages_per_file)(
    unsigned long ino);
*/
typedef void (*trace_ioscheduler_block_rq_insert_fptr)(struct request *rq);

#endif
