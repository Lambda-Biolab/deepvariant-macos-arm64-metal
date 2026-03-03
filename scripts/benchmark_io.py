#!/usr/bin/env python3
# Copyright 2024 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
"""Benchmark TFRecord dataset pipeline throughput for different cycle_length values.

Measures how fast the data pipeline can supply pileup image batches, isolated
from GPU/CoreML inference. Answers: "Is _DEFAULT_INPUT_READ_THREADS=32 optimal
for 8-shard input, or does over-provisioning threads waste CPU?"

Usage:
  source ~/.deepvariant/venv/bin/activate
  python3 scripts/benchmark_io.py \\
    --examples ~/deepvariant-benchmark/mixed-precision/runs/run_1/make_examples.tfrecord@8
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from third_party.nucleus.io import sharded_file_utils

# CoreML inference speed to compare against (ex/s, from prior benchmark).
COREML_THROUGHPUT = 998.0
METAL_GPU_THROUGHPUT = 1392.0

EXAMPLE_SHAPE = [100, 221, 7]  # WGS model input shape
PREFETCH_BUFFER_BYTES = 16 * 1000 * 1000  # same as _DEFAULT_PREFETCH_BUFFER_BYTES
WARMUP_BATCHES = 5
MEASURE_BATCHES = 60


def _build_dataset(file_pattern: str, batch_size: int, cycle_length: int):
  """Build TFRecord dataset matching call_variants.get_dataset() logic."""
  proto_features = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
  }

  # Expand shard pattern using nucleus sharded_file_utils (handles @N notation).
  glob_pattern = sharded_file_utils.normalize_to_sharded_file_pattern(
      file_pattern
  )
  paths = tf.io.gfile.glob(glob_pattern)
  # DeepVariant writes .tfrecord-NNNNN-of-MMMMM.gz; try adding .gz if needed.
  if not paths:
    paths = tf.io.gfile.glob(glob_pattern + '.gz')
  if not paths:
    raise FileNotFoundError(
        f'No files found matching: {file_pattern}  (glob: {glob_pattern}[.gz])'
    )

  h, w, c = EXAMPLE_SHAPE

  def _parse(record):
    parsed = tf.io.parse_single_example(record, proto_features)
    image = tf.io.decode_raw(parsed['image/encoded'], tf.uint8)
    image = tf.reshape(image, EXAMPLE_SHAPE)
    return tf.cast(image, tf.float32) / 128.0 - 1.0

  ds = tf.data.Dataset.from_tensor_slices(paths)

  def _load(fname):
    return tf.data.TFRecordDataset(
        fname,
        compression_type='GZIP',
        buffer_size=PREFETCH_BUFFER_BYTES,
    )

  ds = ds.interleave(
      _load,
      cycle_length=cycle_length,
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return ds


def benchmark_cycle_length(
    examples: str,
    batch_size: int,
    cycle_length: int,
) -> dict:
  """Measure dataset throughput for one cycle_length value."""
  ds = _build_dataset(examples, batch_size, cycle_length)

  # Warmup: iterate WARMUP_BATCHES to prime prefetch buffers.
  ds_iter = iter(ds)
  for _ in range(WARMUP_BATCHES):
    try:
      _ = next(ds_iter)
    except StopIteration:
      ds_iter = iter(ds)

  # Measure MEASURE_BATCHES.
  total_examples = 0
  t_start = time.perf_counter()
  for _ in range(MEASURE_BATCHES):
    try:
      batch = next(ds_iter)
    except StopIteration:
      ds_iter = iter(ds)
      batch = next(ds_iter)
    total_examples += batch.shape[0]
  elapsed = time.perf_counter() - t_start

  return {
      'cycle_length': cycle_length,
      'total_examples': total_examples,
      'wall_time_s': round(elapsed, 3),
      'examples_per_sec': round(total_examples / elapsed, 1),
  }


def main():
  parser = argparse.ArgumentParser(
      description='Benchmark TFRecord pipeline throughput vs cycle_length'
  )
  parser.add_argument(
      '--examples', required=True,
      help='Path to make_examples TFRecord shards (supports @N notation)',
  )
  parser.add_argument(
      '--batch_size', type=int, default=128,
      help='Batch size (default: 128, matching CoreML optimum)',
  )
  args = parser.parse_args()

  examples = os.path.expanduser(args.examples)

  print(f'Dataset pipeline throughput benchmark')
  print(f'  examples:   {examples}')
  print(f'  batch_size: {args.batch_size}')
  print(f'  warmup:     {WARMUP_BATCHES} batches, measure: {MEASURE_BATCHES} batches')
  print(f'  CoreML inference ceiling: {COREML_THROUGHPUT:.0f} ex/s')
  print(f'  Metal GPU inference ceiling: {METAL_GPU_THROUGHPUT:.0f} ex/s')
  print()
  print(f'{"cycle_length":>14}  {"ex/s":>8}  {"time_s":>7}  {"vs CoreML":>11}  note')
  print('-' * 65)

  results = []
  for cycle_length in [1, 4, 8, 16, 32]:
    r = benchmark_cycle_length(examples, args.batch_size, cycle_length)
    results.append(r)
    vs_coreml = r['examples_per_sec'] / COREML_THROUGHPUT
    note = ''
    if r['examples_per_sec'] < COREML_THROUGHPUT:
      note = '<-- I/O BOTTLENECK vs CoreML'
    elif r['examples_per_sec'] < METAL_GPU_THROUGHPUT:
      note = '<-- I/O BOTTLENECK vs Metal GPU'
    print(
        f'{cycle_length:>14}  {r["examples_per_sec"]:>8.0f}  '
        f'{r["wall_time_s"]:>7.2f}  {vs_coreml:>10.2f}x  {note}'
    )

  # Summary.
  best = max(results, key=lambda r: r['examples_per_sec'])
  default = next(r for r in results if r['cycle_length'] == 32)
  print()
  print(f'Best cycle_length: {best["cycle_length"]} ({best["examples_per_sec"]:.0f} ex/s)')
  print(f'Default (32):      {default["examples_per_sec"]:.0f} ex/s')
  if best['cycle_length'] != 32:
    gain = (best['examples_per_sec'] - default['examples_per_sec']) / default['examples_per_sec'] * 100
    print(f'Potential pipeline gain from lowering cycle_length: {gain:+.1f}%')
    if gain > 5:
      print('  --> Worth testing end-to-end with benchmark.sh')
    else:
      print('  --> Gain within noise, no action needed')
  else:
    print('Default (32) is already optimal.')

  all_above_coreml = all(r['examples_per_sec'] > COREML_THROUGHPUT for r in results)
  all_above_metal = all(r['examples_per_sec'] > METAL_GPU_THROUGHPUT for r in results)
  print()
  if all_above_metal:
    print('CONCLUSION: I/O pipeline always exceeds Metal GPU throughput.')
    print('  _DEFAULT_INPUT_READ_THREADS is NOT a bottleneck at any tested value.')
    print('  No changes needed to call_variants.py.')
  elif all_above_coreml:
    print('CONCLUSION: I/O pipeline always exceeds CoreML throughput.')
    print('  For the current production path (CoreML), I/O is not a bottleneck.')
  else:
    print('CONCLUSION: I/O can be a bottleneck at low cycle_length values.')
    print('  Consider testing with benchmark.sh at the optimal cycle_length.')


if __name__ == '__main__':
  main()
