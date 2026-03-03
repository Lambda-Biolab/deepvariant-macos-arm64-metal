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
"""Inference benchmarking and profiling framework for DeepVariant.

Provides structured per-stage timing, throughput, and peak memory measurements
for the call_variants stage. Output is a JSON report for easy comparison
across configurations (batch sizes, model backends, hardware).

Usage:
  python3 -m deepvariant.benchmark \\
    --examples /path/to/examples.tfrecord.gz \\
    --checkpoint ~/.deepvariant/models/wgs \\
    --output benchmark_report.json

  # Compare two configurations side-by-side:
  python3 -m deepvariant.benchmark \\
    --examples /path/to/examples.tfrecord.gz \\
    --checkpoint ~/.deepvariant/models/wgs \\
    --compare_configs "batch_size=128,use_coreml=true:batch_size=64,use_coreml=false"

  # Profile sub-stages: loading, parsing, inference, writing.
  python3 -m deepvariant.benchmark --examples ... --checkpoint ... --profile
"""

import argparse
import contextlib
import json
import os
import platform
import resource
import sys
import time
from typing import Any, Dict, Generator, List, Optional


def _peak_memory_mb() -> float:
  """Returns current peak RSS in MB (cross-platform)."""
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  # macOS reports in bytes; Linux reports in kilobytes.
  if platform.system() == 'Darwin':
    return usage / (1024 * 1024)
  else:
    return usage / 1024


@contextlib.contextmanager
def timed_stage(name: str, results: Dict[str, Any]) -> Generator:
  """Context manager measuring wall time and peak memory for a named stage.

  Args:
    name: Stage name used as the key in results.
    results: Dict that receives timing results under key `name`.

  Yields:
    Nothing (use as `with timed_stage('stage', r): ...`).
  """
  start_time = time.perf_counter()
  start_mem_mb = _peak_memory_mb()
  try:
    yield
  finally:
    elapsed = time.perf_counter() - start_time
    end_mem_mb = _peak_memory_mb()
    results[name] = {
        'wall_time_s': round(elapsed, 4),
        'peak_memory_delta_mb': round(max(end_mem_mb - start_mem_mb, 0), 1),
    }


def _get_system_info() -> Dict[str, Any]:
  """Returns system information for the benchmark report."""
  info = {
      'platform': platform.platform(),
      'processor': platform.processor(),
      'python_version': platform.python_version(),
  }
  try:
    import subprocess
    # macOS: get chip name
    result = subprocess.run(
        ['sysctl', '-n', 'machdep.cpu.brand_string'],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode == 0:
      info['cpu'] = result.stdout.strip()
  except Exception:  # pylint: disable=broad-except
    pass
  try:
    import tensorflow as tf
    info['tensorflow_version'] = tf.__version__
    gpus = tf.config.list_physical_devices('GPU')
    info['gpu_devices'] = [g.name for g in gpus]
  except Exception:  # pylint: disable=broad-except
    pass
  return info


def benchmark_call_variants(
    examples_path: str,
    checkpoint_path: str,
    batch_size: int = 1024,
    use_coreml: bool = False,
    use_tflite: bool = False,
    tflite_model_path: str = '',
    num_batches: int = 0,
    profile: bool = False,
) -> Dict[str, Any]:
  """Benchmark call_variants sub-stages.

  Measures time and peak memory for:
    - model_load: Loading the model from checkpoint/saved_model/CoreML/TFLite
    - dataset_init: Creating the TFRecord dataset pipeline
    - inference: Running model forward passes across all batches
    - per_batch: Per-batch breakdown (if profile=True)

  Args:
    examples_path: Path to TFRecord.gz examples (output of make_examples).
    checkpoint_path: Path to TF checkpoint or SavedModel directory.
    batch_size: Inference batch size.
    use_coreml: If True, use CoreML model (Apple Silicon only).
    use_tflite: If True, use TFLite INT8 model.
    tflite_model_path: Path to .tflite file (if use_tflite=True).
    num_batches: If > 0, limit inference to this many batches.
    profile: If True, record per-batch timing.

  Returns:
    Dict with benchmark results suitable for JSON serialization.
  """
  import numpy as np
  import tensorflow as tf

  results: Dict[str, Any] = {
      'config': {
          'examples_path': examples_path,
          'checkpoint_path': checkpoint_path,
          'batch_size': batch_size,
          'use_coreml': use_coreml,
          'use_tflite': use_tflite,
          'num_batches_limit': num_batches,
      },
      'system': _get_system_info(),
      'stages': {},
  }

  # ── Stage 1: Model loading ──────────────────────────────────────────────────
  model = None
  coreml_model = None
  tflite_interpreter = None
  input_details = None
  output_details = None
  example_shape = None

  with timed_stage('model_load', results['stages']):
    if use_coreml:
      import coremltools as ct
      coreml_model = ct.models.MLModel(
          tflite_model_path if tflite_model_path else
          os.path.join(checkpoint_path, 'deepvariant_wgs.mlmodel'),
          compute_units=ct.ComputeUnit.ALL,
      )
      spec = coreml_model.get_spec()
      coreml_input_shape = list(
          spec.description.input[0].type.multiArrayType.shape
      )
      example_shape = coreml_input_shape[1:]  # drop batch dim

    elif use_tflite:
      tflite_path = tflite_model_path or os.path.join(
          checkpoint_path, 'deepvariant_wgs_int8.tflite'
      )
      tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
      tflite_interpreter.allocate_tensors()
      input_details = tflite_interpreter.get_input_details()
      output_details = tflite_interpreter.get_output_details()
      # Derive shape from interpreter (convert to Python int for JSON safety).
      shape = input_details[0]['shape']  # [1, H, W, C]
      example_shape = [int(x) for x in shape[1:]]

    else:
      use_saved_model = tf.io.gfile.exists(
          os.path.join(checkpoint_path, 'saved_model.pb')
      )
      if use_saved_model:
        model = tf.saved_model.load(checkpoint_path)
        sig = model.signatures['serving_default']
        input_spec = list(sig.structured_input_signature[1].values())[0]
        example_shape = list(input_spec.shape[1:])  # drop batch dim
      else:
        from deepvariant import dv_utils, keras_modeling
        json_path = os.path.join(checkpoint_path, 'example_info.json')
        import json as jsonlib
        with tf.io.gfile.GFile(json_path) as f:
          info = jsonlib.load(f)
        example_shape = info['shape']
        model = keras_modeling.inceptionv3(
            example_shape, init_backbone_with_imagenet=False
        )
        model.load_weights(checkpoint_path).expect_partial()

  results['stages']['model_load']['model_type'] = (
      'coreml' if use_coreml else 'tflite' if use_tflite else
      'saved_model' if not use_tflite and not use_coreml else 'checkpoint'
  )
  results['example_shape'] = example_shape

  # ── Stage 2: Dataset initialization ────────────────────────────────────────
  with timed_stage('dataset_init', results['stages']):
    from third_party.nucleus.io import sharded_file_utils

    h, w, c = example_shape

    proto_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'variant/encoded': tf.io.FixedLenFeature((), tf.string),
        'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
    }

    def _parse(example):
      parsed = tf.io.parse_single_example(example, proto_features)
      image = tf.io.decode_raw(parsed['image/encoded'], tf.uint8)
      image = tf.reshape(image, example_shape)
      image = tf.cast(image, tf.float32) / 128.0 - 1.0
      return image, parsed['variant/encoded'], parsed['alt_allele_indices/encoded']

    ds = tf.data.TFRecordDataset.list_files(
        sharded_file_utils.normalize_to_sharded_file_pattern(examples_path),
        shuffle=False,
    )

    def _load(fname):
      return tf.data.TFRecordDataset(fname, compression_type='GZIP',
                                     buffer_size=16 * 1000 * 1000)

    ds = ds.interleave(_load, cycle_length=8, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if num_batches > 0:
      ds = ds.take(num_batches)

  # ── Stage 3: Inference ──────────────────────────────────────────────────────
  batch_times: List[float] = []
  total_examples = 0
  is_gpu = bool(tf.config.list_physical_devices('GPU'))

  with timed_stage('inference', results['stages']):
    for images, _variants, _alt_alleles in ds:
      t_batch = time.perf_counter()

      if coreml_model is not None:
        _preds = coreml_model.predict({'input_1': images.numpy()})['Identity']
      elif tflite_interpreter is not None:
        # INT8 inference: quantize input, run, dequantize output.
        scale, zero_point = input_details[0]['quantization']
        q_input = np.clip(
            np.round(images.numpy() / scale + zero_point), -128, 127
        ).astype(np.int8)
        # Resize interpreter for current batch if needed.
        tflite_interpreter.resize_tensor_input(
            input_details[0]['index'], list(q_input.shape)
        )
        tflite_interpreter.allocate_tensors()
        tflite_interpreter.set_tensor(input_details[0]['index'], q_input)
        tflite_interpreter.invoke()
        _preds = tflite_interpreter.get_tensor(output_details[0]['index'])
      elif use_saved_model:
        _preds = sig(images)['classification'].numpy()
      else:
        if is_gpu:
          _preds = model(images, training=False).numpy()
        else:
          _preds = model.predict_on_batch(images)

      batch_elapsed = time.perf_counter() - t_batch
      batch_size_actual = images.shape[0]
      total_examples += batch_size_actual

      if profile:
        batch_times.append({
            'batch_size': int(batch_size_actual),
            'wall_time_s': round(batch_elapsed, 4),
            'examples_per_sec': round(batch_size_actual / batch_elapsed, 1),
        })

  # ── Compute derived metrics ─────────────────────────────────────────────────
  inference_time = results['stages']['inference']['wall_time_s']
  results['summary'] = {
      'total_examples': total_examples,
      'inference_wall_time_s': inference_time,
      'examples_per_second': round(total_examples / max(inference_time, 1e-6), 1),
      'ms_per_example': round(1000.0 * inference_time / max(total_examples, 1), 3),
  }

  if profile and batch_times:
    results['per_batch'] = batch_times
    per_example_times = [
        b['wall_time_s'] / b['batch_size'] for b in batch_times
    ]
    results['summary']['per_batch_p50_ms'] = round(
        1000.0 * float(np.percentile(per_example_times, 50)), 3
    )
    results['summary']['per_batch_p95_ms'] = round(
        1000.0 * float(np.percentile(per_example_times, 95)), 3
    )

  return results


def main():
  parser = argparse.ArgumentParser(
      description='Benchmark DeepVariant call_variants sub-stages'
  )
  dv_home = os.environ.get('DEEPVARIANT_HOME', os.path.expanduser('~/.deepvariant'))

  parser.add_argument(
      '--examples', required=True,
      help='Path to make_examples TFRecord.gz output (supports shards: @N)',
  )
  parser.add_argument(
      '--checkpoint',
      default=os.path.join(dv_home, 'models', 'wgs'),
      help='Path to TF checkpoint or SavedModel directory (default: %(default)s)',
  )
  parser.add_argument(
      '--batch_size', type=int, default=1024,
      help='Inference batch size (default: 1024)',
  )
  parser.add_argument(
      '--use_coreml', action='store_true',
      help='Use CoreML model (Apple Silicon only)',
  )
  parser.add_argument(
      '--use_tflite', action='store_true',
      help='Use INT8 TFLite model',
  )
  parser.add_argument(
      '--tflite_model', default='',
      help='Path to .tflite model file (default: {checkpoint}/deepvariant_wgs_int8.tflite)',
  )
  parser.add_argument(
      '--num_batches', type=int, default=0,
      help='Limit inference to this many batches (0 = all)',
  )
  parser.add_argument(
      '--profile', action='store_true',
      help='Record per-batch timing breakdown',
  )
  parser.add_argument(
      '--output', default=None,
      help='JSON file to write results to (default: print to stdout)',
  )

  args = parser.parse_args()

  results = benchmark_call_variants(
      examples_path=args.examples,
      checkpoint_path=args.checkpoint,
      batch_size=args.batch_size,
      use_coreml=args.use_coreml,
      use_tflite=args.use_tflite,
      tflite_model_path=args.tflite_model,
      num_batches=args.num_batches,
      profile=args.profile,
  )

  report_json = json.dumps(results, indent=2)

  if args.output:
    with open(args.output, 'w') as f:
      f.write(report_json)
    print(f'Benchmark report written to: {args.output}')
  else:
    print(report_json)

  # Print human-readable summary.
  s = results['summary']
  print(
      f'\n=== Benchmark Summary ===\n'
      f'  Total examples:   {s["total_examples"]}\n'
      f'  Inference time:   {s["inference_wall_time_s"]:.1f}s\n'
      f'  Throughput:       {s["examples_per_second"]:.0f} examples/sec\n'
      f'  Latency:          {s["ms_per_example"]:.2f} ms/example\n'
      f'  Model load time:  {results["stages"]["model_load"]["wall_time_s"]:.1f}s\n'
  )


if __name__ == '__main__':
  main()
