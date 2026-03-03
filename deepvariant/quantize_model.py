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
"""INT8 post-training quantization for DeepVariant models via TFLite.

Converts a DeepVariant TensorFlow SavedModel to an INT8-quantized TFLite model
using representative calibration data. INT8 models are ~3-4x smaller and run
~2-3x faster on CPU and edge devices compared to float32.

Usage:
  # Using TFRecord calibration data (recommended for best accuracy):
  python3 -m deepvariant.quantize_model \\
    --model_dir ~/.deepvariant/models/wgs \\
    --examples /path/to/examples.tfrecord.gz \\
    --output ~/.deepvariant/models/wgs/deepvariant_wgs_int8.tflite

  # Using random data for quick conversion (may reduce accuracy slightly):
  python3 -m deepvariant.quantize_model \\
    --model_dir ~/.deepvariant/models/wgs \\
    --output ~/.deepvariant/models/wgs/deepvariant_wgs_int8.tflite \\
    --num_calibration_samples 200

  # Verify output against TF SavedModel:
  python3 -m deepvariant.quantize_model --model_dir ... --verify
"""

import argparse
import os
import sys
import time
from typing import Generator, Optional

import numpy as np


def _load_example_shape(model_dir: str):
  """Loads input shape from example_info.json in the model directory."""
  import json
  import tensorflow as tf

  json_path = os.path.join(model_dir, 'example_info.json')
  if not tf.io.gfile.exists(json_path):
    raise FileNotFoundError(
        f'example_info.json not found in {model_dir}. '
        'This file is required to determine the input shape.'
    )
  with tf.io.gfile.GFile(json_path) as f:
    info = json.load(f)
  shape = info['shape']  # [H, W, C]
  return shape


def _make_tfrecord_representative_dataset(
    examples_path: str,
    input_shape,
    num_samples: int,
    input_name: str,
) -> Generator:
  """Creates a representative dataset generator from TFRecord examples.

  Args:
    examples_path: Path to TFRecord.gz file(s) with encoded pileup examples.
    input_shape: [H, W, C] of pileup images.
    num_samples: Number of calibration samples to yield.
    input_name: Name of the input tensor in the saved model.

  Yields:
    Dict mapping input_name to a float32 array of shape [1, H, W, C].
  """
  import tensorflow as tf

  h, w, c = input_shape

  proto_features = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
  }

  ds = tf.data.TFRecordDataset(examples_path, compression_type='GZIP')
  count = 0
  for record in ds.take(num_samples):
    parsed = tf.io.parse_single_example(record, proto_features)
    image = tf.io.decode_raw(parsed['image/encoded'], tf.uint8)
    image = tf.reshape(image, [h, w, c])
    image = tf.cast(image, tf.float32) / 128.0 - 1.0
    image = tf.expand_dims(image, 0)  # [1, H, W, C]
    yield {input_name: image.numpy()}
    count += 1
  print(f'  Used {count} TFRecord examples for calibration.')


def _make_random_representative_dataset(
    input_shape,
    num_samples: int,
    input_name: str,
) -> Generator:
  """Creates a representative dataset from random data (fallback).

  Args:
    input_shape: [H, W, C] of pileup images.
    num_samples: Number of calibration samples to yield.
    input_name: Name of the input tensor in the saved model.

  Yields:
    Dict mapping input_name to a float32 array of shape [1, H, W, C].
  """
  h, w, c = input_shape
  rng = np.random.default_rng(seed=42)
  for _ in range(num_samples):
    sample = rng.standard_normal((1, h, w, c)).astype(np.float32)
    yield {input_name: sample}


def quantize_to_tflite(
    model_dir: str,
    output_path: str,
    examples_path: Optional[str] = None,
    num_calibration_samples: int = 200,
    verify: bool = False,
) -> None:
  """Convert a DeepVariant SavedModel to INT8 quantized TFLite.

  Args:
    model_dir: Path to the TF SavedModel directory.
    output_path: Destination path for the .tflite file.
    examples_path: Optional path to TFRecord.gz examples for calibration.
      If None, random data is used (slightly less accurate quantization).
    num_calibration_samples: Number of examples to use for INT8 calibration.
    verify: If True, compare TFLite and TF SavedModel outputs on a test input.
  """
  import tensorflow as tf

  saved_model_path = model_dir
  if not os.path.exists(os.path.join(saved_model_path, 'saved_model.pb')):
    print(f'ERROR: No saved_model.pb found in {saved_model_path}')
    sys.exit(1)

  print(f'Loading TF SavedModel from {saved_model_path} ...')
  input_shape = _load_example_shape(model_dir)
  h, w, c = input_shape

  # Determine input tensor name from saved model signature.
  model = tf.saved_model.load(saved_model_path)
  sig = model.signatures['serving_default']
  input_name = list(sig.structured_input_signature[1].keys())[0]
  output_name = list(sig.structured_outputs.keys())[0]
  print(f'  Input: {input_name}  shape=[1, {h}, {w}, {c}]')
  print(f'  Output: {output_name}')

  # Build converter.
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  # Representative dataset for INT8 calibration.
  if examples_path:
    print(f'  Using TFRecord examples for calibration: {examples_path}')

    def rep_dataset():
      yield from _make_tfrecord_representative_dataset(
          examples_path, input_shape, num_calibration_samples, input_name
      )
  else:
    print(
        f'  No examples provided — using random data for calibration '
        f'({num_calibration_samples} samples). For best accuracy, provide '
        '--examples /path/to/examples.tfrecord.gz'
    )

    def rep_dataset():
      yield from _make_random_representative_dataset(
          input_shape, num_calibration_samples, input_name
      )

  converter.representative_dataset = rep_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.float32  # Keep output as float32

  print(f'Converting to INT8 TFLite ({num_calibration_samples} calibration samples) ...')
  t0 = time.time()
  tflite_model = converter.convert()
  elapsed = time.time() - t0
  print(f'  Conversion took {elapsed:.1f}s')

  # Save the quantized model.
  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  with open(output_path, 'wb') as f:
    f.write(tflite_model)
  size_mb = os.path.getsize(output_path) / (1024 * 1024)
  print(f'  Saved: {output_path} ({size_mb:.1f} MB)')

  # Verify: compare TFLite vs TF SavedModel on a test input.
  if verify:
    print('Verifying accuracy (random input, batch=1) ...')
    test_input = np.random.rand(1, h, w, c).astype(np.float32)

    # TF SavedModel prediction.
    tf_output = sig(tf.constant(test_input))
    tf_probs = list(tf_output.values())[0].numpy()

    # TFLite prediction.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # TFLite INT8 inputs need quantization.
    scale, zero_point = input_details[0]['quantization']
    quantized_input = np.clip(
        np.round(test_input / scale + zero_point), -128, 127
    ).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()
    tflite_probs = interpreter.get_tensor(output_details[0]['index'])

    max_diff = np.max(np.abs(tf_probs - tflite_probs))
    print(f'  TF output:     {tf_probs[0]}')
    print(f'  TFLite output: {tflite_probs[0]}')
    print(f'  Max abs diff:  {max_diff:.6f}')

    if max_diff < 0.05:
      print('  PASS: Outputs match within tolerance (< 0.05)')
    else:
      print(f'  WARNING: Max diff {max_diff:.6f} exceeds 0.05 tolerance')
      print('  The model may still produce acceptable variant calls.')

  print('Done.')


def main():
  parser = argparse.ArgumentParser(
      description='Convert DeepVariant TF SavedModel to INT8 TFLite'
  )
  dv_home = os.environ.get('DEEPVARIANT_HOME', os.path.expanduser('~/.deepvariant'))
  default_model_dir = os.path.join(dv_home, 'models', 'wgs')

  parser.add_argument(
      '--model_dir',
      default=default_model_dir,
      help='Path to TF SavedModel directory (default: %(default)s)',
  )
  parser.add_argument(
      '--output',
      default=None,
      help=(
          'Output .tflite path '
          '(default: {model_dir}/deepvariant_wgs_int8.tflite)'
      ),
  )
  parser.add_argument(
      '--examples',
      default=None,
      help=(
          'Path to TFRecord.gz examples for INT8 calibration. '
          'If not provided, random data is used.'
      ),
  )
  parser.add_argument(
      '--num_calibration_samples',
      type=int,
      default=200,
      help='Number of examples to use for INT8 calibration (default: 200)',
  )
  parser.add_argument(
      '--verify',
      action='store_true',
      help='Verify TFLite output matches TF output on random input',
  )

  args = parser.parse_args()

  output_path = args.output
  if output_path is None:
    output_path = os.path.join(args.model_dir, 'deepvariant_wgs_int8.tflite')

  quantize_to_tflite(
      model_dir=args.model_dir,
      output_path=output_path,
      examples_path=args.examples,
      num_calibration_samples=args.num_calibration_samples,
      verify=args.verify,
  )


if __name__ == '__main__':
  main()
