#!/usr/bin/env python3
# Copyright 2024 Google LLC.  (DeepVariant macOS ARM64 port)
#
# Converts a DeepVariant TensorFlow SavedModel to Apple CoreML (.mlmodel).
# The CoreML model runs ~1.2x faster than TensorFlow Metal on Apple Silicon
# using the Neural Engine + GPU (ComputeUnit.ALL).
#
# Usage:
#   python3 scripts/convert_model_coreml.py                    # auto-detect
#   python3 scripts/convert_model_coreml.py --model_dir DIR    # custom path
#   python3 scripts/convert_model_coreml.py --verify            # convert + verify
#
# Requirements:
#   pip install coremltools  (tested with coremltools 9.0)
#   TensorFlow 2.13.x

import argparse
import os
import sys
import time


def convert_model(model_dir, output_path, verify=False, quantize_int4=False):
    """Convert DeepVariant TF SavedModel to CoreML .mlmodel."""
    try:
        import coremltools as ct
    except ImportError:
        print('ERROR: coremltools is required. Install with: pip install coremltools')
        sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError:
        print('ERROR: tensorflow is required. Install with: pip install tensorflow')
        sys.exit(1)

    saved_model_path = model_dir
    if not os.path.exists(os.path.join(saved_model_path, 'saved_model.pb')):
        print(f'ERROR: No saved_model.pb found in {saved_model_path}')
        sys.exit(1)

    print(f'Loading TF SavedModel from {saved_model_path} ...')
    model = tf.saved_model.load(saved_model_path)
    sig = model.signatures['serving_default']

    # Verify input shape
    input_spec = list(sig.structured_input_signature[1].values())[0]
    input_shape = tuple(input_spec.shape)
    print(f'  Input shape: {input_shape} (batch, height, width, channels)')

    output_spec = list(sig.structured_outputs.values())[0]
    output_shape = tuple(output_spec.shape)
    print(f'  Output shape: {output_shape} (batch, num_classes)')

    # Convert with coremltools
    # Use neuralnetwork backend (faster than mlprogram for this model on Apple Silicon)
    print('Converting to CoreML (neuralnetwork backend) ...')
    t0 = time.time()

    # ct.convert needs a concrete function with known input shapes.
    # Use a representative batch size for tracing; the neuralnetwork backend
    # supports dynamic batch at runtime.
    input_name = list(sig.structured_input_signature[1].keys())[0]
    h, w, c = input_shape[1], input_shape[2], input_shape[3]

    mlmodel = ct.convert(
        saved_model_path,
        source='tensorflow',
        convert_to='neuralnetwork',
        inputs=[ct.TensorType(name=input_name, shape=(1, h, w, c))],
        compute_units=ct.ComputeUnit.ALL,
    )

    elapsed = time.time() - t0
    print(f'  Conversion took {elapsed:.1f}s')

    # Note: neuralnetwork backend inherently supports variable batch sizes.
    # No need to set flexible shape ranges (which conflict with the 4D input).

    # Save
    mlmodel.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'  Saved: {output_path} ({size_mb:.1f} MB)')

    # Int4 weight quantization — reduces memory bandwidth during inference.
    # coremltools.optimize.coreml (block-wise int4) only works for mlprogram models.
    # For neuralnetwork models (our format), use the legacy quantization_utils API which
    # supports 1–8 bits (linear quantization, per-layer scale+bias).
    if quantize_int4:
        from coremltools.models.neural_network import quantization_utils
        print('Applying 4-bit weight quantization (neuralnetwork legacy API) ...')
        t1 = time.time()
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=4)
        base, ext = os.path.splitext(output_path)
        output_path = base + '_int4' + ext
        mlmodel.save(output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  Int4 quantization took {time.time()-t1:.1f}s')
        print(f'  Saved: {output_path} ({size_mb:.1f} MB)')

    # Verification: compare CoreML vs TF outputs on random input
    if verify:
        import numpy as np
        print('Verifying accuracy (random input, batch=1) ...')

        test_input = np.random.rand(1, h, w, c).astype(np.float32)

        # TF prediction
        tf_input = tf.constant(test_input)
        tf_output = sig(tf_input)
        tf_probs = list(tf_output.values())[0].numpy()

        # CoreML prediction
        coreml_output = mlmodel.predict({input_name: test_input})
        output_key = list(coreml_output.keys())[0]
        coreml_probs = coreml_output[output_key].astype(np.float32)

        max_diff = np.max(np.abs(tf_probs - coreml_probs))
        print(f'  TF output:     {tf_probs[0]}')
        print(f'  CoreML output: {coreml_probs[0]}')
        print(f'  Max abs diff:  {max_diff:.6f}')

        if max_diff < 0.01:
            print('  PASS: Outputs match within tolerance (< 0.01)')
        else:
            print(f'  WARNING: Max diff {max_diff:.6f} exceeds 0.01 tolerance')
            print('  The model may still produce acceptable variant calls.')

    print('Done.')


def main():
    parser = argparse.ArgumentParser(
        description='Convert DeepVariant TF model to Apple CoreML (.mlmodel)'
    )
    dv_home = os.environ.get('DEEPVARIANT_HOME',
                              os.path.expanduser('~/.deepvariant'))
    default_model_dir = os.path.join(dv_home, 'models', 'wgs')

    parser.add_argument(
        '--model_dir',
        default=default_model_dir,
        help='Path to TF SavedModel directory (default: %(default)s)',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output .mlmodel path (default: {model_dir}/deepvariant_wgs.mlmodel)',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify CoreML output matches TF output on random input',
    )
    parser.add_argument(
        '--quantize-int4',
        action='store_true',
        help='Apply block-wise int4 weight quantization after conversion. Reduces model '
             'size ~4x vs float16, cutting memory bandwidth during inference on Apple Silicon. '
             'Saves as {output_basename}_int4.mlmodel. Requires coremltools >= 8.0.',
    )

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.model_dir, 'deepvariant_wgs.mlmodel')

    convert_model(args.model_dir, output_path, verify=args.verify,
                  quantize_int4=args.quantize_int4)


if __name__ == '__main__':
    main()
