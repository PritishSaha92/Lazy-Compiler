# Loop and floating-remainder regressions

The final runner roster uses all six packages in `loops`, the two FP32 packages
in `remainder`, and the two FP64 packages in `remainder_words`.

The original `remainder/floating_f64_mod` and
`remainder/floating_f64_remainder` packages are retained to explain the initial
failed run. Their full-width I64 bit payloads lose low bits through the runner's
binary64 number conversion. Use the `remainder_words` replacements for runner
validation. They transport the same IEEE values and results as exact I32 pairs;
the reference and zero tolerance have not been relaxed.

Regenerate current packages from the repository root into new directories:

```console
python tests/test_loop_bindings.py --export build/new-loop-cases
python tests/test_floating_remainder.py --export build/new-remainder-cases
```

See the [phase-8 report](../../docs/PHASE8.md) for the original failures,
independent oracles, final results and precise semantics.
