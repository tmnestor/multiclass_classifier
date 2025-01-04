# MLP Classifier with Enhanced CPU Performance

A robust multilayer perceptron (MLP) classifier implementation with advanced CPU optimizations, hyperparameter tuning, and comprehensive validation features.

## Features

### CPU Optimizations
- Automatic MKL-DNN and MKL optimizations detection and enablement
- Mixed precision training support (bfloat16 where available)
- Optimized thread management
- Memory-efficient data loading with memory mapping
- Performance monitoring and profiling

### Training & Validation
- Hyperparameter tuning with Optuna
- Learning rate scheduling with warmup
- Cross-validation support
- Early stopping with configurable patience
- Checkpoint management
- Comprehensive model validation including:
  - Confidence analysis
  - Confusion matrix visualization
  - Detailed classification metrics
  - Per-class performance analysis

### Data Management
- Memory-efficient dataset handling for large datasets
- Automatic memory threshold detection
- Optimized data loading with prefetching
- Support for persistent workers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Configure your model in `config.yaml`
2. Prepare your data in CSV format
3. Run the trainer:

```bash
python MLP_trainer.py
```

## Configuration

Key configuration sections in `config.yaml`:

```yaml
training:
  dataloader:
    num_workers: 8  # Adjust based on CPU cores
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2

  cpu_optimization:
    enable_mkldnn: true
    num_threads: "auto"
    use_bfloat16: true

  performance:
    enable_mkl: true
    mixed_precision: true
    batch_size_multiplier: 2
```

## Validation Metrics

The model provides comprehensive validation metrics:

- Overall accuracy and F1-score
- Per-class precision, recall, and F1-score
- Confidence analysis for correct/incorrect predictions
- Visual confusion matrix
- Cross-validation results with standard deviation

## Performance Monitoring

- CPU optimization status logging
- Training progress visualization
- Resource utilization tracking
- Detailed logging with configurable verbosity

## Examples

See `examples/` directory for usage examples and notebook demonstrations.

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{mlp_classifier,
  title={MLP Classifier with Enhanced CPU Performance},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multiclass_classifier}
}
