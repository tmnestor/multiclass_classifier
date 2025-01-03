# CPU-Optimized Multiclass Classifier

A PyTorch-based multiclass classifier optimized for CPU training with advanced performance features and memory efficiency.

## Features

- **CPU Optimizations**
  - MKL-DNN/oneDNN acceleration support
  - Automatic mixed precision with bfloat16
  - Optimized thread configuration
  - Memory-efficient dataset handling

- **Training Features**
  - OneCycleLR learning rate scheduling with warmup
  - Automated hyperparameter tuning using Optuna
  - Early stopping with configurable patience
  - Comprehensive logging and metrics tracking
  - Memory-efficient dataset loading for large datasets

- **Monitoring & Debugging**
  - Learning curves visualization
  - CPU optimization status logging
  - Detailed training progress tracking
  - Performance profiling support

## Requirements

```bash
pip install torch pandas numpy tqdm optuna pyyaml psutil seaborn matplotlib scikit-learn
```

## Configuration

The system is configured through `config.yaml`. Key configuration sections:

### CPU Optimization Settings
```yaml
cpu_optimization:
  enable_mkldnn: true
  num_threads: "auto"
  use_bfloat16: true
```

### DataLoader Settings
```yaml
dataloader:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

### Performance Settings
```yaml
performance:
  enable_mkldnn: true
  enable_mkl: true
  mixed_precision: true
  batch_size_multiplier: 2
  grad_accum_steps: 4
```

## Usage

1. Prepare your data in CSV format with features and a target column.
2. Update the data paths in `config.yaml`:
   ```yaml
   data:
     train_path: "data/train.csv"
     val_path: "data/val.csv"
     target_column: "target"
   ```
3. Run the training:
   ```bash
   python MLP_trainer.py
   ```

## Memory Efficiency

The system includes automatic memory management:
- Switches to memory-mapped datasets for large data
- Configurable batch sizes and prefetch factors
- Automatic cleanup of temporary files
- Efficient CPU memory utilization

## Logging

Logs are organized in the `logs` directory:
- `training.log`: Training progress and metrics
- `cpu_optimization.log`: CPU optimization status
- `hyperparameter_tuning.log`: Tuning progress and results

## Model Checkpointing

Best models are automatically saved to the path specified in config:
```yaml
model:
  save_path: "checkpoints/best_model.pt"
```

## Performance Tips

1. Adjust `batch_size` and `batch_size_multiplier` based on your CPU memory
2. Set `num_workers` to 2x your CPU cores
3. Enable `persistent_workers` for faster data loading
4. Use `mixed_precision` if your CPU supports bfloat16

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - See LICENSE file for details
