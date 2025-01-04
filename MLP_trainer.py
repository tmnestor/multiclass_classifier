import random
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import warnings
from typing import Dict, Optional, List
import platform

import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.backends.mkldnn
import torch.backends.mkl
import multiprocessing
from torch.utils.data import DataLoader, Dataset
import psutil
import optuna
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from torch.optim import Optimizer
from typing import Optional, Dict, Any

import contextlib
import tempfile
import atexit

class CPUOptimizationManager:
    _instance = None
    _initialized = False
    _mixed_precision_logged = False  # Add flag for logging
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CPUOptimizationManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger("CPUOptimizer")
            self.enabled_features = []
            self.disabled_features = []
            self._log_initial_status()
            self._initialized = True
    
    def _log_initial_status(self):
        """Log CPU optimization status once at initialization"""
        self.logger.info("CPU Optimization Status:")
        self.logger.info(f"CPU Architecture: {platform.processor()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        # Try enabling MKL-DNN
        try:
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                self.enabled_features.append('MKL-DNN')
            else:
                self.disabled_features.append('MKL-DNN')
        except:
            self.disabled_features.append('MKL-DNN')
            
        # Try enabling MKL
        try:
            if hasattr(torch.backends, 'mkl'):
                torch.backends.mkl.enabled = True
                self.enabled_features.append('MKL')
            else:
                self.disabled_features.append('MKL')
        except:
            self.disabled_features.append('MKL')
            
        # Configure threading
        try:
            num_threads = multiprocessing.cpu_count()
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            self.enabled_features.append(f'Multi-threading ({num_threads} threads)')
        except:
            self.disabled_features.append('Custom thread configuration')
            
        # Check bfloat16 support
        try:
            if hasattr(torch, 'bfloat16'):
                self.enabled_features.append('Mixed Precision (bfloat16)')
                if not self._mixed_precision_logged:
                    self.logger.info("Using mixed precision training with bfloat16")
                    self._mixed_precision_logged = True
            else:
                self.disabled_features.append('Mixed Precision')
        except:
            self.disabled_features.append('Mixed Precision')

        # Log enabled features
        if self.enabled_features:
            self.logger.info("Enabled optimizations:")
            for feature in self.enabled_features:
                self.logger.info(f"  - {feature}")
        
        # Log disabled features
        if self.disabled_features:
            self.logger.info("Disabled/Unavailable optimizations:")
            for feature in self.disabled_features:
                self.logger.info(f"  - {feature}")

    def supports_mixed_precision(self):
        return 'Mixed Precision (bfloat16)' in self.enabled_features

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
# Set up logging
def setup_logger(name='MLPTrainer', config=None):
    """Enhanced logger setup with separate console and file handlers"""
    if config is None:
        log_dir = "logs"
        console_level = "WARNING"
        file_level = "INFO"
    else:
        log_dir = config.get('logging', {}).get('directory', 'logs')
        console_level = config.get('logging', {}).get('console_level', 'WARNING')
        file_level = config.get('logging', {}).get('file_level', 'INFO')

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # File handler with detailed formatting
    log_file = os.path.join(log_dir, f"{name.lower()}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, file_level))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    
    # Console handler with minimal formatting
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level))
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def set_performance_configs():
    """Configure PyTorch CPU performance settings"""
    torch.backends.mkldnn.enabled = True
    torch.backends.mkl.enabled = True
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(multiprocessing.cpu_count())
    
    try:
        proc = psutil.Process()
        proc.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except:
        pass

class MemoryEfficientDataset(Dataset):
    def __init__(self, df, target_column, chunk_size=1000):
        """Memory efficient dataset implementation using memory mapping with proper cleanup"""
        # Create temporary directory for mmap files that will be automatically cleaned up
        self.temp_dir = tempfile.mkdtemp()
        self.features_file = os.path.join(self.temp_dir, 'features.mmap')
        self.labels_file = os.path.join(self.temp_dir, 'labels.mmap')
        self.chunk_size = chunk_size
        
        # Register cleanup on program exit
        atexit.register(self.cleanup)
        
        try:
            # Save data to memory-mapped files
            features = df.drop(target_column, axis=1).values
            labels = df[target_column].values
            
            self.features_mmap = np.memmap(
                self.features_file, 
                dtype='float32', 
                mode='w+', 
                shape=features.shape
            )
            self.labels_mmap = np.memmap(
                self.labels_file, 
                dtype='int64', 
                mode='w+', 
                shape=labels.shape
            )
            
            # Write data in chunks with proper synchronization
            for i in range(0, len(features), chunk_size):
                end_idx = min(i + chunk_size, len(features))
                self.features_mmap[i:end_idx] = features[i:end_idx]
                self.labels_mmap[i:end_idx] = labels[i:end_idx]
                # Force synchronization to disk
                self.features_mmap.flush()
                self.labels_mmap.flush()
            
            self.len = len(features)
            
        except Exception as e:
            self.cleanup()
            raise e
    
    def cleanup(self):
        """Clean up memory-mapped files and temporary directory"""
        try:
            if hasattr(self, 'features_mmap'):
                self.features_mmap._mmap.close()
                del self.features_mmap
            if hasattr(self, 'labels_mmap'):
                self.labels_mmap._mmap.close()
                del self.labels_mmap
            
            # Remove temporary files
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.features_file)
                os.remove(self.labels_file)
                os.rmdir(self.temp_dir)
        except Exception as e:
            warnings.warn(f"Error during cleanup: {e}")
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # Load data from memory-mapped files
        X = torch.FloatTensor(self.features_mmap[idx])
        y = torch.LongTensor([self.labels_mmap[idx]])
        return X, y.squeeze()
    
    def __del__(self):
        """Ensure cleanup on object destruction"""
        self.cleanup()

# Update CustomDataset to use MemoryEfficientDataset for large datasets
class CustomDataset(Dataset):
    def __init__(self, df, target_column):
        memory_threshold = 1e9  # 1GB
        estimated_memory = df.memory_usage().sum()
        
        if estimated_memory > memory_threshold:
            self.dataset = MemoryEfficientDataset(df, target_column)
            self.use_memory_efficient = True
        else:
            self.use_memory_efficient = False
            # Convert to tensors without pinning memory initially
            self.features = torch.FloatTensor(df.drop(target_column, axis=1).values)
            self.labels = torch.LongTensor(df[target_column].values)
            
            # Only pin memory if CUDA is available
            if torch.cuda.is_available():
                self.features = self.features.pin_memory()
                self.labels = self.labels.pin_memory()
    
    def __len__(self):
        return len(self.dataset) if self.use_memory_efficient else len(self.features)
    
    def __getitem__(self, idx):
        if self.use_memory_efficient:
            return self.dataset[idx]
        return self.features[idx], self.labels[idx]
    
    def __del__(self):
        """Ensure proper cleanup"""
        if self.use_memory_efficient:
            del self.dataset
        else:
            del self.features
            del self.labels

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes=3, dropout_rate=0.2, use_batch_norm=True):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        
class CPUOptimizer:
    """Handles CPU-specific optimizations with graceful fallbacks"""
    _instance = None
    _initialized = False
    
    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(CPUOptimizer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config, logger):
        # Only initialize once
        if not self._initialized:
            self.config = config
            self.logger = logger
            self.enabled_features = []
            self.disabled_features = []
            self.setup_cpu_optimizations()
            self._initialized = True
            self._log_optimization_status()

    def setup_cpu_optimizations(self) -> None:
        """Configure available CPU optimizations"""
        perf_config = self.config['training'].get('performance', {})
        
        # Try enabling MKL-DNN
        if perf_config.get('enable_mkldnn', False):
            try:
                import torch.backends.mkldnn
                torch.backends.mkldnn.enabled = True
                self.enabled_features.append('MKL-DNN')
            except:
                self.disabled_features.append('MKL-DNN')
                self.logger.warning("MKL-DNN optimization not available")

        # Try enabling MKL
        if perf_config.get('enable_mkl', False):
            try:
                import torch.backends.mkl
                torch.backends.mkl.enabled = True
                self.enabled_features.append('MKL')
            except:
                self.disabled_features.append('MKL')
                self.logger.warning("MKL optimization not available")

        # Configure number of threads
        try:
            num_threads = os.cpu_count()
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            self.enabled_features.append(f'Multi-threading ({num_threads} threads)')
        except:
            self.disabled_features.append('Custom thread configuration')
            self.logger.warning("Failed to set custom thread configuration")

        # Try enabling mixed precision
        if perf_config.get('mixed_precision', False):
            try:
                # Check if the CPU supports bfloat16
                if not hasattr(torch, 'bfloat16'):
                    raise RuntimeError("bfloat16 not supported")
                self.enabled_features.append('Mixed Precision (bfloat16)')
            except:
                self.disabled_features.append('Mixed Precision')
                self.logger.warning("Mixed precision training not available on this CPU")

    def _log_optimization_status(self) -> None:
        """Log the status of CPU optimizations"""
        self.logger.info("CPU Optimization Status:")
        self.logger.info(f"CPU Architecture: {platform.processor()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        
        if self.enabled_features:
            self.logger.info("Enabled optimizations:")
            for feature in self.enabled_features:
                self.logger.info(f"  - {feature}")
        
        if self.disabled_features:
            self.logger.info("Disabled/Unavailable optimizations:")
            for feature in self.disabled_features:
                self.logger.info(f"  - {feature}")

class WarmupScheduler:
    """Handles learning rate warmup and scheduling"""
    def __init__(self, optimizer: Optimizer, config: Dict[str, Any], num_training_steps: int):
        self.warmup_steps = config['best_model']['warmup_steps']
        if config['optimization']['warmup']['enabled']:
            # Linear warmup followed by cosine decay
            self.warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            # OneCycleLR for the rest of training
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=config['best_model']['learning_rate'],
                total_steps=num_training_steps - self.warmup_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            self.warmup = None
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=config['best_model']['learning_rate'],
                total_steps=num_training_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )

    def step(self, step_num: int):
        if self.warmup and step_num < self.warmup_steps:
            self.warmup.step()
        else:
            self.scheduler.step()

class PyTorchTrainer:
    def __init__(self, model, criterion, optimizer, config=None, device='cpu', verbose=False):
        """Initialize trainer with optional config parameter."""
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        
        # Initialize scheduler-related attributes with safe defaults
        self.warmup_steps = 0
        self.scheduler = None
        self.current_step = 0
        
        # Only try to access config if it exists and has best_model
        if config is not None and 'best_model' in config:
            self.warmup_steps = config['best_model'].get('warmup_steps', 0)
            if config['optimization']['warmup']['enabled']:
                num_training_steps = config['training']['epochs']  # We'll set this properly when train() is called
                self.setup_warmup_scheduler(num_training_steps)

    def setup_warmup_scheduler(self, num_training_steps):
        """Setup the learning rate scheduler with warmup."""
        if self.warmup_steps > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                total_steps=num_training_steps - self.warmup_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )

    def train(self, train_loader, val_loader, epochs, metric='accuracy'):
        """Trains the model for specified number of epochs."""
        # Setup scheduler if not already done
        if self.scheduler is None and self.warmup_steps > 0:
            num_training_steps = len(train_loader) * epochs
            self.setup_warmup_scheduler(num_training_steps)
            
        # ...rest of existing train method code...

    def train_epoch(self, train_loader):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Move data to device
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Update schedulers if they exist
            if self.warmup_steps > 0:
                if self.current_step < self.warmup_steps:
                    self.warmup_scheduler.step()
                elif self.scheduler is not None:
                    self.scheduler.step()
                self.current_step += 1
            
            # Log batch progress
            if batch_idx % 10 == 0:  # Log every 10 batches
                accuracy = 100. * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                if self.verbose:
                    print(f'Train Batch {batch_idx}/{len(train_loader)}: '
                          f'Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy

    def evaluate(self, val_loader):
        """Evaluates the model on validation data."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return total_loss / len(val_loader), accuracy, f1

    def train(self, train_loader, val_loader, epochs, metric='accuracy'):
        """Trains the model for specified number of epochs. 
        Monitors specified validation metric for early stopping."""
        # Setup scheduler at the start of training
        self.setup_scheduler(train_loader)
        
        train_losses, val_losses = [], []
        train_metrics, val_metrics = [], []
        best_val_metric = 0
        
        # Add learning rate tracking
        lr_history = []
        
        for epoch in tqdm(range(epochs), desc='Training'):
            # Track learning rate
            lr_history.append(self.optimizer.param_groups[0]['lr'])
            
            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # Select metric based on config
            train_metric = train_accuracy
            val_metric = val_f1 if metric == 'f1' else val_accuracy
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            best_val_metric = max(best_val_metric, val_metric)
            
            if self.verbose:
                metric_name = 'F1' if metric == 'f1' else 'Accuracy'
                metric_value = val_f1 if metric == 'f1' else val_accuracy
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}: Val {metric_name}: {metric_value:.2f}%, LR: {current_lr:.6f}')
        
        # Add LR curve to the learning curves plot
        self.plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics, 
                                metric_name='F1-Score' if metric == 'f1' else 'Accuracy',
                                lr_history=lr_history)
        
        return train_losses, val_losses, train_metrics, val_metrics, best_val_metric
    
    @staticmethod
    def plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='Accuracy', lr_history=None):
        """Plots the learning curves for loss and chosen metric (accuracy or F1)."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Normalize values for better visualization
        max_loss = max(max(train_losses), max(val_losses))
        max_metric = max(max(train_metrics), max(val_metrics))
        
        epochs = range(1, len(train_losses) + 1)
        
        sns.lineplot(data={
            f"Training {metric_name}": [x/max_metric for x in train_metrics],
            f"Validation {metric_name}": [x/max_metric for x in val_metrics],
            "Training Loss": [x/max_loss for x in train_losses],
            "Validation Loss": [x/max_loss for x in val_losses]
        })
        
        if lr_history:
            sns.lineplot(data={"Learning Rate": lr_history}, linestyle='--')
        
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Value")
        plt.title(f"Training and Validation Loss and {metric_name} Curves")
        plt.legend()
        plt.savefig('learning_curves.png')
        plt.close()

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.best_trial_value = float('-inf')
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_params = None
        os.makedirs(os.path.dirname(config['model']['save_path']), exist_ok=True)
        
        # Enhanced logging setup with config
        self.logger = setup_logger('HyperparameterTuner', config)
        
        # Initialize CPU optimization
        self.cpu_optimizer = CPUOptimizer(config, self.logger)
        self.optimizer_types = ['Adam', 'SGD']  # Add supported optimizers
        
    def save_best_model(self, model, optimizer, trial_value, params):
        """Save the best model and its metadata."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_type': params['optimizer_type'],
            'metric_value': trial_value,
            'hyperparameters': params
        }
        torch.save(checkpoint, self.config['model']['save_path'])
    
    def create_model_and_optimizer(self, trial, train_loader=None):
        """
        Create model, optimizer and scheduler
        Args:
            trial: Optuna trial object
            train_loader: DataLoader for training data, needed for scheduler setup
        """
        # Alternate between optimizers based on trial number
        optimizer_name = self.optimizer_types[trial.number % len(self.optimizer_types)]
        
        # Get model architecture parameters
        hidden_layers = []
        n_layers = trial.suggest_int('n_layers', 1, 4)
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int(f'hidden_layer_{i}', 32, 512))
        
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        weight_decay = 0.0 if use_batch_norm else trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        
        # Create model
        model = MLPClassifier(
            input_size=self.config['model']['input_size'],
            hidden_layers=hidden_layers,
            num_classes=self.config['model']['num_classes'],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Get optimizer parameters based on type
        if optimizer_name == 'Adam':
            lr = float(self.config['training']['optimizer_params']['Adam'].get('base_lr', 1e-4))
            betas = tuple(self.config['training']['optimizer_params']['Adam'].get('betas', (0.9, 0.999)))
            eps = float(self.config['training']['optimizer_params']['Adam'].get('eps', 1e-8))
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        else:  # SGD
            lr = float(self.config['training']['optimizer_params']['SGD'].get('lr', 0.01))
            momentum = float(self.config['training']['optimizer_params']['SGD'].get('momentum', 0.9))
            
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        
        # Record trial parameters
        trial_params = {
            'optimizer_type': optimizer_name,
            'n_layers': n_layers,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm,
            'weight_decay': weight_decay,
            'learning_rate': lr
        }
        
        # Add optimizer-specific parameters
        if optimizer_name == 'Adam':
            trial_params.update({
                'betas': betas,
                'eps': eps
            })
        else:  # SGD
            trial_params.update({
                'momentum': momentum
            })
        
        # Create scheduler if needed
        scheduler = None
        if train_loader is not None and self.config['training'].get('scheduler', {}).get('type'):
            scheduler = self._create_scheduler(optimizer, train_loader)
        
        return model, optimizer, scheduler, trial_params

    def _create_scheduler(self, optimizer, train_loader):
        """Helper method to create learning rate scheduler"""
        total_steps = len(train_loader) * self.config['training']['epochs']
        scheduler_config = self.config['training'].get('scheduler', {})
        
        if scheduler_config.get('type') == 'OneCycleLR':
            try:
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=optimizer.param_groups[0]['lr'] * float(scheduler_config['params']['max_lr_factor']),
                    total_steps=total_steps,
                    div_factor=float(scheduler_config['params']['div_factor']),
                    final_div_factor=float(scheduler_config['params']['final_div_factor']),
                    pct_start=float(scheduler_config['params']['pct_start']),
                    anneal_strategy=scheduler_config['params']['anneal_strategy']
                )
                self.logger.info("Created OneCycleLR scheduler")
                return scheduler
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"Failed to create OneCycleLR scheduler: {e}")
                return None
        return None

    # Update objective method to use scheduler
    def objective(self, trial, train_loader, val_loader):
        model, optimizer, scheduler, trial_params = self.create_model_and_optimizer(
            trial, train_loader=train_loader
        )
        criterion = getattr(nn, self.config['training']['loss_function'])()
        
        # Print trial start with clear separation
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Starting Trial {trial.number}")
        self.logger.info(f"Optimizer: {trial_params['optimizer_type']}")
        self.logger.info("Parameters:")
        for key, value in trial_params.items():
            if key != 'optimizer_type':  # Already logged
                self.logger.info(f"  {key}: {value}")
        self.logger.info("-"*50)
        
        trainer = PyTorchTrainer(
            model, criterion, optimizer,
            config=self.config,
            device=self.config['training']['device'],
            verbose=False  # Set to True for more detailed training logs
        )
        
        patience = self.config['optimization']['early_stopping']['patience']
        min_delta = self.config['optimization']['early_stopping']['min_delta']
        best_metric = float('-inf')
        patience_counter = 0
        last_metric = float('-inf')
        
        self.logger.info(f"\nStarting trial {trial.number}")
        self.logger.info(f"Parameters: {trial_params}")
        
        for epoch in range(self.config['training']['epochs']):
            trainer.train_epoch(train_loader)
            _, accuracy, f1 = trainer.evaluate(val_loader)
            
            metric = f1 if self.config['training']['optimization_metric'] == 'f1' else accuracy
            trial.report(metric, epoch)
            
            # Enhanced early stopping logic with detailed logging
            if metric > best_metric + min_delta:
                improvement = metric - best_metric if best_metric != float('-inf') else metric
                self.logger.info(f"Epoch {epoch}: Metric improved by {improvement:.4f}")
                best_metric = metric
                patience_counter = 0
                
                if metric > self.best_trial_value:
                    self.best_trial_value = metric
                    self.logger.info(f"New best trial metric: {metric:.4f}")
                    self.save_best_model(model, optimizer, metric, trial_params)
            else:
                patience_counter += 1
                self.logger.info(f"Epoch {epoch}: No improvement. Patience: {patience_counter}/{patience}")
            
            # Separate pruning logic with logging
            if metric < last_metric - 0.1:
                self.logger.info(f"Trial {trial.number} pruned due to metric deterioration")
                self.logger.info(f"Current: {metric:.4f}, Previous: {last_metric:.4f}")
                raise optuna.TrialPruned("Trial pruned due to metric deterioration")
            
            last_metric = metric
            
            # Early stopping check with logging
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"Best metric achieved: {best_metric:.4f}")
                break
            
            # Log iteration progress
            if epoch % 5 == 0 or epoch == self.config['training']['epochs'] - 1:
                self.logger.debug(
                    f"Epoch {epoch + 1}/{self.config['training']['epochs']}: "
                    f"Metric = {metric:.4f}, Best = {best_metric:.4f}"
                )
        
        self.logger.info(f"Trial {trial.number} finished. Final metric: {best_metric:.4f}\n")
        return best_metric

    def tune(self, train_loader, val_loader):
        """Run hyperparameter tuning with enhanced logging"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting Hyperparameter Tuning")
        self.logger.info(f"Number of trials: {self.config['optimization']['n_trials']}")
        self.logger.info(f"Early stopping patience: {self.config['optimization']['early_stopping']['patience']}")
        self.logger.info(f"Early stopping min delta: {self.config['optimization']['early_stopping']['min_delta']}")
        self.logger.info("="*50 + "\n")
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader),
            n_trials=self.config['optimization']['n_trials']
        )
        
        # Final summary
        self.logger.info("\n" + "="*50)
        self.logger.info("Tuning Completed!")
        self.logger.info(f"Best trial value: {study.best_trial.value:.4f}")
        self.logger.info("Best parameters:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("="*50 + "\n")
        
        return study.best_trial, study.best_params

    # ...rest of existing code...

def restore_best_model(config):
    """Restore best model with correct optimizer type"""
    checkpoint = torch.load(config['model']['save_path'], weights_only=True)
    
    # Create model
    model = MLPClassifier(
        input_size=config['model']['input_size'],
        hidden_layers=checkpoint['hyperparameters']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=checkpoint['hyperparameters']['dropout_rate'],
        use_batch_norm=checkpoint['hyperparameters']['use_batch_norm']
    )
    
    # Create correct optimizer type
    optimizer_type = checkpoint['optimizer_type']
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=checkpoint['hyperparameters']['learning_rate'],
            betas=checkpoint['hyperparameters'].get('betas', (0.9, 0.999)),
            eps=checkpoint['hyperparameters'].get('eps', 1e-8),
            weight_decay=checkpoint['hyperparameters'].get('weight_decay', 0.0)
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=checkpoint['hyperparameters']['learning_rate'],
            momentum=checkpoint['hyperparameters'].get('momentum', 0.9),
            weight_decay=checkpoint['hyperparameters'].get('weight_decay', 0.0)
        )
    
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'model': model,
        'optimizer': optimizer,
        'optimizer_type': optimizer_type,
        'metric_value': checkpoint['metric_value'],
        'hyperparameters': checkpoint['hyperparameters']
    }

def save_best_params_to_config(config_path, best_trial, best_params):
    """Save best parameters to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create best_model section if it doesn't exist
    if 'best_model' not in config:
        config['best_model'] = {}
    
    # Format parameters for config
    hidden_layers = [best_params[f'hidden_layer_{i}'] for i in range(best_params['n_layers'])]
    
    # Get learning rate from config if not in best_params
    learning_rate = best_params.get('lr', config['training']['optimizer_params']['Adam']['base_lr'])
    
    config['best_model'].update({
        'hidden_layers': hidden_layers,
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': learning_rate,  # Use the retrieved learning rate
        'use_batch_norm': best_params['use_batch_norm'],
        'weight_decay': best_params.get('weight_decay', 0.0),
        'best_metric_name': config['training']['optimization_metric'],
        'best_metric_value': best_trial.value
    })
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def train_final_model(config, train_loader, val_loader):
    """Train model using parameters from config."""
    best_model_config = config['best_model']
    
    final_model = MLPClassifier(
        input_size=config['model']['input_size'],
        hidden_layers=best_model_config['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=best_model_config['dropout_rate'],
        use_batch_norm=best_model_config['use_batch_norm']
    )
    
    criterion = getattr(nn, config['training']['loss_function'])()
    optimizer = getattr(torch.optim, config['training']['optimizer_choice'])(
        final_model.parameters(),
        lr=best_model_config['learning_rate'],
        weight_decay=best_model_config['weight_decay']
    )
    
    final_trainer = PyTorchTrainer(
        final_model, criterion, optimizer,
        config=config,
        device=config['training']['device'],
        verbose=True
    )
    
    return final_trainer.train(
        train_loader, 
        val_loader, 
        config['training']['epochs'],
        metric=config['training']['optimization_metric']
    )

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTH longer ONHASHSEED'] = str(seed)

def log_cpu_optimizations(config, logger):
    """Log CPU optimization settings at startup"""
    logger.info("CPU Optimization Status:")
    logger.info(f"CPU Architecture: {platform.processor()}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    
    # Log CPU optimization settings
    cpu_opts = config['training'].get('cpu_optimization', {})
    dataloader_opts = config['training'].get('dataloader', {})
    perf_opts = config['training'].get('performance', {})
    
    enabled_features = []
    if cpu_opts.get('enable_mkldnn'):
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True
            enabled_features.append('MKL-DNN')
    
    if perf_opts.get('enable_mkl'):
        if hasattr(torch.backends, 'mkl'):
            torch.backends.mkl.enabled = True 
            enabled_features.append('MKL')

    if cpu_opts.get('use_bfloat16'):
        if hasattr(torch, 'bfloat16'):
            enabled_features.append('Mixed Precision (bfloat16)')
    
    # Log enabled features
    if enabled_features:
        logger.info("Enabled CPU optimizations:")
        for feature in enabled_features:
            logger.info(f"  - {feature}")
            
    # Log DataLoader settings
    logger.info("\nDataLoader settings:")
    logger.info(f"  Number of workers: {dataloader_opts.get('num_workers', 'auto')}")
    logger.info(f"  Pin memory: {dataloader_opts.get('pin_memory', False)}")
    logger.info(f"  Persistent workers: {dataloader_opts.get('persistent_workers', False)}")
    logger.info(f"  Prefetch factor: {dataloader_opts.get('prefetch_factor', 2)}")

def main():
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Set up logging with config
    logger = setup_logger('MLPTrainer', config)
    
    # Log CPU optimization settings at startup
    log_cpu_optimizations(config, logger)
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create datasets and dataloaders
    train_df = pd.read_csv(config['data']['train_path'])
    val_df = pd.read_csv(config['data']['val_path'])
    train_dataset = CustomDataset(train_df, config['data']['target_column'])
    val_dataset = CustomDataset(val_df, config['data']['target_column'])
    
    # Use correct path to performance settings
    batch_size = config['training']['batch_size']
    if 'training' in config and 'performance' in config['training']:
        batch_size *= config['training']['performance'].get('batch_size_multiplier', 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(
            config['training']['dataloader']['num_workers'],
            multiprocessing.cpu_count()
        ),
        pin_memory=config['training']['dataloader']['pin_memory'],
        prefetch_factor=config['training']['dataloader']['prefetch_factor'],
        persistent_workers=config['training']['dataloader']['persistent_workers'],
        drop_last=config['training'].get('drop_last', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(
            config['training']['dataloader']['num_workers'],
            multiprocessing.cpu_count()
        ),
        pin_memory=config['training']['dataloader']['pin_memory'],
        prefetch_factor=config['training']['dataloader']['prefetch_factor'],
        persistent_workers=config['training']['dataloader']['persistent_workers']
    )
    
    # If best parameters don't exist in config, run hyperparameter tuning
    if 'best_model' not in config:
        tuner = HyperparameterTuner(config)
        best_trial, best_params = tuner.tune(train_loader, val_loader)
        save_best_params_to_config(config_path, best_trial, best_params)
        # Reload config with saved parameters
        config = load_config(config_path)
    
    print("\nBest model parameters from config:")
    for key, value in config['best_model'].items():
        print(f"    {key}: {value}")
    
    # Restore best model from checkpoint
    print("\nRestoring best model from checkpoint...")
    restored = restore_best_model(config)
    model = restored['model']
    optimizer = restored['optimizer']
    
    # Create criterion for evaluation
    criterion = getattr(nn, config['training']['loss_function'])()
    
    # Create trainer for evaluation
    trainer = PyTorchTrainer(
        model, criterion, optimizer,
        config=config,
        device=config['training']['device'],
        verbose=True
    )
    
    # Evaluate restored model
    print("\nEvaluating restored model on validation set...")
    val_loss, val_accuracy, val_f1 = trainer.evaluate(val_loader)
    
    metric_name = config['training']['optimization_metric']
    metric_value = val_f1 if metric_name == 'f1' else val_accuracy
    
    print(f"\nRestored model performance:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"\nBest {metric_name.upper()} from tuning: {restored['metric_value']:.4f}")
    print(f"Current {metric_name.upper()}: {metric_value:.4f}")

if __name__ == "__main__":
    main()