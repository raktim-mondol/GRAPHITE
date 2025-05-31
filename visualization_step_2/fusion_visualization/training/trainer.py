from utils.imports import *

from training.losses import HierarchicalInfoMaxLoss, LossTracker
from tqdm import tqdm
import time
import logging
from pathlib import Path
import torch
from tqdm import tqdm
import time

class HierGATSSLTrainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 device,
                 checkpoint_dir,
                 num_epochs=100,
                 patience=10,
                 lr_patience=5,
                 start_epoch=0):  # Added start_epoch):
        """
        Initialize trainer
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.patience = patience
        
        self.start_epoch = start_epoch  # Store start_epoch
        
        # Initialize loss tracker
        self.loss_tracker = LossTracker()
        
        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=lr_patience,
            factor=0.5,
            min_lr=1e-6,
            verbose=True
        )
        
        # Initialize early stopping
        self.best_loss = float('inf')
        self.no_improve_count = 0
        
        # Setup logger
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        # Create logger
        logger = logging.getLogger('HierGATSSL')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.checkpoint_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def train(self, train_loader):
        """Training loop"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting training...")
        self.logger.info(f"Number of images: {len(train_loader.dataset)}")
        self.logger.info(f"Batch size: {train_loader.batch_size}")
        self.logger.info(f"Total batches per epoch: {len(train_loader)}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info("="*50 + "\n")
        
        #for epoch in range(self.num_epochs):
        for epoch in range(self.start_epoch, self.num_epochs):  # Use start_epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train epoch
            epoch_losses = self.train_epoch(train_loader)
            
            # Log epoch results
            self.logger.info("\nEpoch Summary:")
            for k, v in epoch_losses.items():
                self.logger.info(f"{k}: {v:.4f}")
            
            # Update learning rate scheduler
            self.scheduler.step(epoch_losses['total_loss'])
            
            # Check for improvement
            if epoch_losses['total_loss'] < self.best_loss:
                self.best_loss = epoch_losses['total_loss']
                self.no_improve_count = 0
                self.save_checkpoint(epoch, epoch_losses, is_best=True)
                self.logger.info(f"New best model saved! Loss: {self.best_loss:.4f}")
            else:
                self.no_improve_count += 1
                self.logger.info(f"No improvement for {self.no_improve_count} epochs")
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, epoch_losses)
            
            # Plot losses
            self.loss_tracker.plot_losses(
                self.checkpoint_dir / f'losses_epoch_{epoch+1}.png'
            )
            
            # Early stopping check
            if self.no_improve_count >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        
    def train_epoch(self, train_loader):
        """Train for one epoch with detailed monitoring"""
        self.model.train()
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')  # Track data loading time
        losses = {
            'total_loss': AverageMeter('Total Loss', ':.4f'),
            'infomax_loss': AverageMeter('InfoMax Loss', ':.4f'),
            'scale_loss': AverageMeter('Scale Loss', ':.4f')
        }
        
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time] + list(losses.values()),
            prefix='Epoch: '
        )
        
        end = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            try:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                outputs = self.model(batch)
                
                # Compute loss
                loss_dict = self.criterion(outputs)
                
                # Check for NaN losses
                if torch.isnan(loss_dict['total_loss']):
                    self.logger.warning(f"NaN loss detected in batch {batch_idx}!")
                    continue
                
                # Update meters
                for loss_name, loss_value in loss_dict.items():
                    if loss_name in losses:
                        losses[loss_name].update(loss_value.item())
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Print progress
                if batch_idx % 5 == 0:  # Increased frequency of logging
                    self.logger.info(
                        f'Batch: [{batch_idx}/{len(train_loader)}] '
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        f'Total Loss {losses["total_loss"].val:.4f} ({losses["total_loss"].avg:.4f}) '
                        f'InfoMax Loss {losses["infomax_loss"].val:.4f} ({losses["infomax_loss"].avg:.4f}) '
                        f'Scale Loss {losses["scale_loss"].val:.4f} ({losses["scale_loss"].avg:.4f})'
                    )
                
                # Detailed progress every 10 batches
                if batch_idx % 10 == 0:
                    progress.display(batch_idx, self.logger)
                    
                    # Log graph statistics
                    self.logger.info(
                        f"Graph Stats - Nodes: {batch.num_nodes}, "
                        f"Edges: {batch.edge_index.size(1)}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                self.logger.error(f"Stack trace: {traceback.format_exc()}")
                continue
        
        # Compute average losses for the epoch
        avg_losses = {k: meter.avg for k, meter in losses.items()}
        
        # Log final epoch statistics
        self.logger.info(
            f"\nEpoch Summary - "
            f"Avg Time/Batch: {batch_time.avg:.3f}s, "
            f"Avg Data Time/Batch: {data_time.avg:.3f}s"
        )
        
        return avg_losses
        
   
    
    def _compute_average_losses(self, losses):
        """Compute average losses over multiple batches"""
        avg_losses = {}
        for k in losses[0].keys():
            avg_losses[k] = sum(d[k] for d in losses) / len(losses)
        return avg_losses
    
    def save_checkpoint(self, epoch, losses, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'best_loss': self.best_loss,
            'no_improve_count': self.no_improve_count
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
                
                
class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)