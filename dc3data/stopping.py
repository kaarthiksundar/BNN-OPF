import jax.numpy as jnp 
import logging

class PatienceThresholdStoppingCriteria:
    def __init__(self, log, threshold=1e-8, patience=3):
        self.threshold = threshold
        self.patience = patience 
        self.best_loss = jnp.inf
        self.vi_parameters = None 
        self.wait = 0 
        self.log = log
        self.stop_training = False
        
        
    def on_epoch_end(self, epoch, current_loss, vi_parameters):
        if jnp.isnan(current_loss):
            self.log.info(f'Current test loss is NaN; skipping patience check')
        
        self.log.info(f'current loss: {current_loss:.4E}, best loss: {self.best_loss:.4E}')
        if current_loss < self.best_loss - self.threshold:
            self.log.info(f'model store at epoch/round {epoch}; better loss')
            self.best_loss = current_loss
            self.wait = 0
            self.vi_parameters = vi_parameters
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                self.log.info(f'Stopping training early at epoch/round {epoch + 1} due to patience criteria')
                
    def reset_wait(self):
        self.wait = 0
        self.stop_training = False
        
    def reset(self):
        self.best_loss = jnp.inf
        self.vi_parameters = None 
        self.wait = 0 
        self.stop_training = False
