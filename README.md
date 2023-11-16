# TensorflowCallback
adjusted to penalize for overfitting

```py
import tensorflow as tf
class SaveModelOnAdjustedValLoss(tf.keras.callbacks.Callback):
    def __init__(self, path, max_delta, patience=0):
        super(SaveModelOnAdjustedValLoss, self).__init__()
        self.path = path
        self.max_delta = max_delta  # This could be used as a threshold for the allowed loss difference
        self.patience = patience
        self.best_adjusted_val_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        current_train_loss = logs.get('loss')
        loss_diff = abs(current_val_loss - current_train_loss)

        # Adjust validation loss by penalizing based on the difference from training loss
        adjusted_val_loss = current_val_loss + loss_diff

        # Check if the adjusted validation loss is better and within the delta threshold
        if adjusted_val_loss < self.best_adjusted_val_loss and loss_diff < self.max_delta:
            self.best_adjusted_val_loss = adjusted_val_loss
            self.wait = 0
            self.model.save(self.path)
            print(f'\nModel saved at epoch {epoch}, with adjusted val_loss {adjusted_val_loss:.4f}')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'\nNo improvement in adjusted val_loss for {self.patience} epochs. Training stopped.')
                self.model.stop_training = True

```
