from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class LitModel(LightningModule):
    """Lightning module for miniCDDD"""

    def __init__(
            self,
            model,
            learning_rate=5e-4,
            scheduler_decay=0.9,
            scheduler_epochs=10,
            reconstruction_weight=1.0,
            classification_weight=1.0
    ):
        """
        Initialize the Lightning Module

        Args:
            model: The MiniCDDD model
            learning_rate: Initial learning rate
            scheduler_decay: Decay rate for learning rate scheduler
            scheduler_epochs: Steps for scheduler
            reconstruction_weight: Weight for reconstruction loss
            classification_weight: Weight for classification loss
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.scheduler_decay = scheduler_decay
        self.scheduler_epochs = scheduler_epochs
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight

        # Loss functions
        self.reconstruction_loss_fn = nn.CrossEntropyLoss()
        self.classification_loss_fn = nn.MSELoss()

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])

    def forward(self, encoder_inputs, decoder_inputs):
        return self.model(encoder_inputs, decoder_inputs)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(
            optimizer,
            gamma=self.scheduler_decay
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': self.scheduler_epochs,
            }
        }

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def _common_step(self, batch, batch_idx, stage):
        encoder_inputs = batch['encoder_inputs'].to(torch.bfloat16)
        decoder_inputs = batch['decoder_inputs'].to(torch.bfloat16)
        decoder_outputs = batch['decoder_outputs']
        classifier_targets = batch['classifier_outputs']

        # Forward pass
        reconstruction_logits, classification_outputs = self(encoder_inputs, decoder_inputs)

        # Calculate losses
        reconstruction_loss = self.reconstruction_loss_fn(
            reconstruction_logits.view(-1, self.model.vocab_size),
            decoder_outputs.reshape(-1)
        )

        classification_loss = self.classification_loss_fn(
            classification_outputs,
            classifier_targets
        )

        # Combined loss
        total_loss = (
                self.reconstruction_weight * reconstruction_loss +
                self.classification_weight * classification_loss
        )

        # Calculate accuracies
        reconstruction_preds = torch.argmax(reconstruction_logits, dim=-1)
        reconstruction_accuracy = (reconstruction_preds == decoder_outputs).float().mean()

        # Log metrics
        self.log(f'{stage}_total_loss', total_loss)
        self.log(f'{stage}_reconstruction_loss', reconstruction_loss)
        self.log(f'{stage}_classification_loss', classification_loss)
        self.log(f'{stage}_reconstruction_accuracy', reconstruction_accuracy)

        return total_loss

    def get_encoder(self):
        """Get the encoder part of the model"""
        return self.model.encoder

    def get_decoder(self):
        """Get the decoder part of the model"""
        return self.model.decoder

    def get_classifier(self):
        """Get the classifier part of the model"""
        return self.model.classifier

    def get_seq2seq(self):
        """Get encoder and decoder as a sequence-to-sequence model"""
        # This just returns the current model without the classifier output
        return self.model

    def denormalize_predictions(self, predictions, mean, std):
        """Denormalize model predictions using mean and standard deviation"""
        mean_tensor = torch.tensor(mean, device=predictions.device)
        std_tensor = torch.tensor(std, device=predictions.device)
        return predictions * std_tensor + mean_tensor