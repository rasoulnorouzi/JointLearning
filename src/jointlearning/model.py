from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls

# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
Span = Tuple[int, int]  # inclusive indices (token indices)
label2id_bio = {v: k for k, v in id2label_bio.items()}
label2id_rel = {v: k for k, v in id2label_rel.items()}
label2id_cls = {v: k for k, v in id2label_cls.items()}

# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
"""Joint Causal Extraction Model (softmax)
============================================================================

A PyTorch module for joint causal extraction using softmax decoding for BIO tagging.
The model supports class weights for handling imbalanced data.

```python
>>> model = JointCausalModel()        # softmax-based model
```

---------------------------------------------------------------------------
Usage overview
---------------------------------------------------------------------------

**Training**
~~~~~~~~~~~~
(Training code example omitted for brevity, see previous versions)
---------------------------------------------------------------------------
Implementation
---------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class JointCausalModel(nn.Module, PyTorchModelHubMixin):
    """Encoder + three heads with **optional CRF** BIO decoder.

    This model integrates a pre-trained transformer encoder with three distinct
    heads for:
    1. Classification (cls_head): Predicts a global label for the input.
    2. BIO tagging (bio_head): Performs sequence tagging using BIO scheme.
       Can operate with a CRF layer or standard softmax.
    3. Relation extraction (rel_head): Identifies relations between entities
       detected by the BIO tagging head.
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        encoder_name: str = MODEL_CONFIG["encoder_name"],
        num_cls_labels: int = MODEL_CONFIG["num_cls_labels"],
        num_bio_labels: int = MODEL_CONFIG["num_bio_labels"],
        num_rel_labels: int = MODEL_CONFIG["num_rel_labels"],
        dropout: float = MODEL_CONFIG["dropout"],
    ) -> None:
        """Initializes the JointCausalModel.

        Args:
            encoder_name: Name of the pre-trained transformer model to use
                (e.g., "bert-base-uncased").
            num_cls_labels: Number of labels for the classification task.
            num_bio_labels: Number of labels for the BIO tagging task.
            num_rel_labels: Number of labels for the relation extraction task.
            dropout: Dropout rate for regularization.
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout_rate = dropout
        self.enc = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.enc.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_cls_labels),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_bio_labels),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_rel_labels),
        )
        self._init_new_layer_weights()

    def get_config_dict(self) -> Dict:
        """Returns the model's configuration as a dictionary."""
        return {
            "encoder_name": self.encoder_name,
            "num_cls_labels": self.num_cls_labels,
            "num_bio_labels": self.num_bio_labels,
            "num_rel_labels": self.num_rel_labels,
            "dropout": self.dropout_rate,
        }

    @classmethod
    def from_config_dict(cls, config: Dict) -> "JointCausalModel":
        """Creates a JointCausalModel instance from a configuration dictionary."""
        return cls(**config)

    def _init_new_layer_weights(self):
        """Initializes the weights of the newly added linear layers.

        Uses Xavier uniform initialization for weights and zeros for biases.
        """
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub_module in mod.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the input using the transformer model.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens to attend to.

        Returns:
            Tensor of hidden states from the encoder, passed through dropout
            and layer normalization.
        """
        hidden_states = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.layer_norm(self.dropout(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        bio_labels: torch.Tensor | None = None, 
        pair_batch: torch.Tensor | None = None,
        cause_starts: torch.Tensor | None = None,
        cause_ends: torch.Tensor | None = None,
        effect_starts: torch.Tensor | None = None,
        effect_ends: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        """Performs a forward pass through the model.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens to attend to.
            bio_labels: Optional tensor of BIO labels for training.
            pair_batch: Optional tensor indicating which hidden states to use
                for relation extraction.
            cause_starts: Optional tensor of start indices for cause spans.
            cause_ends: Optional tensor of end indices for cause spans.
            effect_starts: Optional tensor of start indices for effect spans.
            effect_ends: Optional tensor of end indices for effect spans.

        Returns:
            A dictionary containing:
                - "cls_logits": Logits for the classification task.
                - "bio_emissions": Emissions from the BIO tagging head.
                - "tag_loss": Loss for the BIO tagging task (if bio_labels provided).
                - "rel_logits": Logits for the relation extraction task (if
                  relation extraction inputs provided).
        """
        # Encode input
        hidden = self.encode(input_ids, attention_mask)

        # Classification head
        cls_logits = self.cls_head(hidden[:, 0])  # Use [CLS] token representation

        # BIO tagging head
        emissions  = self.bio_head(hidden)
        tag_loss: Optional[torch.Tensor] = None

        # Calculate BIO tagging loss if labels are provided
        if bio_labels is not None:
            # Softmax loss (typically handled by the training loop's loss function, e.g., CrossEntropyLoss)
            # Here, we initialize it to 0.0 as a placeholder.
            # The actual loss calculation for softmax would compare emissions with bio_labels.
            tag_loss = torch.tensor(0.0, device=emissions.device)

        # Relation extraction head
        rel_logits: torch.Tensor | None = None
        if pair_batch is not None and cause_starts is not None and cause_ends is not None \
           and effect_starts is not None and effect_ends is not None:
            # Select hidden states corresponding to the pairs for relation extraction
            bio_states_for_rel = hidden[pair_batch] 
            seq_len_rel = bio_states_for_rel.size(1)
            pos_rel = torch.arange(seq_len_rel, device=bio_states_for_rel.device).unsqueeze(0)

            # Create masks for cause and effect spans
            c_mask = ((cause_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= cause_ends.unsqueeze(1))).unsqueeze(2)
            e_mask = ((effect_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= effect_ends.unsqueeze(1))).unsqueeze(2)

            # Compute mean-pooled representations for cause and effect spans
            c_vec = (bio_states_for_rel * c_mask).sum(1) / c_mask.sum(1).clamp(min=1) # Average pooling, clamp to avoid div by zero
            e_vec = (bio_states_for_rel * e_mask).sum(1) / e_mask.sum(1).clamp(min=1) # Average pooling, clamp to avoid div by zero
            
            # Concatenate cause and effect vectors and pass through relation head
            rel_logits = self.rel_head(torch.cat([c_vec, e_vec], dim=1))

        return {
            "cls_logits": cls_logits,
            "bio_emissions": emissions,
            "tag_loss": tag_loss, 
            "rel_logits": rel_logits, 
        }
