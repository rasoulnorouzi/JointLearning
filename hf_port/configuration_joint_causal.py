# configuration_joint_causal.py

from transformers import PretrainedConfig

class JointCausalConfig(PretrainedConfig):
    """
    This is the configuration class for JointCausalModel, following the tutorial's guidelines.
    """
    # The 'model_type' is crucial for AutoClass support, as mentioned in the tutorial.
    model_type = "joint_causal"

    def __init__(
        self,
        encoder_name="bert-base-uncased",
        num_cls_labels=2,
        num_bio_labels=7,
        num_rel_labels=2,
        dropout=0.2,
        **kwargs,
    ):
        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout = dropout
        # As per the tutorial, we must pass any extra kwargs to the superclass.
        super().__init__(**kwargs)