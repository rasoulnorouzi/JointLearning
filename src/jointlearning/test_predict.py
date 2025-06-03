# test_predict.py
"""
Test script for the HuggingFace-compatible `predict` method of JointCausalModel.
Mirrors the previous usage of prediction.py, using the same test sentences and settings.
"""
import json
import torch
from transformers import AutoTokenizer
from model import JointCausalModel
try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_cls
    from .model import JointCausalModel
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_cls  # type: ignore
    from model import JointCausalModel  # type: ignore

# Load model (adjust path or repo as needed)
# If you have a local checkpoint, use the path. Otherwise, use the HuggingFace repo name.
MODEL_PATH = "src\jointlearning\expert_bert_GCE_Softmax_Normal\expert_bert_GCE_Softmax_Normal_model.pt"  # <-- CHANGE THIS to your model path or repo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])

def main():
    # Test sentences (same as in docstring/example)
    test_sents = [
        "Insomnia causes depression and a lack of concentration in children",
        "smoking causes lung cancer and heart disease",
        "exercise improves physical health and mental well-being",
        "Permitting continuous rather than binary ''all-or-nothing'' contributions significantly increases contributions and facilitates provision.",
        "according to smith (2008), there is a need for substantial and consistent support and encouragement for women to participate in studies.",
        "thus, experiencing a task that leads to decrements in feelings of relatedness may affect people's ability to experience intrinsic motivation by also influencing people's mood.;;",
        "It is instructive to contrast this estimate to the one in the previous section, based on the very simple, two-parameter (g, q) model.;;",
        "she also recognized that structurally disadvantaged communities supply the black and brown bodies that fill chicago's jail and illinois's prisons (lavigne et al.;;",
        "the effect of constrained communication and limited information 623 communication, the content of the communication may also reveal which kind of information is more important to the participants.;;",
        "the subjects who were dependent on the other for future aid increased their level of help giving across the trials.;;",
        "instead, depleted participants were more modest in their predictions and more accurate in their predictions than nondepleted participants.;;",
        "the perceived consequences of turning to others for social support, therefore, may influence the expression of pain.;;",
        "moreover, in the context of cooperation in organizational and legal settings, de cremer and tyler (2007) showed that if a party communicates intentions to listen to others and take their interests at heart, cooperative decision making is only promoted if this other is seen as honest and trustworthy.;;",
        "A significant rise in local unemployment rates is a primary driver of increased property crime in the metropolitan area.",
        "Consistent and responsive caregiving in the first year of life is a crucial factor in the development of a secure attachment style in children.",
        "The prolonged drought led to widespread crop failure, which in turn caused a sharp increase in food prices, ultimately contributing to social unrest in the region."
    ]

    # Load model and tokenizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = JointCausalModel.from_pretrained(MODEL_PATH).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model=JointCausalModel(**MODEL_CONFIG)
    model.load_state_dict(torch.load("src\\jointlearning\\expert_bert_GCE_Softmax_Normal\\expert_bert_GCE_Softmax_Normal_model.pt",map_location=DEVICE))
    model.to(DEVICE).eval(); 

    # Run prediction (using the same settings as before)
    results = model.predict(
        test_sents,
        tokenizer=TOKENIZER,
        rel_mode="auto",           # or "auto"
        rel_threshold=0.5,         # adjust as needed
        cause_decision="cls+span" # or "cls_only", "span_only"
    )

    # Print results
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
