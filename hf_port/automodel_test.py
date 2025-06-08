# %%
from transformers import AutoModel, AutoTokenizer
import os
import json


# %%
# This simple, standard call will now work perfectly.
model = AutoModel.from_pretrained(
    "rasoultilburg/SocioCausaNet",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "rasoultilburg/SocioCausaNet",
    trust_remote_code=True,
)
print(f"Successfully loaded model of type: {type(model)}")

# %%
test_sents = [
    "promoting ri might reduce risk factors for drug use and enhance the effects of protective factors (brook et al., 1998).;;",
        "it is also considered that the process could easily be adapted for virtual delivery, thus increasing its accessibility.;;",
        "(corrected for unreliability; Bryk and Raudenbush 1992).;;",
        "big data technologies, however, facilitate the collection and sharing of these data on a large scale.;;",
        "depending on how successful and consistent the analyst was while inventing terms and coupling them with contexts, the resulting network would also become intuitively meaningful to any native speaker (see below).;;",
        "thus, this would intensify interpersonal stress with their family members and increase the risk of relapse.;;",
        "for instance, schneider and turkat (1975) reported that low self-esteem individuals were more dependent on the evaluations of others in determining their sense of self-worth.;;",
        "in many programme areas, this is in fact possible since there are frequently follow-on programmes whose planning stages could deploy a review approach, but of course it is rarely done.;;",
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
        "The prolonged drought led to widespread crop failure, which in turn caused a sharp increase in food prices, ultimately contributing to social unrest in the region.",
        "smoking causes lung cancer!"
        
        ]


results = model.predict(
        test_sents,
        tokenizer=tokenizer,
        rel_mode="auto",           # or "auto"
        rel_threshold=0.5,         # adjust as needed
        cause_decision="cls+span" # or "cls_only", "span_only"
    )
print(json.dumps(results, indent=2, ensure_ascii=False))

# %%
