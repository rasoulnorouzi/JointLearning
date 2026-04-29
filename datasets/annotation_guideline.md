# Annotation Guideline for Causal Sentence Labeling

## 1. Introduction

### 1.1 Purpose

Capturing causal relationships in social science clarifies the mechanisms behind social phenomena and supports the development of robust theories (Hedström & Ylikoski, 2010). Researchers represent these relationships through a Directed Acyclic Graph (DAG), which depicts causes and effects as a network of directed links (Digitale et al., 2022). A DAG helps researchers hypothesize and test the pathways that connect a cause to an effect, examine how variables relate to one another, and strengthen their causal claims (Gross, 2018). Empirical researchers also rely on DAGs to detect patterns across variables, identify potential biases, and refine their assumptions about the mechanisms driving social phenomena (Shrier & Platt, 2008).

### 1.2 Definitions

A **cause** is a condition or event that produces a change or brings about an outcome. In social science, a cause typically appears as a factor, variable, or manipulation that shapes behavior or generates a social phenomenon (Halpern, 2016).

An **effect** is the consequence, outcome, or result that follows from the cause. In social science, effects often take the form of shifts in social behavior or changes in economic conditions (Moustakas, 2023).

A **causal relationship** links two variables such that a change in one (the cause) produces a change in the other (the effect) (Halpern, 2016). As Pearl (1995) put it, the effect "listens to" its causes: when the cause changes, the effect changes with it.

A **cause-effect (CE) span** is a phrase that occupies both roles within the same sentence: it is the effect of one relation and, at the same time, the cause of another. CE spans appear in causal chains where one event produces an intermediate state that in turn produces a further outcome. Annotators tag such spans with the dedicated CE label rather than splitting them across separate cause and effect annotations.

The **connection** between a cause and an effect refers to the directed link that ties them together within the sentence. Annotators mark this connection whenever a cause-effect pair appears, regardless of the sentiment the sentence may evoke. For example, the sentence "smoking causes lung cancer" carries a negative sentiment because lung cancer is a serious disease, but the sentence still expresses a single causal connection between "smoking" and "lung cancer" that the annotator records.

## 2. Annotation Goals

This project produces a labeled dataset of causal statements that identifies the cause, the effect, any cause-effect span that participates in a causal chain, and the connection that links each pair. The dataset trains machine learning models to perform the same task, which in turn allows the automatic extraction of DAGs from written text.

## 3. Datasets

The corpus draws on three sources:

- **Cooperation Databank (CoDa)**: 2,636 studies on human cooperation conducted across 78 societies between 1958 and 2017 (Spadaro et al., 2022).
- **Qualitative Research Methodology Papers**: 446 scholarly articles on qualitative research methodologies, retrieved from Scopus through keyword-based filtering.
- **Cyberball Meta-Analysis Studies**: studies drawn from a meta-analysis of 120 Cyberball experiments on social ostracism (Hartgerink et al., 2015).

To screen academic articles for causal sentences, we use our fine-tuned `ssc-bert` classifier for social science (Norouzi et al., 2024), which sorts sentences into likely causal and likely non-causal pools. We then sample from both pools so the annotated set contains a balanced mix of causal and non-causal examples.

## 4. The Three-Step Annotation Task

Annotators complete three steps in order for every sentence. The Doccano interface (Nakayama et al., 2018) supports each step.

**Step 1: Sentence-level classification.** Read the full sentence and label it as *causal* or *non-causal*. A sentence is causal when it asserts that one variable, condition, or event produces, modifies, or shapes another. Sentences that report correlations, describe methods, or present background information without a causal claim are non-causal. If the sentence is non-causal, stop here and move to the next sentence.

**Step 2: Span marking.** For every sentence labeled causal, mark each cause, effect, and cause-effect (CE) span using the tagging rules in §5. Spans are character-level selections in Doccano.

**Step 3: Relation linking.** Link each cause span to the effect span (or CE span) it produces. Where a CE span participates in a chain, link it as an effect to its upstream cause and as a cause to its downstream effect. Mark every cause-effect pair the sentence supports; do not omit pairs because they feel redundant.

## 5. Guidelines for Identifying Causes, Effects, and Their Connections

To capture a causal sentence and its elements, read the sentence in full and identify a variable (the cause) whose presence or modification leads to an outcome or shifts the state of another variable (the effect). Once you have identified the pair, mark the connection that links them. When an intermediate phrase serves as both effect and cause, tag it as a CE span (see §5.2).

### 5.1 Span Boundary Rules

Choose span boundaries that capture the full meaning of the cause or effect without sweeping in extraneous material. The following rules apply:

- **Mark the minimal span that fully conveys the variable.** Include modifiers, qualifiers, and prepositional phrases when they are essential to the meaning (e.g., "switching from short games to long games," not just "switching"). Drop them when they are decorative.
- **Exclude causal connectors and discourse markers.** Words such as *because*, *causes*, *leads to*, *therefore*, *as a result*, *due to*, and *since* signal the relation; they are not part of the cause or effect span.
- **Exclude leading articles when they add nothing.** Drop bare *the*, *a*, or *an* at the start of a span unless removing them would alter the reading.
- **Keep the span contiguous.** Spans must be a continuous run of characters. If a single conceptual variable appears in two non-adjacent locations, mark whichever stretch best expresses it.
- **Match form across cause and effect when possible.** If the cause is a noun phrase, prefer a noun-phrase effect; if the cause is a clause, a clausal effect is acceptable. Consistency helps downstream models learn stable patterns.
- **Partial overlap is treated as agreement.** During reliability assessment, two annotators who select overlapping but non-identical spans are scored as agreeing (Seki & Mostafa, 2003; Tsai et al., 2006). Aim for the cleanest minimal span; do not stretch boundaries to match another annotator.

### 5.2 Tagging Causal Chains with the CE Span

A causal chain occurs when a sentence describes A producing B, and B in turn producing C. The intermediate element B is simultaneously the effect of A and the cause of C. Tag B with the **CE** label rather than annotating it twice.

Consider the sentence:

> Sanctions reduced foreign investment, which in turn weakened the national currency.

Here, "sanctions" is a cause, "weakened the national currency" is an effect, and "reduced foreign investment" is a CE span. The annotator creates two relations: sanctions → reduced foreign investment, and reduced foreign investment → weakened the national currency. The CE tag makes the dual role explicit and lets the model recover the full chain rather than two disconnected pairs.

Use CE only when the same span genuinely fills both roles within the sentence. If a phrase looks like an intermediate but only one relation is actually expressed, tag it with the role the sentence supports (cause or effect), not CE.

### 5.3 Watch for Causality Indicators

Words such as *lead*, *cause*, *because*, *could lead*, *increase*, and *decrease* often signal a causal claim, and these explicit cues are easy to spot. Causality also surfaces without such markers, however. Compare "Celebrating the victories increases team morale!" with "Celebrating team victories is essential for maintaining high morale." Both sentences treat "celebrating the victories" as the cause and "team morale" as the effect, yet only the first uses an explicit causal indicator. Implicit causal statements appear frequently in social science texts, so read carefully and look beyond surface cues.

### 5.4 Correlation Does Not Equal Causation

Two variables that move together do not necessarily share a causal link. The following sentence describes a correlation rather than a causal relationship: "There is a strong and significant correlation that those subjects who burn money tend to be also those who expect their counterpart to burn theirs." Label such sentences as non-causal in Step 1.

### 5.5 Stay Objective

Annotate only what the sentence states. Set aside personal judgment and outside knowledge. Even when a sentence carries a strong sentiment, mark only the cause-effect pairs that the sentence itself expresses, and do not infer connections that the text does not support.

### 5.6 Order Does Not Matter

Causes always precede effects in the world, but writers do not always present them in that order. Both "Last night's storm resulted in the destroyed house" and "The destroyed house was the result of last night's storm" convey the same causal structure: the cause is "storm" and the effect is "destroyed house." Annotate the pair the same way regardless of the order in which the sentence presents them.

### 5.7 Code All Cause-Effect Pairs

A single sentence may contain several causal pairs. One cause can produce multiple effects, and one effect can follow from multiple causes. Annotate each pair separately. Consider the following sentence:

> In models based on income differences, if responders care for the utility of the other responder, then in addition to receiving disutility because the take authority has a higher income than they do, they will also receive disutility because the take authority has higher income than their friend does.

Here, the effect "disutility" connects to three distinct causes: "care for the utility of the other responder," "take authority has a higher income than they do," and "take authority has higher income than their friend does." Mark each connection.

## 6. Checklist

- Complete the three steps in order: sentence label → span marking → relation linking.
- Every cause requires at least one effect, and every effect requires at least one cause.
- Use the CE tag whenever a span functions as both effect and cause within the same sentence.
- Apply the span boundary rules in §5.1 consistently; prefer the minimal span that fully conveys the variable.
- Annotate based on the text, not on personal feelings or interpretations.
- Read the entire sentence before marking any element.
- Follow the definitions and guidelines consistently across all annotations.

## 7. Examples

| Sentence | Annotation |
|---|---|
| Permitting continuous rather than binary ''all-or-nothing'' contributions significantly increases contributions and facilitates provision. | **Cause:** continuous rather than binary ''all-or-nothing'' contributions → **Effect:** contributions <br> **Cause:** continuous rather than binary ''all-or-nothing'' contributions → **Effect:** provision |
| Conversely, when switching from short games to long games, participants immediately begin to cooperate at a high level. | **Cause:** switching from short games to long games → **Effect:** cooperate |
| However, there are also reasons to think that the treatment will lead to low contributions, since the reaction to the comprehension/advice combination may be defensive. | **Cause:** treatment → **Effect:** contributions <br> **Cause:** reaction to the comprehension/advice combination may be defensive → **Effect:** contributions |
| Sanctions reduced foreign investment, which in turn weakened the national currency and drove up inflation. | **Cause:** sanctions → **CE:** reduced foreign investment <br> **CE:** reduced foreign investment → **Effect:** weakened the national currency <br> **CE:** reduced foreign investment → **Effect:** drove up inflation |
| If the memory advantage for expectancy-incongruent information is abolished under cognitive load, our ability to successfully engage in social cooperation would be impaired because this type of memory is essential for correcting maladaptive behavior tendencies. | **Cause:** cognitive load → **CE:** memory advantage for expectancy-incongruent information <br> **CE:** memory advantage for expectancy-incongruent information → **Effect:** cooperation <br> **Cause:** memory → **Effect:** maladaptive behavior tendencies |
| On one hand, these findings partially confirm our speculation vis-à-vis the role of risk taking: reciprocators were not necessarily cooperators under conditions of uncertainty, as some of them chose to cooperate because they could bear a high risk in anticipating the counterpart's goodwill. | **Cause:** bear a high risk in anticipating the counterpart's goodwill → **Effect:** cooperate |
| Consider Pruitt and Kimmel's (1977) goal/expectation theory, which states that cooperation is more likely when trust is enhanced, but only if people also have a prosocial orientation; if the trust explanation is true, then people with a prosocial orientation should contribute more when group identity is salient. | **Cause:** trust → **Effect:** cooperation <br> **Cause:** prosocial orientation → **Effect:** cooperation <br> **Cause:** prosocial orientation → **Effect:** contribute <br> **Cause:** group identity → **Effect:** contribute |
| Based on the above reasoning, we predicted that individuals who anticipate pride about acting fairly would be more likely to divide resources between themselves and another in a fair way, whereas those who anticipate regret about acting fairly would be less likely to do so. | **Cause:** pride about acting fairly → **Effect:** divide resources between themselves and another in a fair way <br> **Cause:** regret about acting fairly → **Effect:** divide resources between themselves and another in a fair way |
| In models based on income differences, if responders care for the utility of the other responder, then in addition to receiving disutility because the take authority has a higher income than they do, they will also receive disutility because the take authority has higher income than their friend does. | **Cause:** care for the utility of the other responder → **Effect:** disutility <br> **Cause:** take authority has a higher income than they do → **Effect:** disutility <br> **Cause:** take authority has higher income than their friend does → **Effect:** disutility |
| In fact, a recent agent-based simulation also suggests that gossip-based partner selection increases cooperation, whereas the strategy to defect after knowing about free riders' reputation decreases cooperation. | **Cause:** gossip-based partner selection → **Effect:** cooperation <br> **Cause:** defect after knowing about free riders' reputation → **Effect:** cooperation |
| We found that communication still improves group performance even with increased difficulties in communication and limited information about the resource, but the level of benefit for having communication is reduced. | **Cause:** communication → **Effect:** group performance <br> **Cause:** difficulties in communication → **Effect:** communication <br> **Cause:** limited information about the resource → **Effect:** communication |
| If it is true that antisocial punishment is based on the intuitive system and especially likely executed by individuals with a proneness to sadistic tendencies, then the inhibition of the intuitive system should reduce antisocial punishment in individuals with a proneness to sadistic tendencies. | **Cause:** inhibition of the intuitive system → **Effect:** antisocial punishment <br> **Cause:** sadistic tendencies → **Effect:** antisocial punishment |
| In conservation psychology, the accumulated evidence indicates that people's conservation performance is strongly determined by normative prosocial influences. | **Cause:** normative prosocial influences → **Effect:** conservation performance |
| In a highly congested road, for example, at peak hours commuting car drivers are not willing to share the road or let other vehicles use the road space correctly, exacerbating the tragedy associated with the overuse of that road space. | **Cause:** congested road → **Effect:** willing to share the road <br> **Cause:** congested road → **Effect:** let other vehicles use the road space correctly <br> **Cause:** peak hours → **Effect:** willing to share the road <br> **Cause:** peak hours → **Effect:** let other vehicles use the road space correctly |
| In line with these predictions, Pellegrini and Long (2002) observed an initial increase in aggressive competition as adolescents moved from primary to secondary school, and bullying behavior appeared to mediate dominance status during this transition. | **Cause:** moved from primary to secondary school → **Effect:** aggressive competition <br> **Cause:** bullying behavior → **Effect:** dominance status |
| A natural explanation for the limited capability of a leader's signal to influence followers' beliefs under heterogeneity is the lack of followers' trust in the relevance of the signal to participants who do not share the religion of the leader. | **Cause:** leader's signal → **CE:** trust in the relevance of the signal <br> **CE:** trust in the relevance of the signal → **Effect:** followers' beliefs |
| Although punishment may be used to regulate the members of the in-group, the evidence presented here suggests that it is often driven by intergroup bias leading to harsher punishment of out-group members. | **Cause:** punishment → **Effect:** regulate the members of the in-group <br> **Cause:** intergroup bias → **Effect:** punishment <br> **Cause:** intergroup bias → **Effect:** punishment of out-group members |
| Of primary interest in this context is the latter interaction: subjects who had participated first in the Paddle Game declined less in cooperativeness than those who had not, and tit-for-tat subjects with prior Paddle Game experience increased more in cooperativeness over trials than those without such experience. | **Cause:** experience → **Effect:** cooperativeness |
| Violent media may cause viewers to perceive greater anger and less gratitude during a cooperative task and as such be less inclined to cooperate. | **Cause:** violent media → **Effect:** anger <br> **Cause:** violent media → **Effect:** gratitude <br> **Cause:** violent media → **Effect:** cooperate |
| In other words, acceptance would be more common if the subjects' expectations were met because, although a shared concept of fairness exists, fairness is sensitive to different factors (personal, situational, and social) that influence expectations and fairness perception. | **Cause:** expectations were met → **Effect:** acceptance <br> **Cause:** different factors (personal, situational, and social) → **Effect:** expectations <br> **Cause:** different factors (personal, situational, and social) → **Effect:** fairness perception |

## 8. Contact Information

**Rasoul Norouzi**
Email: [r.norouzinikjeh@tilburguniversity.edu](mailto:r.norouzinikjeh@tilburguniversity.edu)

## 9. References

Digitale, J. C., Martin, J. N., & Glymour, M. M. (2022). Tutorial on directed acyclic graphs. *Journal of Clinical Epidemiology*, *142*, 264–267.

Gross, N. (2018). The structure of causal chains. *Sociological Theory*, *36*(4), 343–367. https://doi.org/10.1177/0735275118811377

Halpern, J. Y. (2016). *Actual causality*. MIT Press.

Hartgerink, C. H. J., van Beest, I., Wicherts, J. M., & Williams, K. D. (2015). The ordinal effects of ostracism: A meta-analysis of 120 Cyberball studies. *PLoS ONE*, *10*(5), e0127002. https://doi.org/10.1371/journal.pone.0127002

Hedström, P., & Ylikoski, P. (2010). Causal mechanisms in the social sciences. *Annual Review of Sociology*, *36*(1), 49–67. https://doi.org/10.1146/annurev.soc.012809.102632

Moustakas, L. (2023). Social cohesion: Definitions, causes and consequences. *Encyclopedia*, *3*(3), 1028–1037.

Nakayama, H., Kubo, T., Kamura, J., Taniguchi, Y., & Liang, X. (2018). *Doccano: Text annotation tool for human* [Computer software]. https://github.com/doccano/doccano

Norouzi, R., Kleinberg, B., Vermunt, J., & others. (2024). *Capturing causal claims: A fine-tuned text mining model for extracting causal sentences from social science papers*.

Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*, *82*(4), 669–688.

Seki, K., & Mostafa, J. (2003). An approach to protein name extraction using heuristics and a dictionary. *Proceedings of the American Society for Information Science and Technology*, *40*(1), 71–77.

Shrier, I., & Platt, R. W. (2008). Reducing bias through directed acyclic graphs. *BMC Medical Research Methodology*, *8*, Article 70. https://doi.org/10.1186/1471-2288-8-70

Spadaro, G., Tiddi, I., Columbus, S., Jin, S., ten Teije, A., CoDa Team, & Balliet, D. (2022). The Cooperation Databank: Machine-readable science accelerates research synthesis. *Perspectives on Psychological Science*, *17*(5), 1472–1489.

Tsai, R. T.-H., Wu, S.-H., Chou, W.-C., Lin, Y.-C., He, D., Hsiang, J., Sung, T.-Y., & Hsu, W.-L. (2006). Various criteria in the evaluation of biomedical named entity recognition. *BMC Bioinformatics*, *7*, Article 92. https://doi.org/10.1186/1471-2105-7-92
