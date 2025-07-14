# %%
import json
import itertools

class KrippendorffSpanMatcher:
    def __init__(self, annotator_paths, targets=("cause", "effect")):
        """
        annotator_paths: dict mapping annotator names to file paths.
        targets: tuple of target labels to consider.
        
        Uses character offset-based partial overlap matching only.
        """
        self.annotator_paths = annotator_paths
        self.annotator_names = list(annotator_paths.keys())
        self.targets = targets

        # Process all annotator files once and store annotations.
        self.annotations_list = [self._process_annotations(path) for path in annotator_paths.values()]
        self.num_sentences = len(self.annotations_list[0])
    
    def _process_annotations(self, file_path):
        """
        Processes a JSONL annotation file.
        Each record is converted into a dictionary with:
          - 'entities': original entities with character offsets for span matching.
          - 'text': original sentence text.
        """
        processed = []
        with open(file_path, "r", encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                sentence = record.get("text", "")
                entities = record.get("entities", [])
                
                # Filter entities to keep only target labels (cause, effect)
                # Note: 'non-causal' is sentence-level labeling, not span-level
                filtered_entities = []
                for entity in entities:
                    label = entity.get("label")
                    if label in self.targets:  # Only keep cause/effect spans
                        filtered_entities.append(entity)
                
                processed.append({
                    "entities": filtered_entities,
                    "text": sentence,
                    "original_entities": entities  # Keep original for sentence-level causality detection
                })
        return processed

    def extract_spans(self, entities, target_label):
        """
        Extracts character offset-based spans for the target label.
        Returns a list of (start_offset, end_offset, text) tuples.
        """
        spans = []
        for entity in entities:
            if entity.get("label") == target_label:
                start = entity.get("start_offset")
                end = entity.get("end_offset")
                text = entity.get("text", "")
                spans.append((start, end, text))
        return spans

    def match_score(self, span1, span2):
        """
        Computes the matching score between two spans based on character offset partial overlap.
        Any character overlap returns 1.0 (full agreement), no overlap returns 0.0.
        
        Both span1 and span2 are lists of (start, end, text) tuples.
        """
        if not span1 and not span2:
            return 1.0  # Both empty = perfect match
        if not span1 or not span2:
            return 0.0  # One empty, one not = disagreement

        # Check character overlap - any overlap = perfect match
        for start1, end1, text1 in span1:
            for start2, end2, text2 in span2:
                # Check if spans overlap: max(start1, start2) < min(end1, end2)
                if max(start1, start2) < min(end1, end2):
                    return 1.0  # Any overlap = perfect match
        return 0.0  # No overlap = disagreement

    def relaxed_distance(self, span1, span2):
        """
        Converts the match score into a distance metric.
        Perfect match gives a distance of 0; no overlap gives 1.
        """
        return 1 - self.match_score(span1, span2)

    def average_pairwise_distance(self, spans):
        """
        Computes the average pairwise relaxed distance for a list of spans.
        """
        distances = [
            self.relaxed_distance(spans[i], spans[j])
            for i, j in itertools.combinations(range(len(spans)), 2)
        ]
        return sum(distances) / len(distances) if distances else 0.0

    def _get_sentence_causality_label(self, sentence_annotations):
        """
        Determines if a sentence is "causal" or "non-causal" based on its entities.
        A sentence is causal if it has any 'cause' or 'effect' labels.
        """
        for entity in sentence_annotations["entities"]:
            label = entity.get("label", "")
            if label != "non-causal" and (label == "cause" or label == "effect"):
                return "causal"
        return "non-causal"

    def _nominal_distance(self, label1, label2):
        """
        Nominal distance: 0 if labels are the same, 1 otherwise.
        """
        return 0.0 if label1 == label2 else 1.0

    def _compute_nominal_do(self, matrix):
        """
        Computes observed disagreement (D_o) for nominal data.
        Matrix is (num_sentences x num_annotators).
        """
        total_disagreement = 0.0
        num_sentences_with_pairs = 0
        for row in matrix: # Iterate through sentences
            if len(row) < 2: # Need at least two annotators for a pair
                continue
            
            # Calculate pairwise disagreement for this sentence
            sentence_disagreement = 0.0
            sentence_pairs_count = 0
            for i, j in itertools.combinations(range(len(row)), 2):
                sentence_disagreement += self._nominal_distance(row[i], row[j])
                sentence_pairs_count += 1
            
            if sentence_pairs_count > 0:
                total_disagreement += (sentence_disagreement / sentence_pairs_count)
                num_sentences_with_pairs += 1
        
        return total_disagreement / num_sentences_with_pairs if num_sentences_with_pairs > 0 else 0.0

    def _compute_nominal_de(self, matrix):
        """
        Computes expected disagreement (D_e) for nominal data.
        Matrix is (num_sentences x num_annotators).
        """
        all_labels = [label for row in matrix for label in row]
        if not all_labels:
            return 0.0

        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total_labels = len(all_labels)
        
        expected_disagreement = 0.0
        for label1, count1 in label_counts.items():
            for label2, count2 in label_counts.items():
                if label1 != label2:
                    expected_disagreement += (count1 / total_labels) * (count2 / total_labels)
        
        return expected_disagreement

    def compute_sentence_causality_alpha(self):
        """
        Computes Krippendorff's α for sentence-level causal labeling agreement.
        Uses a proper reliability matrix where rows=sentences, columns=annotators.
        Values: 1=causal, 0=non-causal
        """
        # Build reliability matrix: rows=sentences, columns=annotators
        reliability_matrix = []
        per_annotator_counts = {name: {"causal": 0, "non-causal": 0} for name in self.annotator_names}

        for sentence_idx in range(self.num_sentences):
            sentence_row = []
            for annotator_idx, annotations in enumerate(self.annotations_list):
                label = self._get_sentence_causality_label(annotations[sentence_idx])
                # Convert to binary: 1=causal, 0=non-causal
                binary_value = 1 if label == "causal" else 0
                sentence_row.append(binary_value)
                per_annotator_counts[self.annotator_names[annotator_idx]][label] += 1
            reliability_matrix.append(sentence_row)
        
        # Compute Krippendorff's α using the standard formula for interval data
        alpha = self._compute_krippendorff_alpha_interval(reliability_matrix)
        
        return alpha, per_annotator_counts

    def _compute_krippendorff_alpha_interval(self, reliability_matrix):
        """
        Computes Krippendorff's α for interval data (binary 0/1 values).
        reliability_matrix: list of lists where each inner list is [annotator1_value, annotator2_value, ...]
        """
        # Flatten all annotator-value pairs
        all_pairs = []
        for row_idx, row in enumerate(reliability_matrix):
            for annotator_idx, value in enumerate(row):
                if value is not None:  # Handle missing data
                    all_pairs.append((annotator_idx, value))
        
        if len(all_pairs) < 2:
            return 1.0  # Perfect agreement if insufficient data
        
        # Compute observed disagreement (D_o)
        total_disagreement = 0.0
        total_pairs = 0
        
        for row in reliability_matrix:
            # Only consider rows with at least 2 annotators
            valid_values = [v for v in row if v is not None]
            if len(valid_values) < 2:
                continue
                
            # Compute all pairwise differences within this sentence
            for i in range(len(valid_values)):
                for j in range(i + 1, len(valid_values)):
                    diff = abs(valid_values[i] - valid_values[j])
                    total_disagreement += diff
                    total_pairs += 1
        
        D_o = total_disagreement / total_pairs if total_pairs > 0 else 0.0
        
        # Compute expected disagreement (D_e)
        all_values = [pair[1] for pair in all_pairs]
        if len(all_values) < 2:
            return 1.0
            
        total_expected_disagreement = 0.0
        total_expected_pairs = 0
        
        for i in range(len(all_values)):
            for j in range(i + 1, len(all_values)):
                diff = abs(all_values[i] - all_values[j])
                total_expected_disagreement += diff
                total_expected_pairs += 1
        
        D_e = total_expected_disagreement / total_expected_pairs if total_expected_pairs > 0 else 0.0
        
        # Krippendorff's α = 1 - (D_o / D_e)
        alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
        return max(alpha, 0.0)

    def _get_agreed_causal_sentences_indices(self):
        """
        Identifies sentences where all annotators agreed the sentence is "causal".
        """
        agreed_indices = []
        for sentence_idx in range(self.num_sentences):
            all_causal = True
            for annotations in self.annotations_list:
                if self._get_sentence_causality_label(annotations[sentence_idx]) != "causal":
                    all_causal = False
                    break
            if all_causal:
                agreed_indices.append(sentence_idx)
        return agreed_indices

    def _compute_span_do(self, target_label, sentence_indices):
        """
        Computes the observed disagreement (D_o) across all annotators for a given target label,
        considering only specified sentence indices.
        """
        observed_distances = []
        for idx in sentence_indices:
            # Use character offset-based spans for all modes
            spans = [self.extract_spans(annotations[idx]["entities"], target_label)
                     for annotations in self.annotations_list]
            
            if len(spans) < 2: # Need at least two annotators for a pair
                continue

            distances = [
                self.relaxed_distance(spans[i], spans[j])
                for i, j in itertools.combinations(range(len(spans)), 2)
            ]
            if distances:
                observed_distances.append(sum(distances) / len(distances))
        return sum(observed_distances) / len(observed_distances) if observed_distances else 0.0

    def _compute_span_de(self, target_label, sentence_indices):
        """
        Computes the expected disagreement (D_e) by pooling all spans across specified sentences and annotators.
        """
        all_spans = []
        for annotations in self.annotations_list:
            for idx in sentence_indices:
                sentence = annotations[idx]
                span = self.extract_spans(sentence["entities"], target_label)
                all_spans.append(span)
        return self.average_pairwise_distance(all_spans)

    def compute_cause_span_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes Krippendorff's alpha for cause span agreement,
        optionally filtered by sentences agreed as causal.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)

        D_o = self._compute_span_do("cause", agreed_causal_sentence_indices)
        D_e = self._compute_span_de("cause", agreed_causal_sentence_indices)
        
        alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
        return max(alpha, 0.0)

    def compute_effect_span_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes Krippendorff's alpha for effect span agreement,
        optionally filtered by sentences agreed as causal.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)

        D_o = self._compute_span_do("effect", agreed_causal_sentence_indices)
        D_e = self._compute_span_de("effect", agreed_causal_sentence_indices)
        
        alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
        return max(alpha, 0.0)

    def _extract_links_for_sentence(self, sentence_data):
        """
        Extracts a set of (cause_span, effect_span) tuples for a single sentence.
        Spans are (start, end, text) tuples based on character offsets.
        """
        cause_spans = self.extract_spans(sentence_data["entities"], "cause")
        effect_spans = self.extract_spans(sentence_data["entities"], "effect")
        
        links = set()
        for c_span in cause_spans:
            for e_span in effect_spans:
                # Link is ((c_start, c_end, c_text), (e_start, e_end, e_text))
                links.add((c_span, e_span))
        return links

    def _link_match_score(self, link1, link2):
        """
        Checks if two links (cause_span, effect_span) match using character offset partial overlap.
        A link matches if both the cause and effect spans have partial overlap.
        
        Links are ((c_start, c_end, c_text), (e_start, e_end, e_text))
        """
        cause1_span = link1[0]
        effect1_span = link1[1]
        cause2_span = link2[0]
        effect2_span = link2[1]
        
        # Check if both cause and effect spans match using partial overlap
        cause_match = self.match_score([cause1_span], [cause2_span])
        effect_match = self.match_score([effect1_span], [effect2_span])
        
        return 1.0 if cause_match == 1.0 and effect_match == 1.0 else 0.0

    def compute_link_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes Krippendorff's α for cause-effect link agreement.
        Filters to sentences where all annotators agreed the sentence is causal
        and each annotator annotated at least one cause and one effect span.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)

        # Collect all link sets for valid sentences
        all_link_sets = []
        
        for sentence_idx in agreed_causal_sentence_indices:
            sentence_link_sets = []
            valid_sentence_for_links = True

            for annotator_idx, annotations in enumerate(self.annotations_list):
                sentence_data = annotations[sentence_idx]
                current_annotator_links = self._extract_links_for_sentence(sentence_data)
                
                if not current_annotator_links: # If an annotator has no cause or no effect spans, no links can be formed
                    valid_sentence_for_links = False
                    break
                sentence_link_sets.append(current_annotator_links)
            
            if valid_sentence_for_links and len(sentence_link_sets) >= 2:
                all_link_sets.append(sentence_link_sets)

        if not all_link_sets:
            return 0.0

        # Compute observed disagreement (D_o)
        total_disagreement = 0.0
        total_pairs = 0
        
        for sentence_link_sets in all_link_sets:
            for i, j in itertools.combinations(range(len(sentence_link_sets)), 2):
                set1 = sentence_link_sets[i]
                set2 = sentence_link_sets[j]
                
                # Compute link set distance using the current matching mode
                distance = self._link_set_distance(set1, set2)
                total_disagreement += distance
                total_pairs += 1
        
        D_o = total_disagreement / total_pairs if total_pairs > 0 else 0.0
        
        # Compute expected disagreement (D_e)
        all_links_pooled = [link for sentence_sets in all_link_sets for link_set in sentence_sets for link in link_set]
        D_e = self._compute_link_de(all_links_pooled)
        
        alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
        return max(alpha, 0.0)

    def _link_set_distance(self, set1, set2):
        """
        Computes distance between two link sets using character offset partial overlap.
        Returns proportion of links that don't match.
        """
        if not set1 and not set2:
            return 0.0  # Both empty = perfect match
        if not set1 or not set2:
            return 1.0  # One empty, one not = complete disagreement
        
        # Find matching links
        matched_count = 0
        matched_links_set2 = set()
        
        for link1 in set1:
            for link2 in set2:
                if link2 not in matched_links_set2 and self._link_match_score(link1, link2) == 1.0:
                    matched_count += 1
                    matched_links_set2.add(link2)
                    break
        
        # Symmetric distance: consider both directions
        total_links = len(set1) + len(set2)
        matched_links = 2 * matched_count  # Count matches from both sides
        
        return 1.0 - (matched_links / total_links) if total_links > 0 else 0.0

    def _compute_link_de(self, all_links):
        """
        Computes expected disagreement for link data by comparing all pairs.
        """
        if len(all_links) < 2:
            return 0.0
        
        total_distance = 0.0
        total_pairs = 0
        
        for i, j in itertools.combinations(range(len(all_links)), 2):
            link1 = all_links[i]
            link2 = all_links[j]
            # Distance is 0 if links match perfectly, 1 if they don't match at all
            distance = 1.0 - self._link_match_score(link1, link2)
            total_distance += distance
            total_pairs += 1
        
        return total_distance / total_pairs if total_pairs > 0 else 0.0

    def run_all(self):
        """
        Executes all three agreement tasks and reports the scores.
        """
        results = {}

        print("--- Task 1: Overall Sentence Agreement ---")
        sentence_alpha, per_annotator_counts = self.compute_sentence_causality_alpha()
        results["sentence_causality_alpha"] = sentence_alpha
        results["sentence_causality_counts"] = per_annotator_counts
        
        # Get detailed agreement statistics
        agreement_stats = self.get_sentence_agreement_statistics()
        results["agreement_statistics"] = agreement_stats
        
        print(f"Krippendorff's Alpha (Overall Agreement): {sentence_alpha:.4f}")
        print(f"Overall Agreement Rate: {agreement_stats['overall_agreement_rate']:.1f}% ({agreement_stats['total_agreements']}/{agreement_stats['total_sentences']} sentences)")
        print(f"  • All annotators agreed 'causal': {agreement_stats['causal_agreements']} sentences")
        print(f"  • All annotators agreed 'non-causal': {agreement_stats['non_causal_agreements']} sentences")
        print(f"  • Mixed disagreements: {agreement_stats['mixed_disagreements']} sentences")
        print("Note: Both causal and non-causal agreements contribute positively to overall agreement.")

        agreed_causal_sentences = self._get_agreed_causal_sentences_indices()
        results["agreed_causal_sentences_count"] = len(agreed_causal_sentences)
        print(f"Sentences where all annotators agreed on 'causal' (for span/link analysis): {len(agreed_causal_sentences)}")

        print(f"\n--- Task 2: Cause and Effect Span Agreement (Character Offset Partial Overlap) ---")
        cause_alpha = self.compute_cause_span_alpha(agreed_causal_sentences)
        results["cause_span_alpha"] = cause_alpha
        print(f"Krippendorff's Alpha (Cause Spans, filtered): {cause_alpha:.4f}")

        effect_alpha = self.compute_effect_span_alpha(agreed_causal_sentences)
        results["effect_span_alpha"] = effect_alpha
        print(f"Krippendorff's Alpha (Effect Spans, filtered): {effect_alpha:.4f}")

        print(f"\n--- Task 3: Cause-Effect Link Agreement (Character Offset Partial Overlap) ---")
        
        # Task 3A: Full Link Agreement (Structural Definition)
        link_alpha_full = self.compute_link_alpha(agreed_causal_sentences)
        results["link_alpha_full_structure"] = link_alpha_full
        print(f"Task 3A - Full Structure Agreement: {link_alpha_full:.4f}")
        print("  (Do annotators agree on the full causal structure including spans and links?)")
        
        # Task 3B: Link Agreement over Shared Spans (Span-Controlled Mapping)
        link_alpha_filtered, filtered_stats = self.compute_filtered_link_alpha(agreed_causal_sentences)
        results["link_alpha_filtered_mapping"] = link_alpha_filtered
        results["filtered_link_stats"] = filtered_stats
        print(f"Task 3B - Filtered Mapping Agreement: {link_alpha_filtered:.4f}")
        print(f"  (Assuming annotators agree on spans, do they agree on how to link them?)")
        print(f"  Sentences analyzed: {filtered_stats['sentences_analyzed']}/{filtered_stats['total_agreed_causal_sentences']}")

        # Pairwise analysis
        print("\n" + "=" * 60)
        print("PAIRWISE AGREEMENT ANALYSIS")
        print("=" * 60)
        
        print("\n--- Pairwise Sentence Causality Agreement ---")
        pairwise_sentence = self.compute_pairwise_sentence_causality_alpha()
        results["pairwise_sentence"] = pairwise_sentence
        for pair, data in pairwise_sentence.items():
            print(f"{pair}: α={data['alpha']:.4f} (agreement: {data['agreement_rate']:.1f}%, {data['agreements']}/{data['total_sentences']})")
        
        print("\n--- Pairwise Cause Span Agreement (Filtered) ---")
        pairwise_cause = self.compute_pairwise_span_alpha("cause", agreed_causal_sentences)
        results["pairwise_cause"] = pairwise_cause
        for pair, data in pairwise_cause.items():
            print(f"{pair}: α={data['alpha']:.4f} (sentences: {data['sentences_analyzed']})")
        
        print("\n--- Pairwise Effect Span Agreement (Filtered) ---")
        pairwise_effect = self.compute_pairwise_span_alpha("effect", agreed_causal_sentences)
        results["pairwise_effect"] = pairwise_effect
        for pair, data in pairwise_effect.items():
            print(f"{pair}: α={data['alpha']:.4f} (sentences: {data['sentences_analyzed']})")
        
        print("\n--- Pairwise Link Agreement (Filtered) ---")
        print("Task 3A - Full Structure:")
        pairwise_link_full = self.compute_pairwise_link_alpha(agreed_causal_sentences)
        results["pairwise_link_full"] = pairwise_link_full
        for pair, data in pairwise_link_full.items():
            print(f"  {pair}: α={data['alpha']:.4f} (sentences: {data['sentences_analyzed']})")
        
        print("Task 3B - Filtered Mapping:")
        pairwise_link_filtered = self.compute_pairwise_filtered_link_alpha(agreed_causal_sentences)
        results["pairwise_link_filtered"] = pairwise_link_filtered
        for pair, data in pairwise_link_filtered.items():
            print(f"  {pair}: α={data['alpha']:.4f} (sentences: {data['sentences_analyzed']})")

        return results
    def get_unique_labels(self):
        """
        Computes the unique set of labels for each annotator based on entities.
        Returns a dictionary mapping annotator names to their set of unique labels.
        """
        unique_labels = {}
        for name, annotations in zip(self.annotator_names, self.annotations_list):
            labels_set = set()
            for sentence in annotations:
                for entity in sentence["entities"]:
                    labels_set.add(entity.get("label", ""))
            unique_labels[name] = labels_set
        return unique_labels

    def compute_pairwise_sentence_causality_alpha(self):
        """
        Computes pairwise Krippendorff's α for sentence-level causality between each pair of annotators.
        Returns a dictionary with pairwise results.
        """
        pairwise_results = {}
        
        # Get all annotator pairs
        for i, j in itertools.combinations(range(len(self.annotator_names)), 2):
            annotator1 = self.annotator_names[i]
            annotator2 = self.annotator_names[j]
            pair_name = f"{annotator1} vs {annotator2}"
            
            # Build reliability matrix for just these two annotators
            reliability_matrix = []
            for sentence_idx in range(self.num_sentences):
                label1 = self._get_sentence_causality_label(self.annotations_list[i][sentence_idx])
                label2 = self._get_sentence_causality_label(self.annotations_list[j][sentence_idx])
                
                binary1 = 1 if label1 == "causal" else 0
                binary2 = 1 if label2 == "causal" else 0
                
                reliability_matrix.append([binary1, binary2])
            
            # Compute α for this pair
            alpha = self._compute_krippendorff_alpha_interval(reliability_matrix)
            
            # Count agreements and disagreements
            agreements = sum(1 for row in reliability_matrix if row[0] == row[1])
            disagreements = len(reliability_matrix) - agreements
            agreement_rate = agreements / len(reliability_matrix) * 100
            
            pairwise_results[pair_name] = {
                "alpha": alpha,
                "agreements": agreements,
                "disagreements": disagreements,
                "agreement_rate": agreement_rate,
                "total_sentences": len(reliability_matrix)
            }
        
        return pairwise_results

    def compute_pairwise_span_alpha(self, target_label, agreed_causal_sentence_indices=None):
        """
        Computes pairwise Krippendorff's α for span agreement between each pair of annotators.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)
            
        pairwise_results = {}
        
        for i, j in itertools.combinations(range(len(self.annotator_names)), 2):
            annotator1 = self.annotator_names[i]
            annotator2 = self.annotator_names[j]
            pair_name = f"{annotator1} vs {annotator2}"
            
            # Compute pairwise span disagreement for this target label
            observed_distances = []
            for idx in agreed_causal_sentence_indices:
                spans1 = self.extract_spans(self.annotations_list[i][idx]["entities"], target_label)
                spans2 = self.extract_spans(self.annotations_list[j][idx]["entities"], target_label)
                
                distance = self.relaxed_distance(spans1, spans2)
                observed_distances.append(distance)
            
            D_o = sum(observed_distances) / len(observed_distances) if observed_distances else 0.0
            
            # Compute expected disagreement for this pair
            all_spans_pair = []
            for annotator_idx in [i, j]:
                for idx in agreed_causal_sentence_indices:
                    sentence = self.annotations_list[annotator_idx][idx]
                    span = self.extract_spans(sentence["entities"], target_label)
                    all_spans_pair.append(span)
            
            D_e = self.average_pairwise_distance(all_spans_pair)
            
            alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
            alpha = max(alpha, 0.0)
            
            pairwise_results[pair_name] = {
                "alpha": alpha,
                "observed_disagreement": D_o,
                "expected_disagreement": D_e,
                "sentences_analyzed": len(observed_distances)
            }
        
        return pairwise_results

    def compute_pairwise_link_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes pairwise Krippendorff's α for link agreement between each pair of annotators.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)
            
        pairwise_results = {}
        
        for i, j in itertools.combinations(range(len(self.annotator_names)), 2):
            annotator1 = self.annotator_names[i]
            annotator2 = self.annotator_names[j]
            pair_name = f"{annotator1} vs {annotator2}"
            
            # Collect link sets for valid sentences for this pair
            pair_link_sets = []
            
            for sentence_idx in agreed_causal_sentence_indices:
                sentence_data1 = self.annotations_list[i][sentence_idx]
                sentence_data2 = self.annotations_list[j][sentence_idx]
                
                links1 = self._extract_links_for_sentence(sentence_data1)
                links2 = self._extract_links_for_sentence(sentence_data2)
                
                # Only include if both annotators have links
                if links1 and links2:
                    pair_link_sets.append([links1, links2])
            
            if not pair_link_sets:
                pairwise_results[pair_name] = {
                    "alpha": 0.0,
                    "observed_disagreement": 1.0,
                    "expected_disagreement": 1.0,
                    "sentences_analyzed": 0
                }
                continue
            
            # Compute observed disagreement (D_o)
            total_disagreement = 0.0
            for link_sets in pair_link_sets:
                distance = self._link_set_distance(link_sets[0], link_sets[1])
                total_disagreement += distance
            
            D_o = total_disagreement / len(pair_link_sets)
            
            # Compute expected disagreement (D_e) for this pair
            all_links_pair = [link for link_sets in pair_link_sets for link_set in link_sets for link in link_set]
            D_e = self._compute_link_de(all_links_pair)
            
            alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
            alpha = max(alpha, 0.0)
            
            pairwise_results[pair_name] = {
                "alpha": alpha,
                "observed_disagreement": D_o,
                "expected_disagreement": D_e,
                "sentences_analyzed": len(pair_link_sets)
            }
        
        return pairwise_results

    def compute_pairwise_filtered_link_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes pairwise Krippendorff's α for filtered link agreement between each pair of annotators.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)
            
        pairwise_results = {}
        
        for i, j in itertools.combinations(range(len(self.annotator_names)), 2):
            annotator1 = self.annotator_names[i]
            annotator2 = self.annotator_names[j]
            pair_name = f"{annotator1} vs {annotator2}"
            
            # Collect filtered link sets for valid sentences for this pair
            pair_filtered_link_sets = []
            
            for sentence_idx in agreed_causal_sentence_indices:
                annotator_filtered_links, _ = self._extract_filtered_links_for_sentence(sentence_idx)
                
                if i in annotator_filtered_links and j in annotator_filtered_links:
                    links1 = annotator_filtered_links[i]
                    links2 = annotator_filtered_links[j]
                    pair_filtered_link_sets.append([links1, links2])
            
            if not pair_filtered_link_sets:
                pairwise_results[pair_name] = {
                    "alpha": 0.0,
                    "observed_disagreement": 1.0,
                    "expected_disagreement": 1.0,
                    "sentences_analyzed": 0
                }
                continue
            
            # Compute observed disagreement (D_o)
            total_disagreement = 0.0
            for link_sets in pair_filtered_link_sets:
                distance = self._filtered_link_set_distance(link_sets[0], link_sets[1])
                total_disagreement += distance
            
            D_o = total_disagreement / len(pair_filtered_link_sets)
            
            # Compute expected disagreement (D_e) for this pair
            all_filtered_links_pair = [
                link for link_sets in pair_filtered_link_sets 
                for link_set in link_sets 
                for link in link_set
            ]
            D_e = self._compute_filtered_link_de(all_filtered_links_pair)
            
            alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
            alpha = max(alpha, 0.0)
            
            pairwise_results[pair_name] = {
                "alpha": alpha,
                "observed_disagreement": D_o,
                "expected_disagreement": D_e,
                "sentences_analyzed": len(pair_filtered_link_sets)
            }
        
        return pairwise_results

    def get_sentence_agreement_statistics(self):
        """
        Computes detailed agreement statistics showing that both causal and non-causal 
        agreements are positive contributions to overall agreement.
        """
        # Build reliability matrix
        reliability_matrix = []
        for sentence_idx in range(self.num_sentences):
            sentence_row = []
            for annotator_idx, annotations in enumerate(self.annotations_list):
                label = self._get_sentence_causality_label(annotations[sentence_idx])
                binary_value = 1 if label == "causal" else 0
                sentence_row.append(binary_value)
            reliability_matrix.append(sentence_row)
        
        # Count agreement patterns
        causal_agreements = sum(1 for row in reliability_matrix if row == [1, 1, 1])
        non_causal_agreements = sum(1 for row in reliability_matrix if row == [0, 0, 0])
        mixed_disagreements = len(reliability_matrix) - causal_agreements - non_causal_agreements
        
        total_agreements = causal_agreements + non_causal_agreements
        overall_agreement_rate = total_agreements / len(reliability_matrix) * 100
        
        return {
            "causal_agreements": causal_agreements,
            "non_causal_agreements": non_causal_agreements,
            "total_agreements": total_agreements,
            "mixed_disagreements": mixed_disagreements,
            "total_sentences": len(reliability_matrix),
            "overall_agreement_rate": overall_agreement_rate
        }
    def _find_shared_spans(self, sentence_idx, target_label):
        """
        Finds shared spans across annotators for a specific sentence and label.
        Returns a mapping from shared span IDs to the actual spans from each annotator.
        
        Returns:
            shared_spans_mapping: dict where keys are span_ids (e.g., 'C1', 'E1') 
                                 and values are lists of (annotator_idx, span) tuples
        """
        # Collect all spans from all annotators for this sentence and label
        all_annotator_spans = []
        for annotator_idx, annotations in enumerate(self.annotations_list):
            spans = self.extract_spans(annotations[sentence_idx]["entities"], target_label)
            all_annotator_spans.append((annotator_idx, spans))
        
        # Group overlapping spans into shared span clusters
        shared_spans_mapping = {}
        span_counter = 1
        prefix = 'C' if target_label == 'cause' else 'E'
        
        for annotator_idx, spans in all_annotator_spans:
            for span in spans:
                # Check if this span overlaps with any existing shared span
                matched_shared_id = None
                
                for shared_id, shared_span_list in shared_spans_mapping.items():
                    # Check if current span overlaps with any span in this shared group
                    for _, existing_span in shared_span_list:
                        if self._spans_overlap(span, existing_span):
                            matched_shared_id = shared_id
                            break
                    if matched_shared_id:
                        break
                
                if matched_shared_id:
                    # Add to existing shared span group
                    shared_spans_mapping[matched_shared_id].append((annotator_idx, span))
                else:
                    # Create new shared span group
                    shared_id = f"{prefix}{span_counter}"
                    shared_spans_mapping[shared_id] = [(annotator_idx, span)]
                    span_counter += 1
        
        # Filter to keep only shared spans (those with at least 2 annotators)
        filtered_mapping = {
            shared_id: span_list 
            for shared_id, span_list in shared_spans_mapping.items() 
            if len(set(annotator_idx for annotator_idx, _ in span_list)) >= 2
        }
        
        return filtered_mapping

    def _spans_overlap(self, span1, span2):
        """
        Check if two spans overlap using character offsets.
        """
        start1, end1, _ = span1
        start2, end2, _ = span2
        return max(start1, start2) < min(end1, end2)

    def _extract_filtered_links_for_sentence(self, sentence_idx):
        """
        Extracts links between shared spans for a sentence.
        
        Returns:
            annotator_filtered_links: dict mapping annotator_idx to set of (shared_cause_id, shared_effect_id) tuples
            shared_span_info: dict with mapping information for debugging
        """
        # Find shared cause and effect spans
        shared_causes = self._find_shared_spans(sentence_idx, "cause")
        shared_effects = self._find_shared_spans(sentence_idx, "effect")
        
        if not shared_causes or not shared_effects:
            return {}, {"shared_causes": shared_causes, "shared_effects": shared_effects}
        
        # Create mapping from (annotator_idx, span) to shared_id
        span_to_shared_id = {}
        for shared_id, span_list in shared_causes.items():
            for annotator_idx, span in span_list:
                span_to_shared_id[(annotator_idx, span)] = shared_id
        for shared_id, span_list in shared_effects.items():
            for annotator_idx, span in span_list:
                span_to_shared_id[(annotator_idx, span)] = shared_id
        
        # Extract filtered links for each annotator
        annotator_filtered_links = {}
        
        for annotator_idx, annotations in enumerate(self.annotations_list):
            sentence_data = annotations[sentence_idx]
            original_links = self._extract_links_for_sentence(sentence_data)
            
            filtered_links = set()
            for cause_span, effect_span in original_links:
                # Map to shared span IDs if they exist
                cause_shared_id = span_to_shared_id.get((annotator_idx, cause_span))
                effect_shared_id = span_to_shared_id.get((annotator_idx, effect_span))
                
                if cause_shared_id and effect_shared_id:
                    filtered_links.add((cause_shared_id, effect_shared_id))
            
            if filtered_links:  # Only include annotators with filtered links
                annotator_filtered_links[annotator_idx] = filtered_links
        
        shared_span_info = {
            "shared_causes": shared_causes,
            "shared_effects": shared_effects,
            "span_to_shared_id": span_to_shared_id
        }
        
        return annotator_filtered_links, shared_span_info

    def _filtered_link_set_distance(self, set1, set2):
        """
        Computes distance between two filtered link sets (using shared span IDs).
        Uses the same symmetric distance formula as the full structure version.
        """
        if not set1 and not set2:
            return 0.0  # Both empty = perfect match
        if not set1 or not set2:
            return 1.0  # One empty, one not = complete disagreement
        
        # Find matching links (exact match on shared span IDs)
        matched_count = len(set1.intersection(set2))
        
        # Symmetric distance: consider both directions
        total_links = len(set1) + len(set2)
        matched_links = 2 * matched_count  # Count matches from both sides
        
        return 1.0 - (matched_links / total_links) if total_links > 0 else 0.0

    def compute_filtered_link_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes Krippendorff's α for cause-effect link agreement over shared spans only.
        This version focuses on linking decisions given shared understanding of spans.
        
        Task 3B: "Assuming annotators agree on spans, do they agree on how to link them?"
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)

        # Collect all filtered link sets for valid sentences
        all_filtered_link_sets = []
        sentences_with_shared_spans = 0
        
        for sentence_idx in agreed_causal_sentence_indices:
            annotator_filtered_links, shared_span_info = self._extract_filtered_links_for_sentence(sentence_idx)
            
            # Only include sentences where at least 2 annotators have filtered links
            if len(annotator_filtered_links) >= 2:
                # Convert to list format for consistency
                sentence_filtered_sets = []
                for annotator_idx in sorted(annotator_filtered_links.keys()):
                    sentence_filtered_sets.append(annotator_filtered_links[annotator_idx])
                
                all_filtered_link_sets.append(sentence_filtered_sets)
                sentences_with_shared_spans += 1

        if not all_filtered_link_sets:
            return 0.0, {"sentences_analyzed": 0, "sentences_with_shared_spans": 0}

        # Compute observed disagreement (D_o)
        total_disagreement = 0.0
        total_pairs = 0
        
        for sentence_filtered_sets in all_filtered_link_sets:
            for i, j in itertools.combinations(range(len(sentence_filtered_sets)), 2):
                set1 = sentence_filtered_sets[i]
                set2 = sentence_filtered_sets[j]
                
                distance = self._filtered_link_set_distance(set1, set2)
                total_disagreement += distance
                total_pairs += 1
        
        D_o = total_disagreement / total_pairs if total_pairs > 0 else 0.0
        
        # Compute expected disagreement (D_e)
        all_filtered_links_pooled = [
            link for sentence_sets in all_filtered_link_sets 
            for link_set in sentence_sets 
            for link in link_set
        ]
        D_e = self._compute_filtered_link_de(all_filtered_links_pooled)
        
        alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
        alpha = max(alpha, 0.0)
        
        stats = {
            "sentences_analyzed": len(all_filtered_link_sets),
            "sentences_with_shared_spans": sentences_with_shared_spans,
            "total_agreed_causal_sentences": len(agreed_causal_sentence_indices)
        }
        
        return alpha, stats

    def _compute_filtered_link_de(self, all_filtered_links):
        """
        Computes expected disagreement for filtered link data by comparing all pairs.
        """
        if len(all_filtered_links) < 2:
            return 0.0
        
        total_distance = 0.0
        total_pairs = 0
        
        for i, j in itertools.combinations(range(len(all_filtered_links)), 2):
            link1 = all_filtered_links[i]
            link2 = all_filtered_links[j]
            # Distance is 0 if links match exactly (shared span IDs), 1 if they don't
            distance = 0.0 if link1 == link2 else 1.0
            total_distance += distance
            total_pairs += 1
        
        return total_distance / total_pairs if total_pairs > 0 else 0.0

    def compute_pairwise_filtered_link_alpha(self, agreed_causal_sentence_indices=None):
        """
        Computes pairwise Krippendorff's α for filtered link agreement between each pair of annotators.
        """
        if agreed_causal_sentence_indices is None:
            agreed_causal_sentence_indices = range(self.num_sentences)
            
        pairwise_results = {}
        
        for i, j in itertools.combinations(range(len(self.annotator_names)), 2):
            annotator1 = self.annotator_names[i]
            annotator2 = self.annotator_names[j]
            pair_name = f"{annotator1} vs {annotator2}"
            
            # Collect filtered link sets for valid sentences for this pair
            pair_filtered_link_sets = []
            
            for sentence_idx in agreed_causal_sentence_indices:
                annotator_filtered_links, _ = self._extract_filtered_links_for_sentence(sentence_idx)
                
                if i in annotator_filtered_links and j in annotator_filtered_links:
                    links1 = annotator_filtered_links[i]
                    links2 = annotator_filtered_links[j]
                    pair_filtered_link_sets.append([links1, links2])
            
            if not pair_filtered_link_sets:
                pairwise_results[pair_name] = {
                    "alpha": 0.0,
                    "observed_disagreement": 1.0,
                    "expected_disagreement": 1.0,
                    "sentences_analyzed": 0
                }
                continue
            
            # Compute observed disagreement (D_o)
            total_disagreement = 0.0
            for link_sets in pair_filtered_link_sets:
                distance = self._filtered_link_set_distance(link_sets[0], link_sets[1])
                total_disagreement += distance
            
            D_o = total_disagreement / len(pair_filtered_link_sets)
            
            # Compute expected disagreement (D_e) for this pair
            all_filtered_links_pair = [
                link for link_sets in pair_filtered_link_sets 
                for link_set in link_sets 
                for link in link_set
            ]
            D_e = self._compute_filtered_link_de(all_filtered_links_pair)
            
            alpha = 1 - (D_o / D_e) if D_e != 0 else 1.0
            alpha = max(alpha, 0.0)
            
            pairwise_results[pair_name] = {
                "alpha": alpha,
                "observed_disagreement": D_o,
                "expected_disagreement": D_e,
                "sentences_analyzed": len(pair_filtered_link_sets)
            }
        
        return pairwise_results
