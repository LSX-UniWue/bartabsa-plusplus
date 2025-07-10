from collections import Counter
from logging import getLogger

import torch

logger = getLogger("lightning.pytorch")


class TripletMetric:
    """
    heavily based on https://github.com/yhcc/BARTABSA/blob/main/peng/model/metrics.py
    """

    def __init__(self, eos_token_pointer_id: int, num_labels: int):
        self.eos_token_pointer_id = eos_token_pointer_id  # =0
        self.pad_token_id = 0  # TODO: Dont hardcode this
        self.num_labels = num_labels
        self.word_start_index = num_labels + 1  # +1, shift for eos

        # Initialize the plethora of metrics tracked
        self.reset()

    def reset(self):
        # Aspect + Opinion
        self.ae_oe_fp = self.ae_oe_fn = self.ae_oe_tp = 0
        # Aspect + Sentiment
        self.ae_sc_fp = self.ae_sc_fn = self.ae_sc_tp = 0
        # Triplet Extraction
        self.triplet_fp = self.triplet_fn = self.triplet_tp = 0
        # Total number of tokens processed
        self.total = self.exact_matches = self.invalid = 0

    def evaluate(self, pred: torch.Tensor, tgt_tokens: torch.Tensor):
        """
        Evaluate the predicted tokens against the target tokens

        Args:
            pred (torch.Tensor): The predicted tokens. It is assumed to be a 2D tensor of shape (batch_size, max_seq_len).
            tgt_tokens (torch.Tensor): The target tokens. It is assumed to be a 2D tensor of shape (batch_size, max_seq_len).
            tgt_tokens (torch.Tensor): The target tokens. It is assumed to be a 2D tensor of shape (batch_size, max_seq_len).
        """
        self.total += len(pred)
        # Mask with 1s for the positions after the eos token
        pred_eos_mask = pred.flip(1).eq(self.eos_token_pointer_id).cumsum(1).long()
        target_eos_mask = tgt_tokens.flip(1).eq(self.eos_token_pointer_id).cumsum(1).long()

        # Get the sequence lengths
        pred_seq_len = pred_eos_mask.eq(pred_eos_mask[:, -1:]).sum(1)
        pred_seq_len = (pred_seq_len - 1).tolist()  # -1 to exclude the eos token itself
        target_seq_len = target_eos_mask.eq(target_eos_mask[:, -1:]).sum(1)
        target_seq_len = (target_seq_len - 1).tolist()  # -1 to exclude the eos token itself

        target_spans = self._get_target_span(tgt_tokens)
        pred_spans = []
        for i, (target_seq, pred_seq) in enumerate(zip(target_spans, pred.tolist())):
            pred_seq = pred_seq[: pred_seq_len[i]]

            # If the sequences have the same length, we can directly compare them for an exact match
            if (pred_seq_len[i] == target_seq_len[i]) and (
                tgt_tokens[i, : target_seq_len[i]].eq(pred[i, : pred_seq_len[i]]).sum().item() == target_seq_len[i]
            ):
                self.exact_matches += 1

            is_invalid = False
            triplets = []
            temp_triplet = []
            if len(pred_seq) > 0:
                for predicted_token in pred_seq:
                    temp_triplet.append(predicted_token)

                    # If we have a special token, we can check if we have a complete and valid triplet
                    if predicted_token < self.word_start_index:
                        if len(temp_triplet) == 5 and temp_triplet[0] <= temp_triplet[1] and temp_triplet[2] <= temp_triplet[3]:
                            triplets.append(tuple(temp_triplet))
                        else:
                            is_invalid = True
                        temp_triplet = []
            pred_spans.append(triplets.copy())
            if is_invalid:
                self.invalid += 1

            # Calculate the TP, FP, FN for the Aspect + Sentiment task
            oe_ae_target = [tuple(triplet[:4]) for triplet in target_seq]
            oe_ae_pred = [tuple(triplet[:4]) for triplet in triplets]
            oe_ae_target_counter = Counter(oe_ae_target)
            oe_ae_pred_counter = Counter(oe_ae_pred)
            tp, fp, fn = self._get_tp_fp_fn(set(list(oe_ae_pred_counter.keys())), set(list(oe_ae_target_counter.keys())))
            self.ae_oe_tp += tp
            self.ae_oe_fp += fp
            self.ae_oe_fn += fn

            # Calculate the TP, FP, FN for the Aspect + Sentiment task
            ae_sc_target = {(triplet[0], triplet[1], triplet[4]) for triplet in target_seq}
            ae_sc_pred = {(triplet[0], triplet[1], triplet[4]) for triplet in triplets}
            asts = set([tuple(t) for t in ae_sc_target])
            asps = set(ae_sc_pred)
            for p in list(asps):  # pairs is a 5-tuple
                if p in asts:
                    asts.remove(p)
                    self.ae_sc_tp += 1
                else:
                    self.ae_sc_fp += 1
            self.ae_sc_fn += len(asts)

            # Calculate the TP, FP, FN for the Triplet Extraction task
            target_list = [tuple(triplet) for triplet in target_seq]
            pred_list = [tuple(triplet) for triplet in triplets]
            target_set = set(target_list)
            pred_set = set(pred_list)
            for p in list(pred_set):
                if p in target_set:
                    target_set.remove(p)
                    self.triplet_tp += 1
                else:
                    self.triplet_fp += 1
            self.triplet_fn += len(target_set)

    def _get_target_span(self, tgt_tokens: torch.Tensor) -> list[list[tuple[int, ...]]]:
        """
        Extract target spans from the (batched) target tokens.

        Args:
            target (torch.Tensor): The tensor of target tokens. It is assumed to be a 2D tensor of shape (batch_size, max_seq_len).

        Returns:
            list[list[tuple]]: A list of lists containing tuples of target spans.
                               Each inner list corresponds to one sequence in the batch.
        """
        # Check that we have a valid tensor
        assert tgt_tokens.dim() == 2, f"Expected 2D tensor, got {tgt_tokens.dim()}"

        target_span = []
        # Iterate over each sequence in the batch
        for tokens in tgt_tokens:
            tokens_span = []
            idx = 0  # Start index for token slicing
            while idx < len(tokens) and tokens[idx] != self.eos_token_pointer_id:
                # Ensure that we have at least 5 tokens to form a complete triplet
                if idx + 5 <= len(tokens) and self.eos_token_pointer_id not in tokens[idx : idx + 5]:
                    tokens_span.append(tuple(tokens[idx : idx + 5].tolist()))
                idx += 5  # Move to the next potential triplet

            target_span.append(tokens_span)  # Append the spans found in the current sequence

        return target_span

    def _get_tp_fp_fn(self, pred, target) -> tuple[int, int, int]:
        """
        Calculate the number of true positives, false positives, and false negatives.

        Args:
            pred (set[tuple]): The set of predicted triplets.
            target (set[tuple]): The set of target triplets.

        Returns:
            tuple[int, int, int]: The number of true positives, false positives, and false negatives.
        """
        pred = pred.copy()
        tp = fp = fn = 0
        if isinstance(target, set):
            target = {key: 1 for key in list(target)}
        if isinstance(pred, set):
            pred = {key: 1 for key in list(pred)}
        for key in target.keys():
            t_num = target[key]
            if key not in pred:
                p_num = 0
            else:
                p_num = pred[key]
            tp += min(p_num, t_num)
            fp += max(p_num - t_num, 0)
            fn += max(t_num - p_num, 0)
            if key in pred:
                pred.pop(key)
        fp += sum(pred.values())
        return tp, fn, fp

    def get_metrics(self, reset=False):
        """
        Compute and retrieve the accumulated metric results.

        Args:
        reset (bool): If True, reset the metric counters after retrieval.

        Returns:
        dict[str, float]: A dictionary containing the computed metrics.
        """
        results: dict[str, float] = {}

        # Calculate the Aspect + Opinion metrics
        results["ae_oe_f1"], results["ae_oe_prec"], results["ae_oe_rec"] = self._compute_metrics(self.ae_oe_tp, self.ae_oe_fp, self.ae_oe_fn)

        # Calculate the Aspect + Sentiment metrics
        results["ae_sc_f1"], results["ae_sc_prec"], results["ae_sc_rec"] = self._compute_metrics(self.ae_sc_tp, self.ae_sc_fp, self.ae_sc_fn)

        # Calculate the Triplet Extraction metrics
        results["tuple_f1"], results["tuple_prec"], results["tuple_rec"] = self._compute_metrics(self.triplet_tp, self.triplet_fp, self.triplet_fn)

        results["em"] = round(self.exact_matches / self.total, 4)
        results["invalid"] = round(self.invalid / self.total, 4)

        if reset:
            self.reset()

        return results

    def _compute_metrics(self, tp: int, fp: int, fn: int, beta: float = 1.0, round_and_normalize=True) -> tuple[float, float, float]:
        """
        Compute the F1, precision, and recall scores.

        Args:
            tp (int): The number of true positives.
            fp (int): The number of false positives.
            fn (int): The number of false negatives.
            beta (float): The beta value for the F1 score.
            round_and_normalize (bool): If True, round the scores to 4 decimal places and normalize on a 0-100 scale.

        Returns:
            tuple[float, float, float]: The F1, precision, and recall scores.
        """
        offset_constant = 1e-13
        precision = tp / (tp + fp + offset_constant)
        recall = tp / (tp + fn + offset_constant)
        f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + offset_constant)

        if round_and_normalize:
            f1 = round(f1, 4) * 100
            precision = round(precision, 4) * 100
            recall = round(recall, 4) * 100

        return f1, precision, recall
