from abc import abstractmethod

import torch
from src.metrics.base_metric import Metric


class TupleMetric(Metric):
    def __init__(self, mapping2targetid, eos_token_pointer_id: int):
        super(TupleMetric, self).__init__(eos_token_pointer_id)
        self.pad_token_id = 0  # TODO: Dont hardcode this
        self.num_labels = len(mapping2targetid)
        self.special_token_shift = 1  # +1, shift for eos
        self.word_start_index = self.num_labels + self.special_token_shift  # +1, shift for eos

        self.mapping2targetid = mapping2targetid
        self.reset()

    def reset(self):
        """Reset the metric counters. (Usually at the end of an epoch)"""
        self.all_preds = []
        self.all_targets = []

        self.exact_matches = 0
        self.invalid = 0
        self.total = 0

    def evaluate(self, pred: torch.Tensor, tgt_tokens: torch.Tensor):
        """
        Evaluate the predicted tokens against the target tokens

        Args:
            pred (torch.Tensor): The predicted tokens. It is assumed to be a 2D tensor of shape (batch_size, max_seq_len).
            tgt_tokens (torch.Tensor): The target tokens. It is assumed to be a 2D tensor of shape (batch_size, max_seq_len).
        """
        self.total += len(pred)

        # Mask with 1s for the positions after the eos token to avoid arithmetic errors
        pred_eos_mask = pred.flip(1).eq(self.eos_token_pointer_id).cumsum(1).long()
        target_eos_mask = tgt_tokens.flip(1).eq(self.eos_token_pointer_id).cumsum(1).long()

        # Get the sequence lengths
        pred_seq_len = pred_eos_mask.eq(pred_eos_mask[:, -1:]).sum(1)
        pred_seq_len = (pred_seq_len - 1).tolist()  # -1 to exclude the eos token itself
        target_seq_len = target_eos_mask.eq(target_eos_mask[:, -1:]).sum(1)
        target_seq_len = (target_seq_len - 1).tolist()  # -1 to exclude the eos token itself

        target_spans = self.get_target_span(tgt_tokens)

        for i, (target_seq, pred_seq) in enumerate(zip(target_spans, pred.tolist())):
            pred_seq = pred_seq[: pred_seq_len[i]]

            # If the sequences have the same length, we can directly compare them for an exact match
            if (pred_seq_len[i] == target_seq_len[i]) and (
                tgt_tokens[i, : target_seq_len[i]].eq(pred[i, : pred_seq_len[i]]).sum().item() == target_seq_len[i]
            ):
                self.exact_matches += 1

            # After a task dependent extraction of tuples, we need to validate them
            invalid, tuples = self.extract_pred_tuples(pred_seq)
            self.invalid += invalid

            # Save the target and prediction in a standardized format for evaluation
            self.all_targets.append(set([self.get_eval_format(t) for t in target_seq]))
            self.all_preds.append(set([self.get_eval_format(p) for p in tuples]))

    @abstractmethod
    def get_target_span(self, target):
        """Extracts the target spans from the target tokens."""
        raise NotImplementedError()

    @abstractmethod
    def get_eval_format(self, ps):
        """
        Convert a tuple to a standardized evaluation format.

        This method is implemented differently in each subclass to handle task-specific tuple structures.
        Typically, returning a 4-tuple of frozensets representing different components of the task.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate_tuple(self, tuple):
        """Validate a tuple based on task-specific criteria."""
        raise NotImplementedError()

    def get_metrics(self, reset=False):
        """
        Compute and retrieve the accumulated metric results.

        Args:
        reset (bool): If True, reset the metric counters after retrieval.

        Returns:
        dict[str, float]: A dictionary containing the computed metrics.
        """
        target = {k: v for k, v in enumerate(self.all_targets)}
        prediction = {k: v for k, v in enumerate(self.all_preds)}
        results = self.tuple_f1(target, prediction)

        results["em"] = round(self.exact_matches / self.total, 4)
        results["invalid"] = round(self.invalid / self.total, 4)

        if reset:
            self.reset()

        return results

    def extract_pred_tuples(self, pred_seq):
        """
        Simple base method to extract the tuples from the prediction sequence.
        Works for tasks where the tuples always end on a special token with no special tokens in between.
        All other tasks have to implement their own extraction method.
        """
        invalid = 0
        tuples = []
        cur_tuple = []
        if len(pred_seq):
            for j in pred_seq:
                if j == self.eos_token_pointer_id:
                    if len(cur_tuple) != 0:
                        invalid = 1
                    break

                cur_tuple.append(j)
                if j < self.word_start_index:
                    if self.validate_tuple(cur_tuple):
                        tuples.append(tuple(cur_tuple))
                    else:
                        invalid = 1
                    cur_tuple = []
        return invalid, tuples

    def tuple_f1(self, targets: dict, preds: dict, keep_polarity=True, weighted=True):
        """Calculate the F1 score for the tuples."""
        offset_constant = 1e-13
        prec = tuple_precision(targets, preds, keep_polarity, weighted)
        rec = tuple_recall(targets, preds, keep_polarity, weighted)

        return {
            "tuple_f1": (2 * (prec * rec) / (prec + rec + offset_constant)) * 100,
            "tuple_prec": prec * 100,
            "tuple_rec": rec * 100,
        }


def tuples_in_list(tuple1, list_of_tuples, keep_polarity=True):
    """Check if a tuple is in a list of tuples. (i.e. if its even relevant to process it further)"""
    tpl1, tpl2, tpl3, tpl_c = tuple1
    if len(tpl1) == 0:
        tpl1 = frozenset(["_"])
    if len(tpl2) == 0:
        tpl2 = frozenset(["_"])
    for cmp1, cmp2, cmp3, cmp_c in list_of_tuples:
        if len(cmp1) == 0:
            cmp1 = frozenset(["_"])
        if len(cmp2) == 0:
            cmp2 = frozenset(["_"])
        if len(tpl1.intersection(cmp1)) > 0 and len(tpl2.intersection(cmp2)) > 0 and len(tpl3.intersection(cmp3)) > 0:
            # If keep_polarity is True, we only return True if the tuples have the same polarity
            if keep_polarity:
                if tpl_c == cmp_c:
                    return True
            else:
                return True
    return False


def weighted_score(tuple1, list_of_tuples):
    """
    Calculate the best average overlap between a given tuple and a list of tuples.

    Args:
        tuple1 (tuple): The tuple to compare against the list of tuples.
        list_of_tuples (list): The list of tuples to compare with the given tuple.

    Returns:
        float: The highest average overlap score found.
    """
    best_overlap = 0
    tpl_1, tpl_2, tpl_3, tpl_c = tuple1
    if len(tpl_1) == 0:
        tpl_1 = frozenset(["_"])
    if len(tpl_2) == 0:
        tpl_2 = frozenset(["_"])
    for cmp_1, cmp_2, cmp_3, cmp_c in list_of_tuples:
        if len(cmp_1) == 0:
            cmp_1 = frozenset(["_"])
        if len(cmp_2) == 0:
            cmp_2 = frozenset(["_"])
        if len(cmp_1.intersection(tpl_1)) > 0 and len(cmp_2.intersection(tpl_2)) > 0 and len(cmp_3.intersection(tpl_3)) > 0:
            holder_overlap = len(cmp_1.intersection(tpl_1)) / len(tpl_1)
            target_overlap = len(cmp_2.intersection(tpl_2)) / len(tpl_2)
            exp_overlap = len(cmp_3.intersection(tpl_3)) / len(tpl_3)
            overlap = (holder_overlap + target_overlap + exp_overlap) / 3
            if overlap > best_overlap:
                best_overlap = overlap
    return best_overlap


def tuple_precision(gold, pred, keep_polarity=True, weighted=True, offset_constant=1e-13):
    """
    (Weighted) true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []

    for tpl_idx in pred.keys():
        ptuples = pred[tpl_idx]
        gtuples = gold[tpl_idx]
        for tpl in ptuples:
            if tuples_in_list(tpl, gtuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(tpl, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fp) + offset_constant)


def tuple_recall(gold, pred, keep_polarity=True, weighted=True, offset_constant=1e-13):
    """
    (Weighted) true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []

    for tpl_idx in pred.keys():
        ptuples = pred[tpl_idx]
        gtuples = gold[tpl_idx]
        for tuple in gtuples:
            if tuples_in_list(tuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(tuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + offset_constant)
