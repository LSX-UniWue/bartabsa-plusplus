from src.metrics.base_tuple_metric import TupleMetric, tuple_precision, tuple_recall


class ABSAMetric(TupleMetric):
    def extract_pred_tuples(self, pred_seq):
        """
        Extracts valid ABSA tuples from the prediction sequence. Each tuple consists of aspect start, aspect end, opinion start, opinion end, and sentiment.
        """
        invalid = 0
        tuples = []
        cur_tuples = []
        if len(pred_seq):
            for index, predicted_token in enumerate(pred_seq):
                cur_tuples.append(predicted_token)
                # If we have a special token, we can check if we have a complete and valid triplet
                if predicted_token < self.word_start_index:
                    if len(cur_tuples) == 5 and cur_tuples[0] <= cur_tuples[1] and cur_tuples[2] <= cur_tuples[3]:
                        tuples.append(tuple(cur_tuples))
                    else:
                        invalid = 1
                    cur_tuples = []
        return invalid, tuples

    def get_target_span(self, target):
        """
        Extracts ABSA tuples from the target sequence by splitting into 5-tuples, ending at EOS token.
        """
        tgt_span = []
        for tokens in target:
            tokens_span = []
            while tokens[0] != self.eos_token_pointer_id:
                tokens_span.append(tuple(tokens[:5].tolist()))
                tokens = tokens[5:]
            tgt_span.append(tokens_span)
        return tgt_span

    def validate_tuple(self, tuple):
        """
        No special validation needed for ABSA tuples as they are always 5-tuples.
        """
        pass

    def get_eval_format(self, tpl):
        """
        Formats the ABSA tuple as (aspect, _, opinion, sentiment) for evaluation.
        """
        return (
            frozenset([tuple(tpl[0:2])]),
            frozenset(["_"]),
            frozenset([tuple(tpl[2:4])]),
            frozenset([tpl[-1]]),
        )

    # def tuple_f1(self, targets: dict, preds: dict, keep_polarity=True, weighted=True):
    #     """For ABSA, we want to keep the metrics name the same as with the triplet metric."""
    #     offset_constant = 1e-13
    #     prec = tuple_precision(targets, preds, keep_polarity, weighted)
    #     rec = tuple_recall(targets, preds, keep_polarity, weighted)

    #     return {
    #         "triplet_f1": (2 * (prec * rec) / (prec + rec + offset_constant)) * 100,
    #         "triplet_prec": prec * 100,
    #         "triplet_rec": rec * 100,
    #     }
