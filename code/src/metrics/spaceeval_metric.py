from src.metrics.base_tuple_metric import TupleMetric


class SpaceEvalMetric(TupleMetric):
    tuple_len = 3

    def get_target_span(self, target):
        """
        Extracts SpaceEval entity spans from the target tokens, splitting into 3-tuples (start, end, entity_type).
        """
        tgt_span = []
        for tokens in target:
            tokens_span = []
            while tokens[0] != self.eos_token_pointer_id:
                tokens_span.append(tuple(tokens[: self.tuple_len].tolist()))
                tokens = tokens[self.tuple_len :]
            tgt_span.append(tokens_span)
        return tgt_span

    def validate_tuple(self, tuple):
        """
        Validates SpaceEval tuples: correct length, special tokens in correct range, and valid start/end indices.
        """
        return (
            len(tuple) == self.tuple_len
            and self.special_token_shift <= tuple[2] < self.word_start_index <= tuple[0]
            and tuple[1] >= self.word_start_index
            and tuple[0] <= tuple[1]
        )

    def get_eval_format(self, tpl):
        """
        Formats SpaceEval tuples as (entity, _, _, class) for evaluation.
        """
        return (
            frozenset([tuple(tpl[0:2])]),
            frozenset(["_"]),
            frozenset(["_"]),
            frozenset([tpl[-1]]),
        )
