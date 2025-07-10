from src.metrics.base_tuple_metric import TupleMetric


class SREMetric(TupleMetric):
    tuple_len: int = 5

    def get_target_span(self, target):
        """
        Extracts SRE tuples from the target sequence by splitting into 5-tuples (source start, source end, target start, target end, relation), ending at EOS token.
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
        Validates SRE tuples: correct length, valid token ranges for source and target entities, and a valid relation label.
        """
        if len(tuple) != self.tuple_len:
            return False
        if not tuple[-1] < self.word_start_index:
            return False
        for s, e in [(0, 1), (2, 3)]:
            if not (self.word_start_index <= tuple[s] <= tuple[e]):
                return False
        return True

    def get_eval_format(self, tpl):
        """
        Formats SRE tuples as (source, _, target, _, class) for evaluation.
        """
        return (
            frozenset([tuple(tpl[0:2])]),
            frozenset(["_"]),
            frozenset([tuple(tpl[2:4])]),
            frozenset([tpl[-1]]),
        )
