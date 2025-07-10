from src.metrics.base_tuple_metric import TupleMetric


class GnerMetric(TupleMetric):
    tuple_len = 3

    def __init__(self, mapping2targetid, eos_token_pointer_id: int):
        super(GnerMetric, self).__init__(mapping2targetid, eos_token_pointer_id=eos_token_pointer_id)

    def extract_pred_tuples(self, ps):
        """
        Extracts valid GNER entity spans from the prediction sequence, ending at separator token.
        """
        invalid = 0
        pairs = []
        cur_pair = []

        if len(ps):
            for index, j in enumerate(ps):
                cur_pair.append(j)
                if j < self.word_start_index:
                    if self.validate_tuple(tuple(cur_pair)):
                        pairs.append(tuple(cur_pair))
                    else:
                        invalid = 1
                    cur_pair = []

        return invalid, pairs

    def get_target_span(self, target):
        """
        Extracts GNER entity spans from the target tokens, splitting into 3-tuples (start, end, entity_type).
        """
        tgt_span = []
        for tokens in target:
            tokens_span = []
            while tokens[0] != self.eos_token_pointer_id:
                tokens_span.append(tuple(tokens[: self.tuple_len].tolist()))
                assert self.validate_tuple(tokens_span[-1])
                tokens = tokens[self.tuple_len :]
            tgt_span.append(tokens_span)
        return tgt_span

    def validate_tuple(self, tuple):
        """
        Validates GNER tuples: correct length, valid separator token, and correct token ranges for entity spans and class.
        """
        return len(tuple) == self.tuple_len and tuple[1] >= tuple[0] >= self.word_start_index > tuple[2]

    def get_eval_format(self, tpl):
        """
        Formats GNER tuples as (entity, _, _, class) for evaluation.
        """
        return (
            frozenset([tuple(tpl[0:2])]),
            frozenset(["_"]),
            frozenset(["_"]),
            frozenset([tpl[-1]]),
        )
