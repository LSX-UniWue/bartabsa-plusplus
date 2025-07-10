from src.metrics.base_tuple_metric import TupleMetric


class GabsaMetric(TupleMetric):
    tuple_len = 4

    def __init__(self, mapping2targetid, eos_token_pointer_id: int):
        super(GabsaMetric, self).__init__(mapping2targetid, eos_token_pointer_id=eos_token_pointer_id)
        self.class_start = (
            self.special_token_shift
            + max(self.mapping2targetid["POS"], self.mapping2targetid["NEG"], self.mapping2targetid["NEU"])
            + 1  # +1 for SEP
        )
        self.sep_idx = self.special_token_shift + self.mapping2targetid["SEP"]

    def extract_pred_tuples(self, ps):
        """
        Extracts valid GABSA tuples from the prediction sequence, ending at separator token.
        """
        invalid = 0
        pairs = []
        cur_pair = []

        if len(ps):
            for index, j in enumerate(ps):
                cur_pair.append(j)
                if j == self.sep_idx:
                    if self.validate_tuple(tuple(cur_pair)):
                        pairs.append(tuple(cur_pair))
                    else:
                        invalid = 1
                    cur_pair = []

        return invalid, pairs

    def get_target_span(self, target):
        """
        Extracts GABSA tuples from the target tokens, handling both 3-token (implicit aspect) and 5-token (explicit aspect) spans.
        """
        tgt_span = []
        for tokens in target:
            tokens_span = []
            while tokens[0] != self.eos_token_pointer_id:
                # Since the Aspect is optional, the tuple might be 3 or 5 tokens long
                if tokens[2] == self.sep_idx:
                    length = 3
                else:
                    length = 5
                tokens_span.append(tuple(tokens[:length].tolist()))
                assert self.validate_tuple(tokens_span[-1])
                tokens = tokens[length:]
            tgt_span.append(tokens_span)
        return tgt_span

    def validate_tuple(self, tuple):
        """
        Validates GABSA tuples: correct length, valid separator token, and correct token ranges for aspect (if present), class, and sentiment.
        """
        return tuple[-1] == self.sep_idx and (
            (len(tuple) == 3 and self.word_start_index > tuple[0] >= self.class_start > tuple[1] > self.sep_idx)
            or (len(tuple) == 5 and tuple[1] >= tuple[0] >= self.word_start_index > tuple[2] >= self.class_start > tuple[3] > self.sep_idx)
        )

    def get_eval_format(self, tpl):
        """
        Formats GABSA tuples as ((aspect), _, opinion, sentiment) for evaluation.
        """
        if len(tpl) == 3:
            return frozenset(["_"]), frozenset(["_"]), frozenset([tpl[0]]), frozenset([tpl[1]])
        else:
            return frozenset([tuple(tpl[0:2])]), frozenset(["_"]), frozenset([tpl[2]]), frozenset([tpl[3]])
