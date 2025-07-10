from src.metrics.base_tuple_metric import TupleMetric


class DEFTMetric(TupleMetric):
    tuple_len: int = 7
    num_special_tokens = 1
    link_labels = ["DIR_DEF", "IDIR_DEF", "REF_TO", "AKA", "SUP", "FRAG"]

    def __init__(self, mapping2targetid, eos_token_pointer_id: int):
        super().__init__(mapping2targetid, eos_token_pointer_id)
        self.links = [self.mapping2targetid[l] + self.num_special_tokens for l in self.link_labels]

    def extract_pred_tuples(self, pred_seq):
        """
        Extracts valid DEFT tuples from the prediction span, grouping tokens between link labels.
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
                if j in self.links:
                    if self.validate_tuple(cur_tuple):
                        tuples.append(tuple(cur_tuple))
                    else:
                        invalid = 1
                    cur_tuple = []
        return invalid, tuples

    def get_target_span(self, target):
        """
        Extracts DEFT tuples from the target sequence by splitting into 7-tuples, ending at EOS token.
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
        Validates DEFT tuples: correct length, valid special tokens for entity types, and correct token ranges for source and target entities.
        """
        if len(tuple) != self.tuple_len:
            return False
        for i in [2, 5]:
            if not (self.special_token_shift <= tuple[i] < self.word_start_index and tuple[i] not in self.links):
                return False
        for s, e in [(0, 1), (3, 4)]:
            if not (self.word_start_index <= tuple[s] <= tuple[e]):
                return False
        return True

    def get_eval_format(self, tpl):
        """
        Formats DEFT tuples as (source with type, _, target with type, _, class) for evaluation.
        """
        return (
            frozenset([tuple(tpl[0:3])]),
            frozenset(["_"]),
            frozenset([tuple(tpl[3:6])]),
            frozenset([tpl[-1]]),
        )
