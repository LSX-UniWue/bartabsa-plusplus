from src.metrics.base_tuple_metric import TupleMetric


class SSAMetric(TupleMetric):
    def __init__(self, mapping2targetid, eos_token_pointer_id: int):
        super(SSAMetric, self).__init__(mapping2targetid, eos_token_pointer_id)

        self.sentiment_labels = [mapping2targetid[label] + self.special_token_shift for label in ["POS", "NEU", "NEG"]]
        self.bos_special_tokens = [mapping2targetid[label] + self.special_token_shift for label in ["ASP_BEGIN", "OPN_BEGIN", "HOL_BEGIN"]]

    def extract_pred_tuples(self, pred_seq):
        """
        Extracts valid SSA tuples from the prediction sequence, grouping tokens between special begin tokens and sentiment labels.
        """
        invalid = 0
        tuples = []
        cur_tuple = []
        if len(pred_seq):
            for index, predicted_token in enumerate(pred_seq):
                cur_tuple.append(predicted_token)
                # If we have a special token, we can check if we have a complete and valid triplet
                if predicted_token in self.sentiment_labels:
                    if not self.validate_tuple(cur_tuple):
                        invalid = 1
                    else:
                        tuples.append(cur_tuple)
                    cur_tuple = []
        return invalid, tuples

    def get_target_span(self, target):
        """
        Extracts SSA tuples from the target sequence by splitting after sentiment labels, ending at EOS token.
        """
        tgt_span = []
        for tokens in target:
            tokens_span = []
            cur_span = []
            tokens = tokens.tolist()

            while tokens[0] != self.eos_token_pointer_id:
                cur_span.append(tokens[0])
                if tokens[0] in self.sentiment_labels:
                    assert self.validate_tuple(cur_span), f"Target span should only contain valid tuples"
                    tokens_span.append(cur_span)
                    cur_span = []
                tokens = tokens[1:]
            assert len(cur_span) == 0, "Target span should end with EOS"
            tgt_span.append(tokens_span)
        return tgt_span

    def validate_tuple(self, tuple):
        """
        Validates SSA tuples: correct special tokens for aspect, opinion, and holder, followed by valid token ranges and a sentiment label.
        """
        if not self.mapping2targetid["OPN_BEGIN"] + self.special_token_shift in tuple:
            return False
        if not tuple[-1] in self.sentiment_labels:
            return False
        tuple = tuple[:-1]
        if not tuple[0] in self.bos_special_tokens:
            return False
        tuple = tuple[1:]

        is_invalid_len = lambda x: len(x) == 0 or len(x) % 2 != 0
        cur = []
        for e in tuple:
            if e in self.bos_special_tokens:
                # elements does not match s & e format
                if is_invalid_len(cur):
                    return False
                cur = []
            elif e < self.word_start_index:
                return False
            else:
                cur.append(e)
        return not is_invalid_len(cur)

    def get_eval_format(self, in_tuple):
        """
        Formats SSA tuples as (aspect, holder, opinion, sentiment) for evaluation.
        """
        copy = [*in_tuple]
        try:
            tgt = []
            op = []
            hol = []
            count = 0
            while not in_tuple[0] in self.sentiment_labels:
                if in_tuple[0] == self.mapping2targetid["ASP_BEGIN"] + self.special_token_shift:
                    in_tuple = in_tuple[1:]
                    while in_tuple[0] >= self.word_start_index:
                        tgt.append(tuple(in_tuple[:2]))
                        in_tuple = in_tuple[2:]

                if in_tuple[0] == self.mapping2targetid["OPN_BEGIN"] + self.special_token_shift:
                    in_tuple = in_tuple[1:]
                    while in_tuple[0] >= self.word_start_index:
                        op.append(tuple(in_tuple[:2]))
                        in_tuple = in_tuple[2:]

                if in_tuple[0] == self.mapping2targetid["HOL_BEGIN"] + self.special_token_shift:
                    in_tuple = in_tuple[1:]
                    while in_tuple[0] >= self.word_start_index:
                        hol.append(tuple(in_tuple[:2]))
                        in_tuple = in_tuple[2:]
                count += 1
                if count > 1e3:
                    raise RuntimeError(f"infinite while loop in get_eval_format for {in_tuple}")

            assert len(in_tuple) == 1
            return (frozenset(tgt), frozenset(hol), frozenset(op), in_tuple[0])
        except Exception as e:
            print(f"Error while generating eval format for {copy}")
            raise e
