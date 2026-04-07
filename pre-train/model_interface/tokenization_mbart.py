# coding:utf-8
import sys
import penman
import regex as re
from transformers import MBart50TokenizerFast
from common import postprocessing
from common.penman_interface import encode
from common.constant import raw_special_tokens, recategorizations


class AMRMBartTokenizer(MBart50TokenizerFast):
    INIT = "▁"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modified = 0
        self.recategorizations = set(r.lstrip("Ġ▁") for r in recategorizations)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.remove_pars = False

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary()
        return inst

    def _refresh_vocab_maps(self):
        vocab = self.get_vocab()
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}

    def init_amr_vocabulary(self):
        self._refresh_vocab_maps()
        self.old_enc_size = old_enc_size = len(self.encoder)
        tokens = [t for t in raw_special_tokens if t not in self.encoder]
        if tokens:
            self.add_tokens(tokens)
        self._refresh_vocab_maps()
        self.modified = len(self.encoder) - old_enc_size

        self.amr_bos_token = "<AMR>"
        self.amr_bos_token_id = self.encoder[self.amr_bos_token]
        self.amr_eos_token = "</AMR>"
        self.amr_eos_token_id = self.encoder[self.amr_eos_token]
        print(f"Added {self.modified} AMR tokens")

    def _tok(self, token):
        return self.tokenize(token, add_special_tokens=False)

    def tokenize_amr(self, amr_tokens):
        bpe_tokens = []
        for tokk in amr_tokens:
            is_in_enc = tokk in self.encoder or (self.INIT + tokk) in self.encoder
            is_rel = tokk.startswith(":") and len(tokk) > 1
            is_spc = tokk.startswith("<") and tokk.endswith(">")
            is_of = tokk.startswith(":") and tokk.endswith("-of")
            is_frame = re.match(r".+-\d\d", tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace("_", " ")
                bpe_toks = self._tok("<lit>") + self._tok(tokk) + self._tok("</lit>")
            elif is_rel or is_spc or is_frame or is_of:
                if is_in_enc:
                    bpe_toks = [tokk] if tokk in self.encoder else [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok(tokk[:-3]) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if rel in self.encoder or (self.INIT + rel) in self.encoder:
                        bpe_toks = [rel] if rel in self.encoder else [self.INIT + rel]
                        bpe_toks += ["-of"]
                    else:
                        bpe_toks = self._tok(":") + self._tok(rel[1:]) + ["-of"]
                elif is_rel:
                    bpe_toks = self._tok(":") + self._tok(tokk[1:])
                else:
                    bpe_toks = self._tok(tokk)
            else:
                bpe_toks = [tokk] if tokk in self.encoder else self._tok(tokk)

            bpe_tokens.extend(bpe_toks)
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        return bpe_token_ids

    def decode_amr(self, tokens, restore_name_ops=None):
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print("Decoding failure:", file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph_ = graph = self._fix_and_make_graph(nodes)
        except Exception as e:
            print("Building failure:", file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print("Reconnection 1 failure:")
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print("Reconnction 2 failure:", file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)

    def _fix_and_make_graph(self, nodes):
        from model_interface.tokenization_bart import AMRBartTokenizer

        return AMRBartTokenizer._fix_and_make_graph(self, nodes)
