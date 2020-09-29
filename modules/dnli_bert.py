import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_transformers import (
    BertForSequenceClassification,
    BertTokenizer
)
from parlai.core.build_data import download_from_google_drive


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class DnliBert(object):
    def __init__(self,
                 opt,
                 dnli_lambda=1.0,
                 dnli_k=10,
                 max_seq_length=128,
                 use_cuda=True):
        self.opt = opt
        self.dnli_lambda = dnli_lambda
        self.dnli_k = dnli_k
        self.max_seq_length = max_seq_length
        self.use_cuda = use_cuda
        self.mapper = {0: "contradiction",
                       1: "entailment",
                       2: "neutral"}

        dnli_model, dnli_tokenizer = self._load_dnli_model()
        self.dnli_model = dnli_model
        self.dnli_tokenizer = dnli_tokenizer

    def _load_dnli_model(self):
        # Download pretrained weight
        dnli_model_fname = os.path.join(self.opt['datapath'], 'dnli_model.bin')
        if not os.path.exists(dnli_model_fname):
            print(f"[ Download pretrained dnli model params to {dnli_model_fname}]")
            download_from_google_drive(
                '1Qawz1pMcV0aGLVYzOgpHPgG5vLSKPOJ1',
                dnli_model_fname
            )

        # Load pretrained weight
        print(f"[ Load pretrained dnli model from {dnli_model_fname}]")
        model_state_dict = torch.load(dnli_model_fname)
        dnli_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', state_dict=model_state_dict, num_labels=3)
        if self.use_cuda:
            dnli_model.cuda()
        dnli_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        return dnli_model, dnli_tokenizer

    def rerank_candidates(self, observations, cand_scores):
        sorted_cand_values, sorted_cand_indices = cand_scores.sort(1, descending=True)

        for batch_idx, obs in enumerate(observations):
            full_text = obs['full_text']
            personas = []
            for text in full_text.split('\n'):
                if 'your persona:' in text:
                    personas.append(text.replace('your persona:', ''))
                else:
                    break
            candidates = obs['label_candidates']

            tok_candidates = [self.dnli_tokenizer.tokenize(sent) for sent in candidates]
            tok_personas = [self.dnli_tokenizer.tokenize(sent) for sent in personas]

            dnli_scores = self._compute_dnli_scores(tok_candidates, tok_personas)
            s_1 = sorted_cand_values[batch_idx, 0]
            s_k = sorted_cand_values[batch_idx, self.dnli_k - 1]

            _lambda = self.dnli_lambda
            cand_scores[batch_idx] = cand_scores[batch_idx] - _lambda * (s_1 - s_k) * dnli_scores

        return cand_scores

    def compute_consistency_scores(self, pred, personas):
        """
        preds, and personas must be list of string
        """
        max_seq_length = self.max_seq_length

        pred_tokenized = self.dnli_tokenizer.tokenize(pred)
        personas_tokenized = [self.dnli_tokenizer.tokenize(sent.replace('your persona:', '')) for sent in personas]

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        for idx, persona_tokenized in enumerate(personas_tokenized):
            _pred_tokenized = deepcopy(pred_tokenized)
            _persona_tokenized = deepcopy(persona_tokenized)
            _truncate_seq_pair(_pred_tokenized, _persona_tokenized, max_seq_length - 3)

            tokens = ["[CLS]"] + _pred_tokenized + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += _persona_tokenized + ["[SEP]"]
            segment_ids += [1] * (len(_persona_tokenized) + 1)

            input_ids = self.dnli_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        # Convert inputs to tensors
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
        if self.use_cuda:
            all_input_ids = all_input_ids.cuda()
            all_input_mask = all_input_mask.cuda()
            all_segment_ids = all_segment_ids.cuda()

        # Inference
        self.dnli_model.eval()
        with torch.no_grad():
            logits = self.dnli_model(all_input_ids, all_segment_ids, all_input_mask)
            probs = F.softmax(logits[0], dim=1)

        probs = probs.detach().cpu().numpy()
        idx_max = np.argmax(probs, axis=1)
        val_max = np.max(probs, axis=1)

        consistency_score = 0.0
        for pred_idx in idx_max:
            if pred_idx == 0: # contradict
                consistency_score -= 1.0
            elif pred_idx == 1: # entailment
                consistency_score += 1.0
            elif pred_idx == 2: # neutral
                consistency_score += 0.0

        return consistency_score

    def _compute_dnli_scores(self, tok_candidates, tok_personas):
        max_seq_length = self.max_seq_length

        dnli_scores = []
        for cand_idx, tok_candidate in enumerate(tok_candidates):
            all_input_ids = []
            all_input_mask = []
            all_segment_ids = []
            for tok_persona in tok_personas:
                # Prepare inputs
                # [CLS] candidates [SEP] persona [SEP]
                _tok_candidate = deepcopy(tok_candidate)
                _tok_persona = deepcopy(tok_persona)
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(_tok_candidate, _tok_persona, max_seq_length - 3)

                # Make inputs
                tokens = ["[CLS]"] + _tok_candidate + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                tokens += _tok_persona + ["[SEP]"]
                segment_ids += [1] * (len(_tok_persona) + 1)

                input_ids = self.dnli_tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)

            # Convert inputs to tensors
            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
            all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
            if self.use_cuda:
                all_input_ids = all_input_ids.cuda()
                all_input_mask = all_input_mask.cuda()
                all_segment_ids = all_segment_ids.cuda()

            # Inference
            self.dnli_model.eval()
            with torch.no_grad():
                logits = self.dnli_model(all_input_ids, all_segment_ids, all_input_mask)
                probs = F.softmax(logits[0], dim=1)

            probs = probs.detach().cpu().numpy()
            idx_max = np.argmax(probs, axis=1)
            val_max = np.max(probs, axis=1)
            dnli_score = np.max((idx_max == 0) * val_max)
            dnli_scores.append(dnli_score)
        dnli_scores = torch.tensor(dnli_scores, dtype=torch.float)
        if self.use_cuda:
            dnli_scores = dnli_scores.cuda()
        return dnli_scores
