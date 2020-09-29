import os
import random
import numpy as np
from itertools import chain

import torch
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.torch_agent import Batch, Output
from parlai.core.torch_generator_agent import PPLMetric
from parlai.core.metrics import SumMetric, AverageMetric
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import warn_once
from parlai.agents.transformer.transformer import (
    TransformerGeneratorAgent,
    add_common_cmdline_args
)

from agents.modules import SelfConsciousTransformerModel
from modules.dnli_bert import DnliBert
from agents.history import SelfConsciousHistory, ContextConsciousHistory


def list_to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


class SelfConsciousBlenderAgent(TransformerGeneratorAgent):
    """
    Implementation of the Self-Conscious Blender Agent.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Self-conscious Blender Arguments')
        agent.add_argument(
            '--conscious-target',
            type=str,
            choices=['none', 'self', 'context'],
            default='self',
            help='The target which the agent will be concerned about.',
        )
        agent.add_argument(
            '-a',
            '--alpha',
            type=float,
            default=0,
            help='Rationality parameter for S_1(speaker_1)',
        )
        agent.add_argument(
            '-b',
            '--beta',
            type=float,
            default=1,
            help='Rationality parameter for Listener',
        )
        agent.add_argument(
            '--world_cardinality',
            type=int,
            default=3,
            help='Cardinality of world I:= Number of persona to use RSA model (including GT)',
        )
        agent.add_argument(
            '--worldprior',
            type=str,
            choices=['uniform', 'L0', 'L1'],
            default='L0',
            help='Update world prior with a `uniform` distribution or `L0` or `L1`.',
        )
        agent.add_argument(
            '--use_dnli',
            type=bool,
            default=True,
            help='Whether to use dnli model to measure consistency-score in Convai2 or rerank candidates in DNLI'
        )
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(SelfConsciousBlenderAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt: Opt, shared=None):

        self.task = str.lower(opt['task'].split(':')[-1])

        if opt['conscious_target'] != 'none':
            assert opt['conscious_target'] in self.task, \
                "conscious_target (`" + opt['conscious_target'] + "`) must match task type (`" + self.task + "`)"

        SEED = 46
        random.seed(SEED)
        np.random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.random.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For public self-consciousness
        self.target_persona = opt.get('target_persona', 0)
        self.conscious_target = opt.get('conscious_target', 'self')
        self.world_cardinality = opt.get('world_cardinality', 3)
        self.alpha = 0.0 if self.conscious_target == 'none' else opt.get('alpha', 2.0)
        self.beta = opt.get('beta', 1.0)
        self.worldprior = opt.get('worldprior', 'L0')

        self.eval_type = opt.get('eval_type')
        # self.rank_candidates = opt.get('rank_candidates', True)
        self.multigpu = (
            opt.get('multigpu', False) and self.use_cuda and (opt.get('batchsize') > 1)
        )

        init_model, is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        # Implementation is based on beam_size 1
        self.beam_size = 1
        warn_once(f'This implementation is assumed to have beam-size 1.')

        # Always rank candidates for the ranking metrics
        self.rank_candidates = True
        warn_once(f'rank-candidates is always True for ranking metrics.')

        if opt['use_dnli']:
            if not shared:
                self.dnli_model = DnliBert(opt, use_cuda=self.use_cuda)
            else:
                self.dnli_model = shared['dnli_model']
        else:
            self.dnli_model = None

        self.id = 'SelfConsciousBlender'

        self.reset()

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = SelfConsciousTransformerModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model

    def history_class(self):
        return ContextConsciousHistory if 'context' in self.task else SelfConsciousHistory

    def _model_input(self, batch):
        """
        Override from TorchGeneratorAgent
        passes (batch.text_vec,) to TorchGeneratorAgent._encoder_input()
        TGA._encoder_input() directly passes the result of TGA._model_input()
        change batch.text_vec to batch.distractor_text_vec for pragmatic decoding
        """
        bsz = batch.text_vec.size(0)
        distractor_text_vec = batch.distractor_text_vec.view(bsz * self.world_cardinality, -1).contiguous()
        return (distractor_text_vec,)

    def selfconscious_greedy_generate(self, batch, maxlen):
        """
        Greedy decoding with Public Self-Consciousness
        """

        bsz = batch.text_vec.size(0)
        world_cardinality = self.world_cardinality
        embedding_size = self.opt.get('embedding_size')
        encoder_states = self.model.encoder(*self._encoder_input(batch))

        preds, scores = self.model.selfconscious_decode(encoder_states, maxlen)

        return preds, scores

    def rank(self, batch):
        """
        Rank candidates by PPL score
        """
        bsz = batch.text_vec.size(0)
        world_cardinality = self.world_cardinality
        embedding_size = self.opt.get('embedding_size')
        ranked_candidates = []
        cand_ordering = []
        encoder_states = self.model.encoder(*self._encoder_input(batch))
        batch_dim = encoder_states[0].size(0)  # two possibilities: batchsize or batchsize * world_cardinality

        if bsz != batch_dim:
            enc_output = encoder_states[0].view(bsz, world_cardinality, -1, embedding_size).contiguous()
            enc_output_mask = encoder_states[1].view(bsz, world_cardinality, -1).contiguous()
            encoder_states = (enc_output, enc_output_mask)

        for i in range(bsz):
            num_cands = len(batch.candidate_vecs[i])
            cands, _ = self._pad_tensor(batch.candidate_vecs[i])
            # get [i]th state from encoder_states #num_cands time.
            # because we need same encoder_states for each candidate
            enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)

            # enc: (num_cands, world_cardinality, seqlen, emb_size)
            # scores: (num_cands, max_len, vocab_size)
            scores, _ = self.model.selfconscious_decode_forced(enc, cands)

            cand_losses = F.cross_entropy(
                scores.view(num_cands * cands.size(1), -1),
                cands.view(-1),
                reduction='none',
            ).view(num_cands, cands.size(1))
            # now cand_losses is cands x seqlen size, but we still need to
            # check padding and such
            mask = (cands != self.NULL_IDX)
            mask = mask.half() if self.fp16 else mask.float()
            cand_scores = (-cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

            if self.dnli_model is not None and self.eval_type == 'dnli':
                cand_scores = torch.unsqueeze(cand_scores, 0)
                cand_scores = self.dnli_model.rerank_candidates([batch.observations[i]], cand_scores)
                cand_scores = torch.squeeze(cand_scores)

            _, ordering = cand_scores.sort(descending=True)
            ranked_candidates.append([batch.candidates[i][o] for o in ordering])
            cand_ordering.append(ordering)

        return ranked_candidates, cand_ordering

    def compute_loss(self, batch, return_output=False):
        """
        Override from TorchGeneratorAgent
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')

        bsz = batch.text_vec.size(0)
        world_cardinality = self.world_cardinality
        embedding_size = self.opt.get('embedding_size')
        encoder_states = self.model.encoder(*self._encoder_input(batch))

        enc_output = encoder_states[0].view(bsz, world_cardinality, -1, embedding_size).contiguous()
        enc_output_mask = encoder_states[1].view(bsz, world_cardinality, -1).contiguous()
        encoder_states = (enc_output, enc_output_mask)

        scores, preds = self.model.selfconscious_decode_forced(encoder_states, batch.label_vec)
        model_output = (scores, preds, encoder_states)

        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )

        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token

        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _eval_convai2_step(self, batch):
        """Evaluate a single batch of examples."""

        assert self.alpha >= 0
        if batch.distractor_text_vec is None:
            return None

        self.model.eval()

        # 1. Generation
        assert self.beam_size is 1
        maxlen = self.label_truncate or 256
        if not self.skip_generation:
            preds, scores = self.selfconscious_greedy_generate(batch, maxlen)
        else:
            preds = None

        # 2. Compute PPL with teacher-forced generation
        # calculate loss on targets with teacher forcing
        loss, model_output = self.compute_loss(batch, return_output=True)
        token_losses = self._construct_token_losses(
            batch.label_vec, model_output
        )

        # 3. Rank candidates by computing PPL for each candidates
        if self.rank_candidates:
            ranked_cands, ordering = self.rank(batch)
        else:
            ranked_cands = None

        # 4. Compute consistency score
        additional_metrics = [{'c_score': 0.0} for _ in range(len(batch.observations))]
        output_texts = [self._v2t(p) for p in preds] if preds is not None else None
        if not self.skip_generation:
            if self.opt['use_dnli']:
                c_scores = []
                for text, obs in zip(output_texts, batch.observations):
                    if 'context' in self.task:
                        c_score = self.dnli_model.compute_consistency_scores(text, obs['my_context'])
                    else:
                        persona_strings = obs['my_persona'].split('\n')
                        c_score = self.dnli_model.compute_consistency_scores(text, persona_strings)

                    c_scores.append(c_score)

                for idx, c_score in enumerate(c_scores):
                    additional_metrics[idx]['c_score'] = c_score

        return Output(output_texts, ranked_cands, token_losses=token_losses, metrics=additional_metrics)

    def _eval_dnli_step(self, batch):
        """Evaluate a single batch of examples."""

        assert self.alpha >= 0

        self.model.eval()
        ranked_cands, ordering = self.rank(batch)

        bsz = len(ranked_cands)
        dnli_metrics = []
        for batch_idx in range(bsz):
            dnli_score = {'contradict@1': 0, 'entail@1': 0, 'neutral@1': 0}
            top1_idx = ordering[batch_idx][0].item()
            if top1_idx == 0:
                pass
                # dnli_metrics['dnli_hit@1'] += 1
            elif top1_idx > 0 and top1_idx < 11:
                dnli_score['contradict@1'] += 1
            elif top1_idx >= 11 and top1_idx < 21:
                dnli_score['entail@1'] += 1
            else:
                dnli_score['neutral@1'] += 1
            dnli_metrics.append(dnli_score)

        return Output(text_candidates=ranked_cands, metrics=dnli_metrics)

    def eval_step(self, batch):

        if self.opt['eval_type'] == 'convai2':
            return self._eval_convai2_step(batch)
        elif self.opt['eval_type'] == 'dnli':
            return self._eval_dnli_step(batch)
        else:
            raise NotImplementedError

    def self_observe(self, self_message: Message):
        """
        Override from TorchAgent
        Update the model's reply or label to the history of distractor-fields in History class
        """
        episode_done = self.observation['episode_done']
        use_reply = self.opt.get('use_reply', 'label')

        # actually ingest the label
        if use_reply == 'none':
            # we're not including our own responses anyway.
            reply = None
        elif use_reply == 'label':
            # first look for the true label
            label_key = (
                'labels'
                if 'labels' in self.observation
                else 'eval_labels'
                if 'eval_labels' in self.observation
                else None
            )
            if label_key is not None:
                lbls = self.observation[label_key]
                reply = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
        else:
            # otherwise, we use the last output the model generated
            if self_message is not None:
                reply = self_message['text']
            else:
                reply = None

        super().self_observe(self_message)

        if episode_done:
            return None

        if reply is not None:
            if 'context' in self.task:
                self.history.add_reply_to_distractors(reply, self.observation)
            else:
                self.history.add_reply_to_distractors(reply)

        return reply

    def _ordered_cand_scores_to_cand_text(self, ordered_cand_preds, cand_inds, candidates):
        cand_replies = [None] * len(candidates)

        for idx, order in enumerate(ordered_cand_preds):  # batch_idx, sorted cand_idx
            batch_idx = cand_inds[idx]
            # get the original sentences from candidates by order
            cand_replies[batch_idx] = [candidates[batch_idx][i] for i in order]

        return cand_replies

    def _build_candidates_tensor(self, batch):
        if not batch.candidates:
            return None, None

        cand_inds = [i for i in range(len(batch.candidates)) if batch.candidates[i]]
        cands = [batch.candidate_vecs[i] for i in cand_inds]

        # get the length of the longest candidate in the batch
        max_cand_len = max(
            [max([cand.size(0) for cand in cands_i]) for cands_i in cands]
        )

        for i, c in enumerate(cands):  #  make each instance in batch.cands to a padded tensor
            cands[i] = padded_tensor(c, use_cuda=self.use_cuda,
                                     max_len=max_cand_len,
                                     fp16friendly=self.fp16)[0].unsqueeze(0)

        # (batchsize, num_cands, max_len + a) +a due to fp16
        cands = torch.cat(cands, 0)

        return cands, cand_inds

    def vectorize(self, obs, history, **kwargs):
        """
        Override from TorchAgent
        Vectorize the texts in observation
        """
        super().vectorize(obs, history, **kwargs)  # candidate vecs are vectorized here
        if not self.is_training:
            self._set_distractor_text_vec(obs, history, kwargs['text_truncate'])
        return obs

    def _set_text_vec(self, obs, history, truncate):
        """
        Override from TorchAgent for DNLI evaluation
        This will be called in super().vectorize()
        """
        # WARNING: self.is_training is always False in here
        is_training = False if 'eval_labels' in obs else True

        if is_training or self.opt['eval_type'] == 'convai2':
            return super()._set_text_vec(obs, history, truncate)
        elif self.opt['eval_type'] == 'dnli':
            if 'text' not in obs:
                return obs

            # Vectorize the text
            if 'text_vec' not in obs:
                obs['full_text'] = obs['text']
                vec = self.dict.txt2vec(obs['full_text'])
                obs['text_vec'] = vec

            # check truncation
            if obs.get('text_vec') is not None:
                truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
                obs.force_set('text_vec', torch.LongTensor(truncated_vec))
            return obs
        else:
            raise NotImplementedError

    def _set_distractor_text_vec(self, obs, history, truncate):
        """
        Set 'distractor_text' and 'distractor_text_vec' field in the observation
        """
        if 'distractor_text' not in obs:
            return obs

        if 'distractor_text_vec' not in obs:
            # distractor_text is in the SelfConsciousHistory class
            distractor_string = history.get_history_distractor_str()

            if distractor_string is None:
                return obs

            # Set 'full_distractor_text'
            obs['full_distractor_text'] = distractor_string
            # distractor_text_vec is also in the SelfConsciousHistory class
            # they are already vectorized at SelfConsciousHistory.update_history()
            if distractor_string:
                obs['distractor_text_vec'] = history.get_history_distractor_vec()

        # Check truncation
        if obs.get('distractor_text_vec') is not None:
            truncated_vec = [
                torch.LongTensor(self._check_truncate(text_vec, truncate, True))
                for text_vec in obs['distractor_text_vec']
            ]
            obs.force_set('distractor_text_vec', truncated_vec)
        return obs

    def batchify(self, *args, **kwargs):
        """
        Override from TorchAgent
        Additionally batchify the distractor_text_vec and add it to batch
        """
        kwargs['sort'] = True  # need sort for pack_padded()
        batch = super().batchify(*args, **kwargs)
        sort = False  # we must not sort after super().batchify()

        exs = batch.observations
        d_text_vec, d_lens = None, None
        if any('distractor_text_vec' in ex for ex in exs):
            # Pad distractor vectors
            _d_text_vec = [ex.get('distractor_text_vec', self.EMPTY) for ex in exs]
            _d_text_vec_flattened = list(chain(*_d_text_vec))
            d_text_vec, d_lens = self._pad_tensor(_d_text_vec_flattened)

            # Reshape to (batch_size, world_cardinality, max_length)
            bsz = len(exs)
            d_text_vec = d_text_vec.view(bsz, self.world_cardinality, -1)
            d_lens = list_to_matrix(d_lens, self.world_cardinality)

        batch = Batch(
            distractor_text_vec=d_text_vec,
            distractor_text_lengths=d_lens,
            **dict(batch)
        )

        return batch

    def share(self):
        shared = super().share()
        if self.opt['use_dnli']:
            shared['dnli_model'] = self.dnli_model
        return shared
