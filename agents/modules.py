#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements NN code for transformers.

Original paper: https://arxiv.org/abs/1706.03762. (Vaswani, 2017). The
`Annotated Transformer` (Rush, 2018) is an excellent reading guide which explains
much of the mechanics of the Transformer model
(http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This module also supports special segments (ala BERT;
https://arxiv.org/abs/1810.04805), and a few different variations seen in the
literature (BERT and XLM; https://arxiv.org/abs/1901.07291).
"""

import math
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.agents.transformer.modules import TransformerGeneratorModel


class SelfConsciousTransformerModel(TransformerGeneratorModel):
    """
    Implements a full transformer generator model, with pragmatic self-consciousness.
    """

    def __init__(self, opt, dictionary):
        super().__init__(opt, dictionary)

        self.alpha = 0.0 if opt['conscious_target'] == 'none' else opt['alpha']
        self.beta = opt['beta']
        self.world_cardinality = opt['world_cardinality']
        self.worldprior = opt['worldprior']
        self.target_persona = 0
        self.fp16 = opt['fp16']

    def _initialize_worldpriors(self, bsz, seqlen):
        """
        initialize the world prior with a uniform distribution
        """
        cardinality = self.world_cardinality
        torch_dtype=torch.half if self.fp16 else torch.float
        ones = torch.ones(1, seqlen, cardinality, dtype=torch_dtype, requires_grad=False).cuda()
        uniform_world_prior = torch.log(ones / cardinality)
        world_priors = uniform_world_prior.repeat(bsz, 1, 1).detach()

        return world_priors

    def _pragmatic_reasoning(self, s0_t, worldprior):
        """
        run pragmatic reasoning with the base speaker and its imaginary listener
        """

        vocab_size = self.embeddings.num_embeddings

        # log-scale
        log_score = nn.functional.log_softmax(s0_t, dim=2)
        log_score = log_score.squeeze()  # (bpsz, vocab)

        # (bsz, world_cardinality, vocab)
        log_score = log_score.view(-1, self.world_cardinality, vocab_size)

        # S_0 for L_1
        _literal_speaker = log_score.clone()
        _literal_speaker, _literal_s_next_token_idxs = torch.max(_literal_speaker, dim=-1, keepdim=True)

        # S_0 for the actual given persona (bsz, vocab)
        speaker_prior = log_score.select(1, self.target_persona)  # target persona is always index 0

        # S_0 for L_0
        # (bsz, vocab, world_cardinality)
        log_score = log_score.transpose(dim0=1, dim1=2).contiguous()
        log_score = log_score * self.beta

        # L_0 \propto S_0 * p(i)
        # worldprior should be broadcasted to all the tokens
        # (bsz, vocab, world_cardinality)
        listener_posterior = (log_score + worldprior) - torch.logsumexp(log_score + worldprior, 2, keepdim=True)

        # (bsz, vocab)
        listener_score = listener_posterior.select(2, self.target_persona)  # target persona is always index 0
        listener_score = listener_score * self.alpha

        speaker_posterior = (listener_score + speaker_prior) - torch.logsumexp(listener_score + speaker_prior, 1, keepdim=True)

        # need to unsqueeze in the dimension 1
        speaker_posterior = speaker_posterior.unsqueeze(1)  # (bsz, 1, vocab)

        # L_0 for L_1
        _literal_listener = listener_posterior.transpose(dim0=1, dim1=2).contiguous()
        _literal_listener = torch.gather(_literal_listener, -1, _literal_s_next_token_idxs)

        pragmatic_listener = (_literal_speaker + _literal_listener) - torch.logsumexp(_literal_speaker + _literal_listener, 1, keepdim=True)
        pragmatic_listener = pragmatic_listener.squeeze()

        return speaker_posterior, listener_posterior, pragmatic_listener

    def selfconscious_decode(self, encoder_states, maxlen):
        """
        greedy decoding with pragmatic self-consciousness
        """
        bpsz = encoder_states[0].size(0)
        bsz = bpsz // self.world_cardinality

        inputs_t = self.START.detach().expand(bpsz, 1)
        worldpriors = self._initialize_worldpriors(bsz, maxlen).detach()

        s1_scores = []
        incr_state = None

        for t in range(maxlen):
            worldprior_t = worldpriors.select(1, t).unsqueeze(1)

            latent, incr_state = self.decoder(inputs_t, encoder_states, incr_state)
            _logits = self.output(latent)
            # only get the last timestep's logit
            s0_t = _logits.select(dim=1, index=-1).unsqueeze(1)  # logits shape: (bpsz, 1, vocab)

            # s1_t: (bsz, 1, vocab)
            # listener_posterior: (bsz, vocab, world_cardinality)
            s1_t, l0_t, l1_t = self._pragmatic_reasoning(s0_t, worldprior_t)
            s1_scores.append(s1_t)

            next_token = s1_t.max(2)[1].clone().detach()  # next input is current predicted output idx

            idx_for_tile = torch.arange(bsz).repeat(self.world_cardinality, 1).transpose(0, 1).reshape(-1).cuda()
            inputs_next_t = torch.index_select(next_token, 0, idx_for_tile)
            next_token = next_token.unsqueeze(2)
            tiled_next_token = next_token.repeat(1, 1, self.world_cardinality)

            if self.worldprior != 'uniform':
                # (bsz, vocab, world_cardinality) -> (bsz, 1, world_cardinality)
                updated_world_prior = torch.gather(l0_t, 1, tiled_next_token).clone().detach()
                if t + 1 < maxlen:
                    if self.worldprior == 'L0':
                        worldpriors[:, t + 1, :] = updated_world_prior.squeeze()
                    elif self.worldprior == 'L1':
                        worldpriors[:, t + 1, :] = l1_t
                    else:
                        raise NotImplementedError

            # update inputs for next timestep
            inputs_t = torch.cat((inputs_t, inputs_next_t), dim=1)

        s1_scores = torch.cat(s1_scores, dim=1)  # (bsz, seqlen, vocab)
        _, preds = s1_scores.max(dim=2)

        return preds, s1_scores

    def selfconscious_decode_forced(self, encoder_states, ys):
        """
        faster teacher-forced decoding with pragmatic self-consciousness
        """

        bsz = ys.size(0)
        seqlen = ys.size(1)
        self.longest_label = max(self.longest_label, seqlen)
        emb_size = self.encoder.embedding_size
        enc_outputs = encoder_states[0].view(bsz * self.world_cardinality, -1, emb_size).contiguous()
        enc_outputs_mask = encoder_states[1].view(bsz * self.world_cardinality, -1).contiguous()
        enc_states = (enc_outputs, enc_outputs_mask)
        bpsz = enc_outputs.size(0)

        # tile ys as much as the world_cardinality
        idx_for_tile = torch.arange(bsz).repeat(self.world_cardinality, 1).transpose(0, 1).reshape(-1).cuda()
        tiled_ys = torch.index_select(ys, 0, idx_for_tile)

        inputs = tiled_ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self.START.detach().expand(bpsz, 1), inputs], 1)
        worldpriors = self._initialize_worldpriors(bsz, seqlen).detach()
        s1_scores = []

        latent, _ = self.decoder(inputs, enc_states)
        base_speaker = self.output(latent)

        for t in range(seqlen):

            s0_t = base_speaker.select(dim=1, index=t).unsqueeze(1)  # s0_t: (bpsz, 1, vocab)
            worldprior_t = worldpriors.select(dim=1, index=t).unsqueeze(1)

            # s1_t: (bsz, 1, vocab)
            # l0_t: (bsz, vocab, world_cardinality)
            s1_t, l0_t, l1_t = self._pragmatic_reasoning(s0_t, worldprior_t)
            s1_scores.append(s1_t)

            # Update world_prior with listener posterior
            if t + 1 < seqlen:
                next_tokens = inputs.select(1, t + 1).view(-1, 1)  # (bpsz, 1): the next tokens for each bpsz instance
                next_tokens = next_tokens.unsqueeze(2)
                # [0, 1*world_cardinality, 2*wc, 3*wc, ..., bpsz - 1wc] -> to get the ground-truth personas
                target_persona_idxs = torch.arange(bsz).cuda() * (self.world_cardinality)

                # we only need the next token of the ground-truth persona
                next_token = torch.index_select(next_tokens, 0, target_persona_idxs)  # (bsz, 1, 1)
                tiled_next_token = next_token.repeat(1, 1, self.world_cardinality)  # (bsz, 1, world_cardinality)

                if self.worldprior != 'uniform':
                    # (bsz, vocab, world_cardinality) -> (bsz, 1, world_cardinality)
                    updated_world_prior = torch.gather(l0_t, 1, tiled_next_token).clone().detach()
                    if self.worldprior == 'L0':
                        worldpriors[:, t + 1, :] = updated_world_prior.squeeze()
                    elif self.worldprior == 'L1':
                        worldpriors[:, t + 1, :] = l1_t
                    else:
                        raise NotImplementedError

        s1_scores = torch.cat(s1_scores, 1)  # (bsz, seqlen, vocab)
        _, preds = s1_scores.max(dim=2)

        return s1_scores, preds
