#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is derived from parlai/core/seq2seq/seq2seq.py.
In particular, it's derived from an older version that inherits from TorchAgent rather
than TorchGeneratorAgent.
It should be possible to refactor this file to be comparable to the current
parlai/core/seq2seq/seq2seq.py, i.e. inherit from TorchGeneratorAgent - this would
probably reduce the amount of boilerplate in this file.
However, for simplicity and to keep things as similar as possible to the version used
for the paper, we have kept this file mostly the same.
"""

from parlai.core.torch_agent import Batch, History, TorchAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.torch import padded_tensor, argsort
# from .base_controllable_seq2seq import BaseControllableSeq2seqAgent
# from .util import ConvAI2History
# from .controls import get_ctrl_vec

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict, namedtuple, Counter, deque
from operator import attrgetter

import os
import math
import json
import tempfile
import copy
from itertools import chain


def list_to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


class SelfConsciousHistory(History):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        opt = args[0]
        if opt['eval_type'] == 'convai2':
            self.add_person_tokens = True
        elif opt['eval_type'] == 'dnli':
            self.add_person_tokens = False
        else:
            raise ValueError

        self.world_cardinality = opt.get('world_cardinality', 5)
        self.history_distractor_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_raw_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_vecs = [[] for _ in range(self.world_cardinality)]
        # Will be used for TransferTransfo
        self.history_token_type_ids = []
        self.history_distractor_token_type_ids = [[] for _ in range(self.world_cardinality)]

    def reset(self):
        """Clear the history"""
        super().reset()
        self.history_distractor_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_raw_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_vecs = [[] for _ in range(self.world_cardinality)]
        self.history_token_type_ids = []
        self.history_distractor_token_type_ids = [[] for _ in range(self.world_cardinality)]

    def _update_distractor_strings(self, text, idx):
        history_strings = self.history_distractor_strings[idx]
        if self.size > 0:
            while len(history_strings) >= self.size:
                history_strings.pop(0)
        history_strings.append(text)

    def _update_distractor_raw_strings(self, text, idx):
        history_raw_strings = self.history_distractor_raw_strings[idx]
        if self.size > 0:
            while len(history_raw_strings) >= self.size:
                history_raw_strings.pop(0)
        history_raw_strings.append(text)

    def _update_distractor_vecs(self, text, idx):
        history_vecs = self.history_distractor_vecs[idx]
        if self.size > 0:
            while len(history_vecs) >= self.size:
                history_vecs.pop(0)
        history_vecs.append(self.parse(text))

    def _update_token_type_ids(self, text, idx):
        pass

    def add_reply_to_distractors(self, model_reply):

        # Update model's response to the history
        if model_reply is not None:
            for idx in range(self.world_cardinality):
                self._update_distractor_raw_strings(model_reply, idx)
                # this is causing the repetition of p2 token.
                # need to do this only once. not every loop
                if self.add_person_tokens and idx == 0:
                    model_reply = self._add_person_tokens(model_reply, self.p2_token)
                self._update_distractor_strings(model_reply, idx)
                self._update_distractor_vecs(model_reply, idx)

    # def update_history(self, obs, add_next=None):
    def update_history(self, obs, temp_history=None):
        """
        Update the history with the given observation.
        :param add_next:
            string to append to history prior to updating it with the
            observation
        """
        # super().update_history(obs, add_next)
        super().update_history(obs, temp_history=temp_history)

        # Update previous turn's my response
        # if add_next is not None:
            # for idx in range(self.world_cardinality):
                # self._update_distractor_raw_strings(add_next, idx)
                # # this is causing the repetition of p2 token.
                # # need to do this only once. not every loop
                # if self.add_person_tokens and idx == 0:
                    # add_next = self._add_person_tokens(add_next, self.p2_token)
                # self._update_distractor_strings(add_next, idx)
                # self._update_distractor_vecs(add_next, idx)

        # Update current turn's opponent's response
        if 'distractor_text' in obs:
            assert len(obs['distractor_text']) == self.world_cardinality, \
                f"Numer of distractor_text must be eqaul to world_cardinality. ({len(obs['distractor_text'])} vs {self.world_cardinality})"
            for idx, distractor_text in enumerate(obs['distractor_text']):
                if self.split_on_newln:
                    next_texts = distractor_text.split('\n')
                else:
                    next_texts = [distractor_text]
                for text in next_texts:
                    self._update_distractor_raw_strings(text, idx)
                    if self.add_person_tokens:
                        text = self._add_person_tokens(
                            distractor_text, self.p1_token, self.add_p1_after_newln
                        )
                    self._update_distractor_strings(text, idx)
                    self._update_distractor_vecs(text, idx)

    def get_history_distractor_str(self):
        """Return the list of string version of the distractor histories."""
        if len(self.history_distractor_strings[0]) > 0:
            return [
                self.delimiter.join(history_strings)
                for history_strings in self.history_distractor_strings
            ]
        return None

    def get_history_distractor_vec(self):
        """Return a vectorized version of the distractor histories."""
        if len(self.history_distractor_vecs[0]) == 0:
            return None

        histories = []
        for idx in range(self.world_cardinality):
            history_vecs = self.history_distractor_vecs[idx]

            # if self.vec_type == 'deque':
                # history = deque(maxlen=self.max_len)
                # for vec in history_vecs[:-1]:
                    # history.extend(vec)
                    # history.extend(self.delimiter_tok)
                # history.extend(history_vecs[-1])
            # else:
            # vec type is a list
            history = []
            for vec in history_vecs[:-1]:
                history += vec
                history += self.delimiter_tok
            history += history_vecs[-1]

            histories.append(history)
        return histories

    def get_token_type_ids(self):
        """
        Return a vectorized version of the token_type_ids and
        distractor_token_type_ids
        """
        pass


class ContextConsciousHistory(History):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        opt = args[0]
        if opt['eval_type'] == 'convai2':
            self.add_person_tokens = True
        elif opt['eval_type'] == 'dnli':
            self.add_person_tokens = False
        else:
            raise ValueError

        self.world_cardinality = opt.get('world_cardinality', 5)
        self.history_distractor_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_raw_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_vecs = [[] for _ in range(self.world_cardinality)]
        # Will be used for TransferTransfo
        self.history_token_type_ids = []
        self.history_distractor_token_type_ids = [[] for _ in range(self.world_cardinality)]
        self.eval_type = opt.get('eval_type')

    def reset(self):
        """Clear the history"""
        super().reset()
        self.history_distractor_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_raw_strings = [[] for _ in range(self.world_cardinality)]
        self.history_distractor_vecs = [[] for _ in range(self.world_cardinality)]
        self.history_token_type_ids = []
        self.history_distractor_token_type_ids = [[] for _ in range(self.world_cardinality)]

    def _update_distractor_strings(self, text, idx):
        history_strings = self.history_distractor_strings[idx]
        if self.size > 0:
            while len(history_strings) >= self.size:
                history_strings.pop(0)
        history_strings.append(text)

    def _update_distractor_raw_strings(self, text, idx):
        history_raw_strings = self.history_distractor_raw_strings[idx]
        if self.size > 0:
            while len(history_raw_strings) >= self.size:
                history_raw_strings.pop(0)
        history_raw_strings.append(text)

    def _update_distractor_vecs(self, text, idx):
        history_vecs = self.history_distractor_vecs[idx]
        if self.size > 0:
            while len(history_vecs) >= self.size:
                history_vecs.pop(0)
        history_vecs.append(self.parse(text))

    def _update_token_type_ids(self, text, idx):
        pass

    def add_reply_to_distractors(self, model_reply, obs=None):

        # Update model's response along with distractor responses to the history
        if model_reply is not None and 'distractor_text' in obs:
            distractor_responses = obs['distractor_text']
            assert len(obs['distractor_text']) == self.world_cardinality

            for idx in range(self.world_cardinality):
                self._update_distractor_raw_strings(distractor_responses[idx], idx)
                if self.add_person_tokens:
                    distractor_responses[idx] = self._add_person_tokens(distractor_responses[idx], self.p2_token)
                self._update_distractor_strings(distractor_responses[idx], idx)
                self._update_distractor_vecs(distractor_responses[idx], idx)

    # def update_history(self, obs, add_next=None):
    def update_history(self, obs, temp_history=None):
        """
        Update the history with the given observation.
        :param add_next:
            string to append to history prior to updating it with the
            observation
        """
        super().update_history(obs, temp_history=temp_history)

        # Update current turn's opponent's response
        if self.eval_type == 'convai2':
            if 'text' in obs:
                for idx in range(self.world_cardinality):
                    if self.split_on_newln:
                        next_texts = obs['text'].split('\n')
                    else:
                        next_texts = [obs['text']]
                    for text in next_texts:
                        self._update_distractor_raw_strings(text, idx)
                        if self.add_person_tokens:
                            text = self._add_person_tokens(
                                obs['text'], self.p1_token, self.add_p1_after_newln
                            )
                        self._update_distractor_strings(text, idx)
                        self._update_distractor_vecs(text, idx)
        else:
            if 'distractor_text' in obs:
                distractor_texts = obs['distractor_text']
                for idx, distractor in enumerate(distractor_texts):
                    self._update_distractor_raw_strings(distractor, idx)
                    self._update_distractor_strings(distractor, idx)
                    self._update_distractor_vecs(distractor, idx)

    def get_history_distractor_str(self):
        """Return the list of string version of the distractor histories."""
        if len(self.history_distractor_strings[0]) > 0:
            return [
                self.delimiter.join(history_strings)
                for history_strings in self.history_distractor_strings
            ]
        return None

    def get_history_distractor_vec(self):
        """Return a vectorized version of the distractor histories."""
        if len(self.history_distractor_vecs[0]) == 0:
            return None

        histories = []
        for idx in range(self.world_cardinality):
            history_vecs = self.history_distractor_vecs[idx]

            # if self.vec_type == 'deque':
                # history = deque(maxlen=self.max_len)
                # for vec in history_vecs[:-1]:
                    # history.extend(vec)
                    # history.extend(self.delimiter_tok)
                # history.extend(history_vecs[-1])
            # else:
            # vec type is a list
            history = []
            for vec in history_vecs[:-1]:
                history += vec
                history += self.delimiter_tok
            history += history_vecs[-1]

            histories.append(history)
        return histories

    def get_token_type_ids(self):
        """
        Return a vectorized version of the token_type_ids and
        distractor_token_type_ids
        """
        pass
