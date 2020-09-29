import copy
import os
import math
import json
import random
from operator import itemgetter
from orderedset import OrderedSet
from collections import defaultdict
import pickle

from parlai.utils.misc import warn_once, str_to_msg
from parlai.core.message import Message
from parlai.core.torch_agent import TorchAgent
from parlai.core.teachers import (
    ParlAIDialogTeacher,
    FixedDialogTeacher,
    FbDeprecatedDialogTeacher,
    DialogData
)

from .build import build, make_path

__PATH__ = os.path.abspath(os.path.dirname(__file__))


def _path(opt):
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        warn_once("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    return make_path(opt, datatype + '.txt'), datatype


def _split_persona_and_context(text, eval_type='convai2'):
    if 'your persona:' not in text:
        return None, text
    else:
        if eval_type == 'convai2':
            texts = text.split('\n')
            return '\n'.join(texts[:-1]), texts[-1]
        elif eval_type =='dnli':
            texts = text.split('\n')
            last_idx = 0
            for idx, text in enumerate(texts):
                if 'your persona:' in text:
                    last_idx = idx
            persona_texts = texts[:last_idx+1]
            context_texts = texts[last_idx+1:]
            return '\n'.join(persona_texts), '\n'.join(context_texts)


def _split_personas_and_context(text):
    if 'your persona:' not in text:
        return text, text, text
    else:
        your_personas = []
        partner_personas = []
        context = []
        texts = text.split('\n')
        for text in texts:
            if text.startswith('your persona:'):
                your_personas.append(text)
            elif text.startswith("partner's persona:"):
                partner_personas.append(text)
            else:
                context.append(text)

        return '\n'.join(your_personas), '\n'.join(partner_personas), context


class SelfConsciousDialogueTeacher(FixedDialogTeacher):
    """
     Teacher (i.e. input data supplier) for the Self-conscious Agent.
     SelfConsciousDialogueTeacher (SCDT) supplies data input
     along with the distractors to the Self-conscious Agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt

        datapath, datatype = _path(opt)

        if not shared:
            self.episodes = []
            self.num_exs = 0
            self._setup_data(datapath, datatype)
        else:
            self.episodes = shared['episodes']
            self.num_exs = sum(len(e) for e in self.episodes)
        self.id = 'self_conscious_dialogue'
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Self Conscious Dialogue Teacher arguments')
        agent.add_argument(
            '--eval-type',
            type=str,
            choices=['convai2', 'dnli'],
            default='dnli',
            help='Which validation data to use',
        )

    def _setup_data(self, path, datatype):

        random.seed(46)

        # Data loading with script of ParlAIDialogTeacher
        print(f"[Loading ParlAI text data: {path}]")

        # Read data from ConvAI2
        convai2_datapath = make_path(self.opt, f'{datatype}_both_original.txt')
        convai2_episodes = self._load_convai2_data(convai2_datapath)

        # Get persona pool
        all_personas, persona_to_idx = self._get_persona_pool(self.opt)
        sorted_personas = self._get_sorted_persona_pool(datatype)


        if self.opt['eval_type'] == 'convai2':
            self.episodes = []
            self.num_exs = 0
            eps = []
            with open(path) as read:
                for line in read:
                    msg = str_to_msg(line.rstrip('\n'))
                    if msg:
                        self.num_exs += 1
                        eps.append(msg)
                        if msg.get('episode_done', False):
                            self.episodes.append(eps)
                            eps = []
            if len(eps) > 0:
                # add last episode
                eps[-1].force_set('episode_done', True)
                self.episodes.append(eps)
            # Add label candidates and partner's persona
            for episode_idx, episode in enumerate(self.episodes):
                for turn_idx, turn in enumerate(episode):
                    convai2_turn = convai2_episodes[episode_idx][turn_idx]
                    convai2_text = convai2_turn[0]
                    label_candidates = convai2_turn[3]

                    turn['label_candidates'] = label_candidates
                    if turn_idx == 0:
                        my_persona, partner_persona, _ = _split_personas_and_context(convai2_text)
                        turn['partner_persona'] = partner_persona
                        turn['my_persona'] = my_persona
                    else:
                        turn['partner_persona'] = episode[0]['partner_persona']
                        turn['my_persona'] = episode[0]['my_persona']
        elif self.opt['eval_type'] == 'dnli':
            self.episodes = []
            self.num_exs = 0
            for eval_set in ['attributes', 'havenot', 'likedislike']:
                datapath = make_path(self.opt, f'{datatype}_{eval_set}.jsonl')
                with open(datapath, 'r') as fp:
                    for line in fp:
                        msg = json.loads(line)
                        msg['eval_set'] = eval_set
                        msg['episode_done'] = True

                        # Make 'text'
                        persona_lines = [f'your persona: {x[:-2]}.' for x in msg['persona']]
                        utts = msg['prefix']

                        p1_token, p2_token = TorchAgent.P1_TOKEN, TorchAgent.P2_TOKEN
                        lines = persona_lines
                        # Identify the dialogue lines. It's assumed that p1 goes first.
                        for i, utt in enumerate(utts):
                            if i % 2 == 0:
                                lines.append(f'{p1_token} {utt}')
                            else:
                                lines.append(f'{p2_token} {utt}')
                        text = '\n'.join(lines)

                        msg['text'] = text

                        # Make 'label_candidates'
                        cands = msg['candidates']
                        msg['label_candidates'] = cands['label'] + cands['neg'][:10] \
                            + cands['similar'][:10] + cands['rand'][:10]

                        # Remove unused attributes
                        del msg['persona']
                        del msg['prefix']
                        del msg['triple']
                        del msg['relevant_persona_sentence']
                        del msg['candidates']

                        self.episodes.append([msg])
                        self.num_exs += 1

        # Add distractor personas
        if self.opt['world_cardinality'] > 0:
            num_all_personas = len(all_personas)
            persona_indices = list(range(num_all_personas))
            world_cardinality = self.opt['world_cardinality']
            for episode in self.episodes:
                gt_persona, first_context = _split_persona_and_context(episode[0]['text'], self.opt['eval_type'])
                gt_persona_idx = persona_to_idx.get(gt_persona, -1)

                # Choose random distractor personas
                distractor_indices = random.sample(persona_indices, world_cardinality - 1)
                while gt_persona_idx in distractor_indices:
                    # Resample if gt_persona is sampled
                    distractor_indices = random.sample(persona_indices, world_cardinality - 1)
                distractor_personas = itemgetter(*distractor_indices)(all_personas)
                distractor_personas = list(distractor_personas)

                # Make it to 'distractor_text'
                for turn_idx, turn in enumerate(episode):
                    if turn_idx == 0:
                        turn['distractor_text'] = [
                            '\n'.join([persona, first_context])
                            for persona in [gt_persona] + distractor_personas
                        ]
                    else:
                        turn['distractor_text'] = [turn['text']] * world_cardinality

    def _get_persona_pool(self, opt, remove_duplicate=True):
        print("[loading persona pool from convai2 training data]")
        # Get episodes from training dataset
        datapath = make_path(opt, 'train.txt')
        episodes = []
        eps = []
        with open(datapath) as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    # self.num_exs += 1
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1].force_set('episode_done', True)
            episodes.append(eps)

        # Extract personas from episodes
        persona_set = OrderedSet()
        for episode in episodes:
            first_turn = episode[0]
            text = first_turn['text']
            persona, _ = _split_persona_and_context(text)
            persona_set.add(persona)

        # Remove duplicate
        if remove_duplicate:
            train_persona_fname = os.path.join(__PATH__, 'train_persona_map.pkl')
            with open(train_persona_fname, 'rb') as fp:
                _train_personas = pickle.load(fp)
            train_personas = []
            for personas in _train_personas.values():
                longest_idx = 0
                longest_length = -1
                for idx, persona in enumerate(personas):
                    if len(persona) > longest_length:
                        longest_idx = idx
                        longest_length = len(persona)
                selected_persona = map(lambda x: f"your persona: {x}.",personas[longest_idx])
                selected_persona = '\n'.join(selected_persona)
                train_personas.append(selected_persona)
            persona_set = OrderedSet()
            for train_persona in train_personas:
                persona_set.add(train_persona)

        all_personas = []
        persona_to_idx = {}
        for i, persona in enumerate(persona_set):
            all_personas.append(persona)
            persona_to_idx[persona] = i

        print(f"Total {len(all_personas)} personas in dataset")

        return all_personas, persona_to_idx

    def _get_sorted_persona_pool(self, datatype):
        print("[loading sorted persona pool from convai2 training data]")
        eval_type = self.opt['eval_type']
        if eval_type == 'convai2':
            datapath = make_path(self.opt, 'valid_sorted_50_personas.json')
        elif eval_type == 'dnli':
            datapath = make_path(self.opt, 'dnli_sorted_50_personas.json')
        else:
            raise ValueError("eval_set must be one of convai2 and dnli")

        with open(datapath, 'r') as fp:
            sorted_personas = json.load(fp)
        sorted_personas['idx2persona'] = sorted_personas['train_personas']
        sorted_personas['persona2idx'] = {}
        for idx, persona in enumerate(sorted_personas['train_personas']):
            sorted_personas['persona2idx'][persona] = idx

        return sorted_personas

    def _load_convai2_data(self, datapath):
        """
        Read data in the fbdialog format.
        Returns ``(x, y, r, c)`` tuples.
        ``x`` represents a query, ``y`` represents the labels, ``r`` represents
        any reward, and ``c`` represents any label_candidates.
        The example above will be translated into the following tuples:
        ::
            x: 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?'
            y: ['kitchen']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = True (this is the first example in the episode)
        ::
            x: 'Sam went to the hallway\\nPat went to the bathroom\\nWhere is the
                milk?'
            y: ['hallway']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = False (this is the second example in the episode)
        """
        self.cloze = False  # Set this to use FbDialogTeacher
        convai2_dataloader = FbDeprecatedDialogTeacher.setup_data(self, datapath)
        convai2_episodes = []
        for episode in DialogData._read_episode(self, convai2_dataloader):
            convai2_episodes.append(episode)
        del self.cloze
        return convai2_episodes

    def share(self):
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def num_examples(self):
        return self.num_exs

    def num_episodes(self):
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=None):
        return self.episodes[episode_idx][entry_idx]


class ContextConsciousDialogueTeacher(SelfConsciousDialogueTeacher):
    def _setup_data(self, path, datatype):
        # random.seed(self.opt['random_seed'])  # Set this for pick same distractor persona
        random.seed(46)  # Set this for pick same distractor persona
        # Data loading with script of ParlAIDialogTeacher
        print(f"[Loading ParlAI text data: {path}]")

        # Read data from ConvAI2
        convai2_datapath = make_path(self.opt, f'{datatype}_both_original.txt')
        convai2_episodes = self._load_convai2_data(convai2_datapath)

        if self.opt['eval_type'] == 'convai2':
            self.episodes = []
            self.num_exs = 0
            eps = []
            with open(path) as read:
                for line in read:
                    msg = str_to_msg(line.rstrip('\n'))
                    if msg:
                        self.num_exs += 1
                        eps.append(msg)
                        if msg.get('episode_done', False):
                            self.episodes.append(eps)
                            eps = []
            if len(eps) > 0:
                # add last episode
                eps[-1].force_set('episode_done', True)
                self.episodes.append(eps)
            # Add label candidates and partner's persona
            for episode_idx, episode in enumerate(self.episodes):
                for turn_idx, turn in enumerate(episode):
                    convai2_turn = convai2_episodes[episode_idx][turn_idx]
                    convai2_text = convai2_turn[0]
                    label_candidates = convai2_turn[3]

                    turn['label_candidates'] = label_candidates
                    if turn_idx == 0:
                        my_persona, partner_persona, _ = _split_personas_and_context(convai2_text)
                        turn['partner_persona'] = partner_persona
                        turn['my_persona'] = my_persona
                    else:
                        turn['partner_persona'] = episode[0]['partner_persona']
                        turn['my_persona'] = episode[0]['my_persona']
        elif self.opt['eval_type'] == 'dnli':
            self.episodes = []
            self.num_exs = 0
            for eval_set in ['attributes', 'havenot', 'likedislike']:
                datapath = make_path(self.opt, f'{datatype}_{eval_set}.jsonl')
                with open(datapath, 'r') as fp:
                    for line in fp:
                        msg = json.loads(line)
                        msg['eval_set'] = eval_set
                        msg['episode_done'] = True

                        # Make 'text'
                        persona_lines = [f'your persona: {x[:-2]}.' for x in msg['persona']]
                        utts = msg['prefix']

                        p1_token, p2_token = TorchAgent.P1_TOKEN, TorchAgent.P2_TOKEN
                        lines = persona_lines
                        # Identify the dialogue lines. It's assumed that p1 goes first.
                        for i, utt in enumerate(utts):
                            if i % 2 == 0:
                                lines.append(f'{p1_token} {utt}')
                            else:
                                lines.append(f'{p2_token} {utt}')
                        text = '\n'.join(lines)

                        msg['text'] = text

                        # Make 'label_candidates'
                        cands = msg['candidates']
                        msg['label_candidates'] = cands['label'] + cands['neg'][:10] \
                            + cands['similar'][:10] + cands['rand'][:10]

                        # Remove unused attributes
                        del msg['persona']
                        del msg['prefix']
                        del msg['triple']
                        del msg['relevant_persona_sentence']
                        del msg['candidates']

                        self.episodes.append([msg])
                        self.num_exs += 1

        # Get dialogue history pool
        context_pool = self._get_context_pool(self.opt)

        # Add distractor history
        if self.opt['world_cardinality'] > 0:
            for episode in self.episodes:
                gt_persona, first_context = _split_persona_and_context(episode[0]['text'], self.opt['eval_type'])

                # Select distractor history
                if self.opt['eval_type'] == 'convai2':
                    num_turn = len(episode)
                else:
                    dialogue = first_context.split('\n')
                    num_turn = math.ceil(len(dialogue)/2)
                    if num_turn < min(context_pool.keys()):
                        # orginal_num_turn = num_turn
                        num_turn = min(context_pool.keys())

                context_indices = list(range(len(context_pool[num_turn])))

                distractor_c_indices = random.sample(context_indices, self.opt['world_cardinality'] - 1)
                distractor_contexts = itemgetter(*distractor_c_indices)(context_pool[num_turn])

                # Make it to 'distractor_text'
                if self.opt['eval_type'] == 'convai2':
                    for turn_idx, turn in enumerate(episode):
                        turn['distractor_text'] = turn['labels'] + [c[turn_idx] for c in distractor_contexts]
                        if turn_idx == 0:
                            turn['my_context'] = turn['labels']
                        else:
                            turn['my_context'] = episode[turn_idx - 1]['my_context'] + turn['labels']
                else:
                    # DNLI
                    distractor_text = [episode[0]['text']]
                    for c in distractor_contexts:
                        copied_dialogue = copy.deepcopy(dialogue)
                        for turn_idx, utterance in enumerate(copied_dialogue):
                            if turn_idx % 2 == 1:
                                copied_dialogue[turn_idx] = p2_token + c[turn_idx // 2]
                        distractor_context = '\n'.join([gt_persona] + copied_dialogue)
                        distractor_text.append(distractor_context)
                    episode[0]['distractor_text'] = distractor_text

    def _get_context_pool(self, opt):
        print("[loading history pool from convai2 training data]")
        datapath = make_path(opt, 'train.txt')
        episodes = []
        eps = []
        with open(datapath) as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1].force_set('episode_done', True)
            episodes.append(eps)

        context_pool = defaultdict(list)
        for ep in episodes:
            context_pool[len(ep)].append([turn['labels'][0] for turn in ep])

        return dict(context_pool)


class DefaultTeacher(SelfConsciousDialogueTeacher):
    pass
