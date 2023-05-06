#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/31 17:40
# @Author  : zhangbc0315@outlook.com
# @File    : action_tagger.py
# @Software: PyCharm
import json
from typing import Iterator
from datetime import datetime

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pandas as pd
from cuspy import ConfigUtils
from fastode import FastXML
from synexlass.predictor import Predictor

from property_extractor.word_vec_utils import WordVecUtils
from utils.math_utils import MathUtils

nltk.download('omw-1.4')
nltk.download('wordnet')


class ActionTagger:

    wrong_vbs = []
    wrong_tokens = ['be', 'have']
    KNOWN_ACTS = ['VB-USE', 'VB-CHANGE', 'VB-SUBMERGE', 'VB-SUBJECT',
                  'NN-ADD', 'NN-MIXTURE', 'VB-DILUTE', 'VB-ADD', 'VB-CHARGE',             # ADD
                  'VB-CONTAIN', 'VB-DROP', 'VB-FILL', 'VB-SUSPEND', 'VB-TREAT',           #
                  'VB-APPARATUS', 'NN-APPARATUS', 'VB-CONCENTRATE', 'NN-CONCENTRATE',     # Apparatus
                  'VB-COOL',                                                              # Cool
                  'VB-DEGASS',                                                            # Degass
                  'VB-DISSOLVE',                                                          # Dissolve
                  'VB-DRY', 'NN-DRY',                                                     # Dry
                  'VB-EXTRACT', 'NN-EXTRACT',                                             # Extract
                  'VB-FILTER', 'NN-FILTER',                                               # Filter
                  'VB-HEAT', 'VB-INCREASE',                                               # Heat
                  'VB-IMMERSE',                                                           # Immerse
                  'VB-PARTITION',                                                         # Partition
                  'VB-PRECIPITATE', 'NN-PRECIPITATE',                                     # Precipitate
                  'VB-PURIFY', 'NN-PURIFY',                                               # Purify
                  'VB-QUENCH',                                                            # Quench
                  'VB-RECOVER',                                                           # Recover
                  'VB-REMOVE', 'NN-REMOVE',                                               # Remove
                  'VB-STIR',                                                              # Stir
                  'VB-SYNTHESIZE', 'NN-SYNTHESIZE',                                       # Synthesis
                  'VB-WAIT',                                                              # Wait
                  'VB-WASH',                                                              # Wash
                  'VB-YIELD']                                                             # Yield

    def __init__(self, config):
        self._text_df = pd.read_csv(config.texts_fp, sep='\t', encoding='utf-8')
        self._tagged_text_df = pd.read_csv(config.texts_tagged_fp, sep='\t', encoding='utf-8')
        self._vb_count_fp = config.action_vb_count_fp
        self._wnl = WordNetLemmatizer()
        self._predictor = Predictor()

        self._known_action_fp = config.known_action_fp
        self._unknown_verbs_fp = config.unknown_verbs_fp
        self._unknown_verb_stem_count_fp = config.unknown_verb_stem_count_fp
        self._unknown_verb_score_fp = config.unknown_verb_score_fp

        self._word_vec_utils = WordVecUtils(config)

    def get_vbs(self, xml):
        tagged_tokens, _ = FastXML.get_token_tag_pairs_and_attrs(xml, [], [])
        for tag, token in tagged_tokens:
            if tag.startswith('VB') and tag not in self.wrong_vbs:
                token = self._wnl.lemmatize(token, 'v')
                if token not in self.wrong_tokens:
                    yield token

    def get_action_and_vbs(self, xml):
        tagged_tokens, _ = FastXML.get_token_tag_pairs_and_attrs(xml, [], [])
        for tag, token in tagged_tokens:
            if tag in self.KNOWN_ACTS:
                yield token, tag, True
            elif tag.startswith('VB') and tag not in self.wrong_vbs:
                stem = self._wnl.lemmatize(token, 'v')
                if stem not in self.wrong_tokens:
                    yield token, stem.lower(), False

    def get_tagged_text_by_tid(self, tid: int):
        query_df = self._tagged_text_df.query(f"tid=={tid}")
        if len(query_df) == 0:
            return None
        query_df = query_df.reset_index()
        return query_df['xml'][0]

    def get_tagged_syn_text(self):
        """ 获得解析过的合成文本

        :return:
        """
        num_syn = 0
        with tqdm(total=len(self._text_df))as pbar:
            pbar.set_description("get tagged synthesis texts")
            for n, row in self._text_df.iterrows():
                # if n > 1000:
                #     break
                pbar.update(1)
                now_h = datetime.now().time().hour
                if now_h >= 14 or now_h == 2:
                    pbar.set_postfix_str(f"time is reach: {now_h}")
                    break
                pbar.set_postfix_str(f"num syn text: {num_syn}")
                if row['text_type'] == 'TITLE':
                    continue
                if self._predictor.predict_text(row['text']) == 0:
                    continue
                tid = row['tid']
                tagged_text = self.get_tagged_text_by_tid(tid)
                if tagged_text is None:
                    continue
                num_syn += 1
                yield tagged_text, tid
    # endregion

    # def process(self):
    #     vb_count = {}
    #     for xml_str, _ in self.get_tagged_syn_text():
    #         if xml_str is None:
    #             continue
    #         try:
    #             xml = FastXML.parse_string(xml_str)
    #         except:
    #             continue
    #         for vb in self.get_vbs(xml):
    #             if vb not in vb_count.keys():
    #                 vb_count[vb] = 1
    #             else:
    #                 vb_count[vb] += 1
    #     vb_count_df = pd.DataFrame({'VB': vb_count.keys(), 'count': vb_count.values()})
    #     vb_count_df = vb_count_df.sort_values(by='count', axis=0, ascending=False)
    #     vb_count_df.to_csv(self._vb_count_fp, sep='\t', encoding='utf-8', index=False)

    def process(self):
        known_vbs = {}
        unknown_verbs = {}
        unknown_verb_stem_count = {}
        for xml_str, _ in self.get_tagged_syn_text():
            if xml_str is None:
                continue
            try:
                xml = FastXML.parse_string(xml_str)
            except:
                continue
            for vb, vb_type, is_known in self.get_action_and_vbs(xml):
                vb = vb.lower()
                if is_known:
                    if vb_type not in known_vbs.keys():
                        known_vbs[vb_type] = [vb]
                    elif vb not in known_vbs[vb_type]:
                        known_vbs[vb_type].append(vb)
                else:
                    if vb_type not in unknown_verbs.keys():
                        unknown_verbs[vb_type] = [vb]
                    elif vb not in unknown_verbs[vb_type]:
                        unknown_verbs[vb_type].append(vb)
                    if vb_type not in unknown_verb_stem_count.keys():
                        unknown_verb_stem_count[vb_type] = 1
                    else:
                        unknown_verb_stem_count[vb_type] += 1

        with open(self._known_action_fp, 'w', encoding='utf-8')as f:
            json.dump(known_vbs, f)
        with open(self._unknown_verbs_fp, 'w', encoding='utf-8')as f:
            json.dump(unknown_verbs, f)

        sorted_stems = list(sorted(unknown_verb_stem_count.keys(), key=lambda x: unknown_verb_stem_count[x], reverse=True))
        sorted_counts = [unknown_verb_stem_count[x] for x in sorted_stems]
        verb_stem_count_df = pd.DataFrame({'verb_stem': sorted_stems, 'count': sorted_counts})
        verb_stem_count_df.to_csv(self._unknown_verb_stem_count_fp, sep='\t', encoding='utf-8', index=False)

    def _words_to_vs(self, key_words):
        key_vs = {}
        for key, words in key_words.items():
            vs = []
            for w in words:
                v = self._word_vec_utils.get_vec(w)
                if v is None:
                    continue
                vs.append(v)
            key_vs[key] = vs
            # key_vs[key] = [self._word_vec_utils.get_vec(w) for w in words]
        return key_vs

    def _get_closed_dis(self, vs1: [], vs2: []):
        min_cos = 2
        for v1 in vs1:
            for v2 in vs2:
                cos = abs(MathUtils.cos_similarity(v1, v2))
                if cos < min_cos:
                    min_cos = cos
        return min_cos

    def _get_closed_dis_to_word(self, vs1: [], word_vecs: {}):
        min_cos = 2
        best_word = None
        for w, vs2 in word_vecs.items():
            cos = self._get_closed_dis(vs1, vs2)
            if cos < min_cos:
                min_cos = cos
                best_word = w
        return best_word, min_cos

    def _query_count(self, df: pd.DataFrame, word: str) -> int:
        try:
            query_df = df.query(f"verb_stem == '{word}'")
        except:
            return None
        if len(query_df) == 0:
            return None
        return list(query_df.index)[0]

    def get_new_acts(self):
        with open(self._known_action_fp, 'r', encoding='utf-8')as f:
            known_vbs = json.load(f)
        with open(self._unknown_verbs_fp, 'r', encoding='utf-8')as f:
            unknown_vbs = json.load(f)
        known_vb_vecs = self._words_to_vs(known_vbs)
        unknown_vb_vecs = self._words_to_vs(unknown_vbs)

        verb_stem_count_df = pd.read_csv(self._unknown_verb_stem_count_fp, sep='\t', encoding='utf-8')
        res_data = {'verb_stem': [], 'count': [], 'known_act': [], 'score': []}
        e_num = 0
        with tqdm(total=len(unknown_vb_vecs))as pbar:
            for unknown_vb, unknown_vecs in unknown_vb_vecs.items():
                pbar.update(1)
                pbar.set_postfix_str(f"error: {e_num}")
                best_word, min_cos = self._get_closed_dis_to_word(unknown_vecs, known_vb_vecs)
                count = self._query_count(verb_stem_count_df, unknown_vb)
                if count is None:
                    e_num += 1
                    continue
                res_data['verb_stem'].append(unknown_vb)
                res_data['count'].append(count)
                res_data['known_act'].append(best_word)
                res_data['score'].append(min_cos)

        res_df = pd.DataFrame(res_data)
        res_df = res_df.sort_values(by='score', axis=0, ascending=True)
        res_df.to_csv(self._unknown_verb_score_fp, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    # ActionTagger(ConfigUtils.load_config('../config.json').proj_config).process()
    ActionTagger(ConfigUtils.load_config('../config.json').proj_config).get_new_acts()

    # w = WordNetLemmatizer()
    # print(w.lemmatize('centrifugation', 'n'))
