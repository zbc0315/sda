#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2/28/2023 3:12 PM
# @Author  : zhangbc0315@outlook.com
# @File    : nltk_parser.py
# @Software: PyCharm

import json
import re

from cuspy import ConfigUtils
import nltk
import requests
from chemdataextractor.doc import Paragraph, Sentence, Token
import nltk


class NLTKParser:

    _pos_tags = {'.': 'STOP',
                 ',': 'COMMA'}
    _grammar = "CM: {<B-CM><I-CM>*}\n" \
               "NUM: {<CD>(<DASH><CD>)*}\n" \
               "NUM: {<NUM>(<COMMA><NUM>)*<CC><NUM>}\n" \
               "NN-UNIT: {<UNIT-TIME|UNIT-MASS|UNIT-AMOUNT|UNIT-MOLAR|UNIT-EQ|UNIT-VOL|UNIT-TEMP|UNIT-PH|UNIT-TIMES|UNIT-OTHER|UNIT-ANGLE>}\n" \
               "PROP: {<NUM><NN-UNIT>}\n" \
               "PROP: {<STATE|TIME|TEMP|NN-VACUUM>}\n" \
               "NN-ACTION: {<NN-FLASH|NN-OTHER|NN-ADD|NN-MIXTURE|NN-APPARATUS|NN-CONCENTRATE|NN-DRY|NN-EXTRACT|NN-FILTER|NN-PRECIPITATE|NN-PURIFY|NN-REMOVE|NN-SYNTHESIZE>}\n" \
               "NN-ALL: {<NN|NNS|CM|NN-ACTION|NN-GENERAL|NN-METHOD|NN-PRESSURE|NN-COLUMN|NN-CHROMATOGRAPHY|NN-CYCLE|NN-EXAMPLE|NN-IDENTIFIER|CD-ALPHANUM|NN-ATMOSPHERE>}\n" \
               "NN-ALL: {<NUM><NN-ALL>}\n" \
               "NN-ALL: {<NN-ALL>+}\n"\
               "NN-ALL: {<PROPS|PROP>}\n" \
               "NN-ALL: {<CD-ALPHANUM|NUM>}\n" \
               "NN-ALL: {<NN-ALL>((<COMMA>?<CC><NN-ALL>)|(<COMMA><NN-ALL>))+}\n" \
               "NN-ALL: {<-LRB-><NN-ALL><-RRB->}\n" \
               "NN-ALL: {<NN-ALL>+}\n" \
               "VB-ACTION: {<VB-OTHER|VB-USE|VB-CHANGE|VB-SUBMERGE|VB-SUBJECT|VB-DILUTE|VB-ADD|VB-CHARGE|VB-CONTAIN|VB-DROP|VB-FILL|VB-SUSPEND|VB-TREAT|VB-COOL|VB-DEGASS|VB-DISSOLVE|VB-DRY|VB-EXTRACT|VB-FILTER|VB-HEAT|VB-INCREASE|VB-IMMERSE|VB-PARTITION|VB-PRECIPITATE|VB-PURIFY|VB-QUENCH|VB-RECOVER|VB-REMOVE|VB-STIR|VB-SYNTHESIZE|VB-WAIT|VB-WASH|VB-YIELD>}\n" \
               "VB-ALL: {<VB|VBD|VBP|VB-ACTION|VB-APPARATUS|VB-CONCENTRATE>}\n" \
               "ADJ: {<JJ|JJ-CHEM|JJ-COMPOUND>}\n" \
               "DT: {<DT|DT-THE>}\n" \
               "NounPhrase: {<DT>?<ADJ|JJ|RB>*(<NN-ALL>)}\n" \
               "NounPhrase: {<NounPhrase>((<COMMA><NounPhrase>)|(<COMMA>?<CC><NounPhrase>))+}\n" \
               "PrepPhrase: {<IN><NounPhrase>}\n" \
               "VerbPhrase: {(<NounPhrase|PrepPhrase>?<VBD>?<RB>?<VB-ALL>)|(<RB>?<VB-ALL><NounPhrase|PrepPhrase>)}\n" \
               "VerbPhrase: {<VerbPhrase><VerbPhrase>+}\n" \
               "VerbPhrase: {<IN><RB>?<VB-ALL>}\n" \
               "VerbPhrase: {<VerbPhrase><PrepPhrase>}"

    def __init__(self, regex_tag_fps: [str], dict_tag_fps: [str]):
        self._tag_to_regex = self._load_map_from_fps(regex_tag_fps)
        self._tag_to_str = self._load_map_from_fps(dict_tag_fps)
        self._rp = nltk.RegexpParser(self._grammar)

    # region ===== utils =====
    @classmethod
    def _load_map_from_fps(cls, fps: [str]):
        map_data = {}
        for fp in fps:
            map_data = cls._load_map_from_fp(fp, map_data)
        return map_data

    @staticmethod
    def _load_map_from_fp(fp: str, map_data: {} = None):
        map_data = {} if map_data is None else map_data
        with open(fp, 'r', encoding='utf-8')as f:
            for line in f.readlines():
                if line.startswith('#') or '---' not in line:
                    continue
                data = line.split('---')
                map_data[data[0]] = data[1]
        return map_data
    # endregion

    # region ===== data type process =====
    @classmethod
    def _get_tokens(cls, sent: Sentence):
        tokens = []
        for i, token in enumerate(sent.tokens):
            tokens.append({'text': token.text,
                           'start': token.start,
                           'end': token.end,
                           'index': i})
        return tokens
    # endregion

    # region ===== chem tagger =====
    @staticmethod
    def _chem_oscar_tag(tokens):
        res = requests.post('http://127.0.0.1:8089/tagTokensOscarCustomised', json={'tokens': tokens})
        res_json = json.loads(res.text)
        for i, tag in enumerate(res_json['tags']):
            if tag == 'OSCAR-CM':
                if i != 0 and tokens[i-1].get('tag') is not None and tokens[i-1].get('tag') in ['B-CM', 'I-CM']:
                    tokens[i]['tag'] = 'I-CM'
                else:
                    tokens[i]['tag'] = 'B-CM'
        return tokens

    @staticmethod
    def _chem_cde_tag(sent, tokens):
        for i, tag in enumerate(sent.ner_tags):
            if tag in ['B-CM', 'I-CM']:
                tokens[i]['tag'] = tag
        return tokens

    @classmethod
    def _chem_tag(cls, sent: Sentence, tokens: []):
        tokens = cls._chem_cde_tag(sent, tokens)
        tokens = cls._chem_oscar_tag(tokens)
        return tokens
    # endregion

    # region ===== POS tag =====
    @classmethod
    def _pos_tag(cls, sent: Sentence, tokens: []):
        for i, cde_token in enumerate(sent.tokens):
            pos_tag = cde_token.pos_tag
            if pos_tag in cls._pos_tags.keys():
                pos_tag = cls._pos_tags[pos_tag]
            tokens[i]['tag'] = pos_tag
        return tokens
    # endregion

    # region ===== regex tagger =====
    def _regex_tag(self, tokens: []):
        for token in tokens:
            if token['tag'] in ['B-CM', 'I-CM']:
                continue
            for tag, r in self._tag_to_regex.items():
                if re.match(r, token['text']):
                    token['tag'] = tag
        return tokens
    # endregion

    # region ===== syntax parser =====
    def _syntax_parser(self, tokens):
        token_tags = [(token['text'], token['tag']) for token in tokens]
        return self._rp.parse(token_tags)
    # endregion

    def parse(self, text: str):
        paragraph = Paragraph(text)
        for sent in paragraph.sentences:
            tokens = self._get_tokens(sent)
            tokens = self._pos_tag(sent, tokens)
            tokens = self._chem_tag(sent, tokens)
            tokens = self._regex_tag(tokens)
            tree = self._syntax_parser(tokens)
            tree.draw()
            print(1)


if __name__ == "__main__":
    s = ""
    conf = ConfigUtils.load_config('../config.json')
    NLTKParser([conf.comm_config.regex_tag_fp], []).parse(s)
