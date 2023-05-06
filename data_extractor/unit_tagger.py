#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 12/6/2022 2:00 PM
# @Author  : zhangbc0315@outlook.com
# @File    : unit_tagger.py
# @Software: PyCharm
import os.path

import pandas as pd
import networkx as nx
from cuspy import ConfigUtils


class UnitTagger:

    MIN_COUNT = 10
    DEBUG_FID = 0
    WRONG_TAGS = ['VBZ', 'VBD', 'VBN', 'VBP',
                  'IN', 'IN-IN', 'IN-OF', 'IN-OVER', 'IN-FOR', 'IN-WITH', 'IN-UNDER',
                  'DT-THE',
                  'TO', 'COMMA', 'STOP', 'CC', 'WRB', 'COLON']
    WRONG_WORDS = ['elsevier']

    def __init__(self, config):
        self._sent_with_cd_df = pd.read_csv(config.tag_token_pairs_fp, sep='\t', encoding='utf-8')
        self._units_fp = config.units_fp
        self._filter_units_fp = config.filter_units_fp
        self.DEBUG_DP = config.debug_dp
        if not os.path.exists(self.DEBUG_DP):
            os.mkdir(self.DEBUG_DP)

    @classmethod
    def _split_tokens_by_cd(cls, tagged_tokens):
        tag_tokens_list = []
        for i, (tag, token) in enumerate(tagged_tokens):
            for ts in tag_tokens_list:
                ts.append((tag, token))
            if tag == 'CD':
                tag_tokens_list.append([])
        return tag_tokens_list

    @classmethod
    def _contain_cd(cls, tagged_tokens):
        if len(tagged_tokens) == 0:
            return False
        tags, tokens = list(zip(*tagged_tokens))
        return 'CD' in tags

    def _count_unit(self):
        unit_tag_tokens = {}
        for _, row in self._sent_with_cd_df.iterrows():
            tagged_tokens = eval(row.tag_token_pairs)
            if not self._contain_cd(tagged_tokens):
                continue
            candidate_units_list = self._split_tokens_by_cd(tagged_tokens)
            for candidate_units in candidate_units_list:
                if len(candidate_units) == 0:
                    continue
                if candidate_units[0][1] not in unit_tag_tokens.keys():
                    unit_tag_tokens[candidate_units[0][1]] = [candidate_units]
                else:
                    unit_tag_tokens[candidate_units[0][1]].append(candidate_units)
        return unit_tag_tokens

    @classmethod
    def _create_graph(cls, unit_tokens):
        graph = nx.DiGraph()
        graph.add_node(0, unit='', unit_tag='', tokens=[], level=0, terminal=False)
        i = 1
        for unit, tag_tokens in unit_tokens.items():
            if len(tag_tokens) < 10:
                continue
            graph.add_node(i, unit=unit, unit_tag=tag_tokens[0][0][0], tokens=tag_tokens, level=1, terminal=False)
            graph.add_edge(0, i)
            i += 1
        return graph, i

    @classmethod
    def _is_leaf(cls, graph: nx.DiGraph, node_idx: int):
        return len(list(graph.neighbors(node_idx))) == 0

    @classmethod
    def need_right_square(cls, text, left_square, right_square):
        need = 0
        for c in text:
            if c == left_square:
                need += 1
            elif c == right_square and need >= 1:
                need -= 1
        return need > 0

    @classmethod
    def is_numeric(cls, word, tag):
        if tag == 'CD':
            if '.' in word:
                return True
            if not word.isdigit():
                return True
            num = int(word)
            if num >= 5:
                return True
            return False
        else:
            return False

    @classmethod
    def _split_tokens(cls, tag_tokens_list, level):
        unit_tokens = {}
        for tag_tokens in tag_tokens_list:

            tags, tokens = list(zip(*tag_tokens))
            old_unit = ' '.join(tokens[:level])
            new_unit = old_unit
            if len(tag_tokens) < level + 1:
                pass
            elif tag_tokens[level][0] in cls.WRONG_TAGS:
                pass
            elif cls.is_numeric(tag_tokens[level][1], tag_tokens[level][0]):
                pass
            elif tag_tokens[level][1] == ')' and not cls.need_right_square(old_unit, '(', ')'):
                pass
            elif tag_tokens[level][1] == ']' and not cls.need_right_square(old_unit, '[', ']'):
                pass
            elif tag_tokens[level][1] == '}' and not cls.need_right_square(old_unit, '{', '}'):
                pass
            else:
                new_unit = ' '.join(tokens[:level+1])
            if new_unit not in unit_tokens.keys():
                unit_tokens[new_unit] = [tag_tokens]
            else:
                unit_tokens[new_unit].append(tag_tokens)
        return unit_tokens

    @classmethod
    def generate_child(cls, graph: nx.DiGraph, i):
        stop = True
        childs = {}
        for node_idx in graph.nodes:
            if not cls._is_leaf(graph, node_idx):
                continue
            if graph.nodes[node_idx]['terminal']:
                continue
            node_data = graph.nodes[node_idx]
            unit_tokens = cls._split_tokens(node_data['tokens'], node_data['level'])
            has_child = False
            childs[node_idx] = []
            old_unit = node_data['unit']
            for unit, tag_tokens in unit_tokens.items():
                if len(tag_tokens) < cls.MIN_COUNT:
                    continue
                else:
                    has_child = True
                    stop = False
                    tags, tokens = list(zip(*tag_tokens[0]))
                    new_level = node_data['level'] if unit == old_unit else node_data['level'] + 1
                    new_unit_tag = ' '.join(tags[:new_level])
                    childs[node_idx].append({'unit': unit, 'unit_tag': new_unit_tag, 'tokens': tag_tokens,
                                             'level': new_level, 'terminal': unit == old_unit})
            if not has_child:
                graph.nodes[node_idx]['terminal'] = True
        for idx, child in childs.items():
            for c in child:
                graph.add_node(i, **c)
                graph.add_edge(idx, i)
                i += 1
        return graph, stop, i

    @classmethod
    def _graph_to_texts(cls, graph: nx.DiGraph, node_idx: int):
        node_data = graph.nodes[node_idx]
        tab = '\t'*node_data['level']
        unit = node_data['unit']
        unit_tag = node_data['unit_tag']
        terminal = node_data['terminal']
        level = node_data['level']
        count = len(node_data['tokens'])
        yield f"{tab}{node_idx}/{level}:【{unit}】【{unit_tag}】 - {count}/{terminal}"
        for child in graph.neighbors(node_idx):
            for line in cls._graph_to_texts(graph, child):
                yield line

    def _graph_to_txt_fp(self, graph: nx.DiGraph):
        txt_fp = os.path.join(self.DEBUG_DP, f"{self.DEBUG_FID}.txt")
        self.DEBUG_FID += 1
        lines = list(self._graph_to_texts(graph, 0))
        with open(txt_fp, 'w', encoding='utf-8')as f:
            f.write('\n'.join(lines))

    @classmethod
    def _extract_units(cls, graph: nx.DiGraph, tsv_fp: str):
        unit_data = {'unit': [], 'tag': [], 'count': [], 'level': []}
        for node_idx in graph.nodes:
            if not cls._is_leaf(graph, node_idx):
                continue
            node_data = graph.nodes[node_idx]
            unit_data['unit'].append(node_data['unit'])
            unit_data['tag'].append(node_data['unit_tag'])
            unit_data['count'].append(len(node_data['tokens']))
            unit_data['level'].append(node_data['level'])
        unit_data_df = pd.DataFrame(unit_data)
        unit_data_df = unit_data_df.sort_values('count', ascending=False)
        unit_data_df.to_csv(tsv_fp, sep='\t', encoding='utf-8', index=False)
        # return sort_unit_count

    @classmethod
    def check_square(cls, text: str, ls: str, rs: str):
        num = 0
        for c in text:
            if c == ls:
                num += 1
            elif c == rs:
                num -= 1
            if num < 0:
                return False
        return num == 0

    @classmethod
    def _is_right_unit(cls, ser: pd.Series) -> bool:
        if ser['count'] < 100:
            return False

        unit = ser.unit
        if not cls.check_square(unit, '(', ')'):
            return False
        if not cls.check_square(unit, '[', ']'):
            return False
        if not cls.check_square(unit, '{', '}'):
            return False
        if all([len(token) >= 4 for token in unit.split(' ')]):
            return False
        if unit[0].isdigit():
            return False
        if any([word in unit.lower() for word in cls.WRONG_WORDS]):
            return False

        tags = ser.tag.split(' ')
        for et in cls.WRONG_TAGS:
            if et in tags:
                return False

        return True

    @classmethod
    def _filter_units(cls, unit_fp: str, filter_unit_fp: str):
        unit_df = pd.read_csv(unit_fp, sep='\t', encoding='utf-8')
        unit_df = unit_df.loc[unit_df.apply(cls._is_right_unit, axis=1)]
        unit_df.to_csv(filter_unit_fp, sep='\t', encoding='utf-8', index=False)

    def tag(self):
        unit_tag_tokens = self._count_unit()
        graph, now_i = self._create_graph(unit_tag_tokens)
        stop = False
        self._graph_to_txt_fp(graph)
        while not stop:
            graph, stop, now_i = self.generate_child(graph, now_i)
            self._graph_to_txt_fp(graph)
        self._extract_units(graph, self._units_fp)
        self._filter_units(self._units_fp, self._filter_units_fp)


if __name__ == "__main__":
    # UnitTagger(ConfigUtils.load_config('../config.json').proj_peroxidase_config).tag()
    # UnitTagger(ConfigUtils.load_config('../config.json').proj_pvk_config).tag()
    # UnitTagger(ConfigUtils.load_config('../config.json').proj_ligand_config).tag()
    UnitTagger(ConfigUtils.load_config('../config.json').proj_sd_config).tag()
    UnitTagger(ConfigUtils.load_config('../config.json').proj_catalase_config).tag()
    UnitTagger(ConfigUtils.load_config('../config.json').proj_gox_config).tag()
    UnitTagger(ConfigUtils.load_config('../config.json').proj_gpx_config).tag()
    UnitTagger(ConfigUtils.load_config('../config.json').proj_oxi_config).tag()
