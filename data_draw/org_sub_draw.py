#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 10:44
# @Author  : zhangbc0315@outlook.com
# @File    : org_sub_draw.py
# @Software: PyCharm

import os.path
import math

import numpy as np
import pandas as pd
from cuspy import ConfigUtils

from utils.draw_utils import DrawUtils
from utils.chem_utils import ChemUtils


class OrgSubDraw:

    def __init__(self, config):
        self._atom_count_df = pd.read_csv(config.atom_count_fp, sep='\t', encoding='utf-8')
        self._bond_count_df = pd.read_csv(config.bond_count_fp, sep='\t', encoding='utf-8')
        self._frag_count_df = pd.read_csv(config.frag_count_fp, sep='\t', encoding='utf-8')
        self._group_count_df = pd.read_csv(config.group_count_fp, sep='\t', encoding='utf-8')

        self._atom_link_count_df = pd.read_csv(config.atom_link_count_fp, sep='\t', encoding='utf-8')
        self._bond_link_count_df = pd.read_csv(config.bond_link_count_fp, sep='\t', encoding='utf-8')
        self._group_link_count_df = pd.read_csv(config.group_link_count_fp, sep='\t', encoding='utf-8')
        # self._frag_link_count_df = pd.read_csv(config.frag_link_count_fp, sep='\t', encoding='utf-8')

        self._fig_dp = config.fig_dp
        if not os.path.exists(self._fig_dp):
            os.mkdir(self._fig_dp)

    @classmethod
    def _get_key_and_count(cls, data_df, col_name):
        keys = []
        counts = []
        for _, row in data_df.iterrows():
            key = row[col_name]
            count = row['count']
            if count < 50:
                continue
            keys.append(key)
            counts.append(count)
        return keys, counts

    def _draw_heat(self, els: [str], counts: [int], df, name):
        data = np.ones((len(els), len(els)))
        num = len(els)
        for _, row in df.iterrows():
            if row.k1 not in els or row.k2 not in els:
                continue
            idx1 = els.index(row.k1)
            idx2 = els.index(row.k2)
            data[idx2][num-idx1-1] = math.log10(row['count'])
            data[idx1][num-idx2-1] = math.log10(row['count'])
        DrawUtils.draw_heat(data, els, self._fig_dp, fig_name=name)

    @classmethod
    def _get_bond_cate(cls, bond_symbols: [str]):
        types = ['-', '=', '#', ':', 'o']
        # types = [':', '#', '=', '-', 'o']
        idxes = []
        for bond_symbol in bond_symbols:
            for idx in range(len(types)-1, 0, -1):
                t = types[idx]
            # for idx, t in enumerate(types):
                if t in bond_symbol:
                    idxes.append(idx)
                    break
            idxes.append(4)
        cates = [{"name": t} for t in types]
        return idxes, cates

    @classmethod
    def _get_atom_cate(cls, atom_symbols: [str]):
        rows = set()
        for el in atom_symbols:
            row = ChemUtils.get_row_by_symbol(el.title())
            if el.islower():
                rows.add(-row)
            else:
                rows.add(row)
        rows = sorted(list(rows))
        idxes = []
        for el in atom_symbols:
            row = ChemUtils.get_row_by_symbol(el.title())
            if el.islower():
                row = -row
            idxes.append(rows.index(row))
        # idxes = [rows.index(ChemUtils.get_row_by_symbol(el.title())) for el in atom_symbols]
        categories = [{"name": f"{row}th row"} for row in rows]
        return idxes, categories

    def _draw_graph(self, data_link_df, els, counts, col_name):
        idxes = []
        cates = []
        lwr = 1
        if col_name == 'bond':
            idxes, cates = self._get_bond_cate(els)
            lwr = 0.5
        elif col_name == 'atom':
            idxes, cates = self._get_atom_cate(els)
            lwr = 2
        elif col_name == 'group':
            idxes, cates = self._get_bond_cate(els)
            lwr = 2
        DrawUtils.draw_graph(els, counts, data_link_df, idxes, cates, self._fig_dp, fig_name=col_name,
                             line_width_ratio=lwr)
        # else:
        #     DrawUtils.draw_graph_without_cate(els, counts, data_link_df, self._fig_dp, fig_name=col_name)

    def _draw(self, data_df, link_data_df, col_name):
        keys, counts = self._get_key_and_count(data_df, col_name)
        DrawUtils.draw_hbar(keys, counts, self._fig_dp, fig_name=col_name, ylabel=col_name)
        self._draw_heat(keys, counts, link_data_df, col_name)
        self._draw_graph(link_data_df, keys, counts, col_name)

    def process(self):
        self._draw(self._atom_count_df, self._atom_link_count_df, 'atom')
        self._draw(self._bond_count_df, self._bond_link_count_df, 'bond')
        self._draw(self._group_count_df, self._group_link_count_df, 'group')
        # self._draw(self._frag_count_df, 'frag')


if __name__ == "__main__":
    osd = OrgSubDraw(ConfigUtils.load_config('../_config.json').proj_config)
    osd.process()
