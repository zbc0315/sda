#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 16:38
# @Author  : zhangbc0315@outlook.com
# @File    : alloy_draw.py
# @Software: PyCharm
import math
import os.path

import numpy as np
import pandas as pd
from cuspy import ConfigUtils

from utils.chem_utils import ChemUtils
from utils.draw_utils import DrawUtils


class AlloyDraw:

    def __init__(self, config):
        self._name = config.name
        self._els_count_df = pd.read_csv(config.elements_count_fp, sep='\t', encoding='utf-8')
        self._els_link_count_df = pd.read_csv(config.elements_link_count_fp, sep='\t', encoding='utf-8')
        self._papers_fp = config.papers_fp
        self._elements_count_for_years_fp = config.elements_count_for_years_fp
        self._elements_link_count_for_years_fp = config.elements_link_count_for_years_fp
        self._needed_rows = config.needed_rows
        self._fig_dp = config.fig_dp
        if not os.path.exists(self._fig_dp):
            os.mkdir(self._fig_dp)

    def _get_els_and_count(self, max_num: int):
        els = []
        counts = []
        colors = []
        for n, (_, row) in enumerate(self._els_count_df.iterrows()):
            el = row['element']
            count = row['count']
            # a = ChemUtils.get_atomic_num(el)
            if n >= max_num:
                break
            els.append(el)
            counts.append(count)
            colors.append(ChemUtils.get_color_by_symbol(el))
        return els, counts, colors

    def _draw_count_bar(self, max_num: int = 20):
        els, counts, colors = self._get_els_and_count(max_num)
        DrawUtils.draw_hbar(els.copy(), counts.copy(), colors.copy(), self._fig_dp, title=self._name)
        DrawUtils.draw_bar(els.copy(), counts.copy(), colors.copy(), self._fig_dp, title=self._name)
        return els, counts, colors

    def _draw_heat(self, els: [str], counts: [int]):
        # els.reverse()
        data = np.ones((len(els), len(els)))
        num = len(els)
        for _, row in self._els_link_count_df.iterrows():
            if row.k1 not in els or row.k2 not in els:
                continue
            idx1 = els.index(row.k1)
            idx2 = els.index(row.k2)
            data[idx2][num-idx1-1] = math.log10(row['count'])
            data[idx1][num-idx2-1] = math.log10(row['count'])
        # for i, count in enumerate(counts):
        #     data[i][num-i-1] = count
        DrawUtils.draw_heat(data, els, self._fig_dp, title=self._name)

    def _draw_graph(self, els, counts):
        rows = set()
        for el in els:
            rows.add(ChemUtils.get_row_by_symbol(el))
        rows = sorted(list(rows))
        idxes = [rows.index(ChemUtils.get_row_by_symbol(el)) for el in els]
        categories = [{"name": f"{row}th row"} for row in rows]
        DrawUtils.draw_graph(els, counts, self._els_link_count_df, idxes, categories, self._fig_dp)

    def _count_years(self, els_count_year_df: pd.DataFrame, select_year: bool = True):
        if select_year:
            years = [int(col.split('_')[-1]) for col in els_count_year_df.columns if col.startswith('count_before') and 2011<= int(col.split('_')[-1])<=2022]
        else:
            years = [int(col.split('_')[-1]) for col in els_count_year_df.columns if col.startswith('count_before')]
        papers_df = pd.read_csv(self._papers_fp, sep='\t', encoding='utf-8')
        # years = set(papers_df.year)
        years = sorted(years, key=lambda x: x)
        year_counts = {}
        for year in years:
            papers_before_df = papers_df.loc[papers_df.year.map(lambda x: x<=year)]
            year_counts[year] = len(papers_before_df)
        return year_counts

    def _reset_count_per_year_in_df(self, df: pd.DataFrame, prefix: str = "count_before"):
        years = [int(col.split('_')[-1]) for col in df.columns if col.startswith(prefix)]
        years = sorted(years, key=lambda x: x, reverse=True)
        # years = list(reversed(years))
        for n, year in enumerate(years):
            if n == len(years) - 1:
                continue
            col = f"{prefix}_{year}"
            last_col = f"{prefix}_{years[n+1]}"
            df[col] = df[col] - df[last_col]
        df = df.sort_values(by=f"{prefix}_{2022}", ascending=False)
        return df

    @classmethod
    def _count_els_by_els(cls, df: pd.DataFrame, years: [int]):
        els_counts = {}
        # for year in years:
        #     col = f"count_before_{year}"
        #     s = df[col].sum()
        #     if s > 0:
        #         df[col] = df[col] / s
        for _, row in df.iterrows():
            els_counts[row['element']] = []
            for n, year in enumerate(years):
                col = f"count_before_{year}"
                # els_counts[row['element']].append(math.pow(row[col], 1/3))
                # els_counts[row['element']].append(math.log2(row[col]+1))
                els_counts[row['element']].append(row[col])
        return els_counts

    @classmethod
    def _count_els_by_els_for_plot(cls, df: pd.DataFrame, years: [int]):
        els_counts = {}
        for n, year in enumerate(years):
            # ratio = 1 - (len(years) - n - 1) / len(years)
            col = f"count_before_{year}"
            s = df[col].max()
            if s > 0:
                df[col] = df[col] / s
        for _, row in df.iterrows():
            els_counts[row['element']] = []
            for n, year in enumerate(years):
                col = f"count_before_{year}"
                # els_counts[row['element']].append(math.sqrt(row[col]) if row[col] > 0 else 0)
                els_counts[row['element']].append(row[col])
                # els_counts[row['element']].append(math.log2(row[col] + 1))
        return els_counts

    def _draw_river(self):
        els_count_year_df = pd.read_csv(self._elements_count_for_years_fp, sep='\t', encoding='utf-8')
        els_count_year_df = els_count_year_df.loc[:9]
        year_counts = self._count_years(els_count_year_df)
        els_counts = self._count_els_by_els(els_count_year_df, list(year_counts.keys()))
        DrawUtils.draw_slack_river(year_counts, els_counts, self._fig_dp, 'slack_river')
        els_counts = self._count_els_by_els_for_plot(els_count_year_df, list(year_counts.keys()))
        DrawUtils.draw_slack_plot(year_counts, els_counts, self._fig_dp, 'slack_plot')

    def _draw_per_year(self):
        els_count_year_df = pd.read_csv(self._elements_count_for_years_fp, sep='\t', encoding='utf-8')
        els_count_year_df = self._reset_count_per_year_in_df(els_count_year_df)
        els_count_year_df = els_count_year_df.loc[:9]
        year_counts = self._count_years(els_count_year_df)
        els_counts = self._count_els_by_els(els_count_year_df, list(year_counts.keys()))
        DrawUtils.draw_slack_river(year_counts, els_counts, self._fig_dp, 'per_year_river')
        DrawUtils.draw_slack_plot(year_counts, els_counts, self._fig_dp, 'per_year_plot')

    def _draw_3d_bar(self):
        els_count_year_df = pd.read_csv(self._elements_count_for_years_fp, sep='\t', encoding='utf-8')

        els_count_year_df = els_count_year_df.loc[:9]
        year_counts = self._count_years(els_count_year_df)
        years = list(year_counts.keys())
        el_counts = self._count_els_by_els(els_count_year_df, years)
        els = list(el_counts.keys())
        el_to_color = {el: ChemUtils.get_color_by_symbol(el) for el in els}
        DrawUtils.draw_bar3d(years, els, el_counts, el_to_color)

    @classmethod
    def _count_el_for_year(cls, df: pd.DataFrame, years):
        el_counts_for_year = {}
        for year in years:
            el_counts_for_year[year] = []
            col = f"count_before_{year}"
            for _, row in df.iterrows():
                el_counts_for_year[year].append(row[col])
        return el_counts_for_year

    def _draw_stack_bars_by_year(self):
        els_count_year_df = pd.read_csv(self._elements_count_for_years_fp, sep='\t', encoding='utf-8')
        els_count_year_df = els_count_year_df.loc[:19]
        year_counts = self._count_years(els_count_year_df)
        years = list(year_counts.keys())
        el_counts_for_year = self._count_el_for_year(els_count_year_df, years)
        els = list(els_count_year_df['element'])
        colors = [ChemUtils.get_color_by_symbol(el) for el in els]
        DrawUtils.draw_hbar_for_years(year_counts, el_counts_for_year, els, colors, self._fig_dp, 'stack_bars')
        return els, years
        # DrawUtils.draw_hbar_and_plot_for_years(year_counts, el_counts_for_year, els, colors)

    def _draw_bars_per_year(self):
        els_count_year_df = pd.read_csv(self._elements_count_for_years_fp, sep='\t', encoding='utf-8')
        els_count_year_df = self._reset_count_per_year_in_df(els_count_year_df)
        els_count_year_df = els_count_year_df.reset_index()
        els_count_year_df = els_count_year_df.loc[:19]

        year_counts = self._count_years(els_count_year_df, select_year=False)
        years = list(year_counts.keys())
        for n in range(len(year_counts.keys())-1, 0, -1):
            year = years[n]
            last_year = years[n-1]
            year_counts[year] = year_counts[year] - year_counts[last_year]
        new_years_counts = {}
        for year, count in year_counts.items():
            if 2011 <= year <= 2022:
                new_years_counts[year] = count
        years = list(new_years_counts.keys())
        el_counts_for_year = self._count_el_for_year(els_count_year_df, years)
        els = list(els_count_year_df['element'])
        colors = [ChemUtils.get_color_by_symbol(el) for el in els]
        DrawUtils.draw_hbar_for_years(new_years_counts, el_counts_for_year, els, colors, self._fig_dp, 'per_years_bars')
        return els, years

    def _draw_stack_heats_by_year(self, els: [str], years: [int]):
        elements_link_count_for_years_df = pd.read_csv(self._elements_link_count_for_years_fp,
                                                       sep='\t', encoding='utf-8')
        data_list = []
        num = len(els)
        for year in years:
            data = np.zeros((len(els), len(els)))
            col = f"link_count_before_{year}"
            for _, row in elements_link_count_for_years_df.iterrows():
                if row.k1 not in els or row.k2 not in els:
                    continue
                idx1 = els.index(row.k1)
                idx2 = els.index(row.k2)
                # data[idx2][num-idx1-1] = math.log10(row[col]+1)
                # data[idx1][num-idx2-1] = math.log10(row[col]+1)
                data[num - idx1 - 1][idx2] = math.log10(row[col] + 1)
                data[num - idx2 - 1][idx1] = math.log10(row[col] + 1)
            data = data / data.max()
            data_list.append(data)
        DrawUtils.draw_stack_heats_for_years(data_list, list(reversed(els)), years, self._fig_dp, 'stack_heats')

    def _draw_heats_by_year(self, els: [str], years: [int]):
        elements_link_count_for_years_df = pd.read_csv(self._elements_link_count_for_years_fp,
                                                       sep='\t', encoding='utf-8')
        elements_link_count_for_years_df = self._reset_count_per_year_in_df(elements_link_count_for_years_df, prefix="link_count_before")
        elements_link_count_for_years_df = elements_link_count_for_years_df.reset_index()
        data_list = []
        num = len(els)
        for year in years:
            data = np.zeros((len(els), len(els)))
            col = f"link_count_before_{year}"
            for _, row in elements_link_count_for_years_df.iterrows():
                if row.k1 not in els or row.k2 not in els:
                    continue
                idx1 = els.index(row.k1)
                idx2 = els.index(row.k2)
                # data[idx2][num-idx1-1] = math.log10(row[col]+1)
                # data[idx1][num-idx2-1] = math.log10(row[col]+1)
                data[num - idx1 - 1][idx2] = math.log10(row[col] + 1)
                data[num - idx2 - 1][idx1] = math.log10(row[col] + 1)
            data_list.append(data)
        DrawUtils.draw_stack_heats_for_years(data_list, list(reversed(els)), years, self._fig_dp, 'per_year_heats')

    def process(self):
        if os.path.exists(self._elements_count_for_years_fp):
            # self._draw_3d_bar()
            # self._draw_bars_by_year()
            els, years = self._draw_stack_bars_by_year()
            self._draw_stack_heats_by_year(els, years)
            els, years = self._draw_bars_per_year()
            self._draw_heats_by_year(els, years)
            # self._draw_river()
            # self._draw_per_year()
        els, counts, colors = self._draw_count_bar()
        self._draw_heat(els, counts)
        DrawUtils.draw_graph_by_nx(els, counts, colors, self._els_link_count_df, self._fig_dp, title=self._name)


if __name__ == "__main__":
    ad = AlloyDraw(ConfigUtils.load_config('../config.json').proj_oer_years_config)
    # ad = AlloyDraw(ConfigUtils.load_config('../config.json').proj_peroxidase_years_config)
    ad.process()
