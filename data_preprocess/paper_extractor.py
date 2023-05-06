#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/14 10:47
# @Author  : zhangbc0315@outlook.com
# @File    : paper_extractor.py
# @Software: PyCharm

from typing import Iterator
import os
import logging

from tqdm import tqdm
import pandas as pd
from cuspy import ConfigUtils

from utils.tsv_utils import TSVUtils


class PaperExtractor:

    def __init__(self, config):
        self._origin_papers_dp = config.origin_papers_dp
        self._papers_fp = config.papers_fp

    def _iter_origin_paper_df(self) -> Iterator[pd.DataFrame]:
        for fn in os.listdir(self._origin_papers_dp):
            fp = os.path.join(self._origin_papers_dp, fn)
            yield pd.read_csv(fp, sep='\t', encoding='utf-8')

    @classmethod
    def _add_years(cls, wos_fn: str, papers_df: pd.DataFrame):
        if '-' not in wos_fn:
            return papers_df
        year_str = wos_fn.split('-')[0]
        if len(year_str) != 4:
            return papers_df
        year = int(year_str)
        if year < 1000 or year > 2023:
            return papers_df
        papers_df['year'] = [year] * len(papers_df)

    @classmethod
    def _get_year(cls, wos_fn: str):
        default_year = 999
        if '-' not in wos_fn:
            return default_year
        year_str = wos_fn.split('-')[0]
        if len(year_str) != 4:
            return default_year
        year = int(year_str)
        if year < 1000 or year > 2023:
            return default_year
        return year

    @classmethod
    def wos_to_df(cls, wos_fp: str):
        paper_df = pd.DataFrame(columns=['pid', 'source_fp', 'source_id', 'doi', 'title', 'abstract'])
        try:
            wos_df = pd.read_csv(wos_fp, sep='\t', encoding='utf-8')
        except Exception as e:
            return None
        for idx, row in wos_df.iterrows():
            paper_df = paper_df.append({'pid': idx,
                                        'source_fp': wos_fp,
                                        'source_id': idx,
                                        'doi': row.DI,
                                        'title': row.TI,
                                        'abstract': row.AB}, ignore_index=True)
        return paper_df

    def process(self):
        paper_df = pd.DataFrame(columns=['pid', 'source_fp', 'source_id', 'doi', 'title', 'abstract', 'year'])
        pid = 0
        tot = len(list(os.listdir(self._origin_papers_dp)))
        if os.path.exists(self._papers_fp):
            os.remove(self._papers_fp)
        with tqdm(total=tot)as pbar:
            for source_fn in os.listdir(self._origin_papers_dp):
                pbar.update(1)
                source_fp = os.path.join(self._origin_papers_dp, source_fn)
                year = self._get_year(source_fn)
                try:
                    source_df = pd.read_csv(source_fp, sep='\t', encoding='utf-8')
                except Exception as e:
                    continue
                for idx, row in source_df.iterrows():
                    paper_df = paper_df.append({'pid': pid,
                                                'source_fp': source_fp,
                                                'source_id': idx,
                                                'doi': row.DI,
                                                'title': row.TI,
                                                'abstract': row.AB,
                                                'year': year}, ignore_index=True)
                    pid += 1
                    if len(paper_df) >= 1000:
                        TSVUtils.df_to_tsv(paper_df, self._papers_fp, mode='a')
                        paper_df = pd.DataFrame(columns=['pid', 'source_fp', 'source_id', 'doi', 'title', 'abstract', 'year'])
        if len(paper_df) > 0:
            TSVUtils.df_to_tsv(paper_df, self._papers_fp, mode='a')
        # paper_df.to_csv(self._papers_fp, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    pe = PaperExtractor(ConfigUtils.load_config('./config.json').proj_config)
    pe.process()
