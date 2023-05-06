#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 10:52
# @Author  : zhangbc0315@outlook.com
# @File    : nlp_tagger.py
# @Software: PyCharm
import os.path

from tqdm import tqdm
import pandas as pd
from cuspy import ConfigUtils
from fastode import FastLog
from chem_nlpy import ChemicalTagger

from utils.tsv_utils import TSVUtils


class NLPTagger:

    def __init__(self, config):
        self._text_fp = config.texts_fp
        self._text_tagged_fp = config.texts_tagged_fp
        self._logger = FastLog(config.log_fp, 'INFO')
        self._tagged_log_fp = config.tagged_log_fp

    def _add_tagged_log(self, fn):
        if os.path.exists(self._tagged_log_fp):
            mode = 'a'
        else:
            mode = 'w'
        with open(self._tagged_log_fp, mode, encoding='utf-8')as f:
            f.write(str(fn))
            f.write('\n')

    def _load_tagged_fns(self):
        if not os.path.exists(self._tagged_log_fp):
            return set([])
        with open(self._tagged_log_fp, 'r', encoding='utf-8')as f:
            fns = set(f.read().split('\n'))
        return fns

    def get_max_tid(self):
        if os.path.exists(self._text_tagged_fp):
            _df = pd.read_csv(self._text_tagged_fp, sep='\t', encoding='utf-8')
            max_tid = _df.tid.max()
        else:
            max_tid = -1
        return max_tid

    def _load_parsed_tids(self):
        if not os.path.exists(self._text_tagged_fp):
            return set([])
        df = pd.read_csv(self._text_tagged_fp, sep='\t', encoding='utf-8')
        return set(df['tid'].values)

    def process(self):
        parsed_tids = self._load_parsed_tids()
        text_df = pd.read_csv(self._text_fp, sep='\t', encoding='utf-8')
        text_tagged_df = pd.DataFrame(columns=['pid', 'tid', 'text_type', 'year', 'xml'])
        # tagged_pids = self._load_tagged_fns()
        with tqdm(total=len(text_df))as pbar:
            for idx, row in text_df.iterrows():
                pbar.update(1)
                if row.pid in parsed_tids:
                    print(f"tagged: {row.pid}")
                    continue
                year = row['year']
                # self._add_tagged_log(row.pid)
                # if row.tid <= max_tid:
                #     continue
                xml_str = ChemicalTagger.tag_text(row.text, host="http://localhost:8088/nlpj")
                # xml_str = ChemicalTagger.tag_text(row.text, host="http://114.214.205.122:8088/nlpj")
                xml_str = xml_str.replace('\n', '')
                text_tagged_df = text_tagged_df.append({'pid': row.pid,
                                                        'tid': row.tid,
                                                        'text_type': row.text_type,
                                                        'year': year,
                                                        'xml': xml_str}, ignore_index=True)
                if len(text_tagged_df) >= 2000:
                    TSVUtils.df_to_tsv(text_tagged_df, self._text_tagged_fp, mode='a')
                    text_tagged_df = pd.DataFrame(columns=['pid', 'tid', 'text_type', 'year', 'xml'])
        if len(text_tagged_df) > 0:
            TSVUtils.df_to_tsv(text_tagged_df, self._text_tagged_fp, mode='a')
        # text_tagged_df.to_csv(self._text_tagged_fp, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    nt = NLPTagger(ConfigUtils.load_config('./config.json').proj_config)
    nt.process()
