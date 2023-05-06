#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/22 10:23
# @Author  : zhangbc0315@outlook.com
# @File    : tag_token_extractor.py
# @Software: PyCharm
import os.path

from tqdm import tqdm
import pandas as pd
from cuspy import ConfigUtils
from fastode import FastLog, FastXML

from utils.tsv_utils import TSVUtils


class TagTokenExtractor:

    def __init__(self, config):
        self._logger = FastLog(config.log_fp, 'INFO').logger
        self._texts_tagged_fp = config.texts_tagged_fp
        self._tag_token_pairs_fp = config.tag_token_pairs_fp

    def process(self):
        texts_tagged_df = pd.read_csv(self._texts_tagged_fp, sep='\t', encoding='utf-8')
        tag_token_pairs_df = pd.DataFrame(columns=['pid', 'tid', 'text_type', 'year', 'tag_token_pairs', 'attr'])
        if os.path.exists(self._tag_token_pairs_fp):
            os.remove(self._tag_token_pairs_fp)
        with tqdm(total=len(texts_tagged_df))as pbar:
            for _, row in texts_tagged_df.iterrows():
                pbar.update(1)
                year = row['year']
                try:
                    xml = FastXML.parse_string(row.xml)
                except Exception as e:
                    self._logger.warning(f"xml is wrong: \n"
                                         f"pid:{row.pid}, tid:{row.tid},\n"
                                         f"xml:{row.xml}")
                    continue
                tag_token_pairs, attrs = FastXML.get_token_tag_pairs_and_attrs(xml, ['OSCARCM'], ['smiles'])
                tag_token_pairs_df = tag_token_pairs_df.append({'pid': row.pid,
                                                                'tid': row.tid,
                                                                'text_type': row.text_type,
                                                                'year': year,
                                                                'tag_token_pairs': tag_token_pairs,
                                                                'attr': attrs}, ignore_index=True)
                if len(tag_token_pairs_df) >= 2000:
                    TSVUtils.df_to_tsv(tag_token_pairs_df, self._tag_token_pairs_fp, mode='a')
                    tag_token_pairs_df = pd.DataFrame(columns=['pid', 'tid', 'text_type', 'year', 'tag_token_pairs', 'attr'])
        # tag_token_pairs_df.to_csv(self._tag_token_pairs_fp, sep='\t', encoding='utf-8', index=False)
        if len(tag_token_pairs_df) > 0:
            TSVUtils.df_to_tsv(tag_token_pairs_df, self._tag_token_pairs_fp, mode='a')


if __name__ == "__main__":
    tte = TagTokenExtractor(ConfigUtils.load_config('./config.json').proj_config)
    tte.process()
