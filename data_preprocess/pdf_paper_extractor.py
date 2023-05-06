#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 11/9/2022 1:53 PM
# @Author  : zhangbc0315@outlook.com
# @File    : pdf_paper_extractor.py
# @Software: PyCharm

import os
import logging

import pandas as pd
from tqdm import tqdm
from cuspy import ConfigUtils

from utils.pdf_utils import PDFUtils


logging.basicConfig(level=logging.ERROR)


class PDFPaperExtractor:

    def __init__(self, config):
        self._origin_pdf_dp = config.origin_papers_dp
        self._papers_fp = config.papers_fp

    @classmethod
    def _extract_pdf(cls, pdf_fp: str) -> [str]:
        try:
            texts = PDFUtils.pdf_to_text(pdf_fp)
        except Exception as e:
            # raise e
            print(e)
            return []
        return texts

    def process(self):
        pdf_fns = list(os.listdir(self._origin_pdf_dp))
        papers_df = pd.DataFrame(columns=['pid', 'source_fp', 'source_id',
                                          'doi', 'title', 'abstract', 'contexts'])
        num_err = 0
        with tqdm(total=len(pdf_fns))as pbar:
            pbar.set_description("Pdf paper extractor")
            for n, pdf_fn in enumerate(pdf_fns):
                pbar.set_postfix_str(f"err: {num_err} ; pdf: {pdf_fn}")
                pbar.update(1)
                pdf_fp = os.path.join(self._origin_pdf_dp, pdf_fn)
                texts = self._extract_pdf(pdf_fp)
                if len(texts) == 0:
                    num_err += 1
                    continue
                papers_df = papers_df.append({'pid': n,
                                              'source_fp': pdf_fp,
                                              'source_id': n,
                                              'doi': None,
                                              'title': None,
                                              'abstract': None,
                                              'contexts': texts}, ignore_index=True)
        papers_df.to_csv(self._papers_fp, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    ppe = PDFPaperExtractor(ConfigUtils.load_config('./config.json').proj_config)
    ppe.process()
