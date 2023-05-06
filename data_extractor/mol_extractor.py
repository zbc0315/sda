#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 15:35
# @Author  : zhangbc0315@outlook.com
# @File    : mol_extractor.py
# @Software: PyCharm
import typing

import pandas as pd
from tqdm import tqdm
from cuspy import ConfigUtils


class MolExtractor:

    def __init__(self, config):
        self._tag_token_pairs_fp = config.tag_token_pairs_fp
        self._mols_fp = config.mols_fp
        self._unique_mols_fp = config.unique_mols_fp
        self._paper_and_mol_fp = config.paper_and_mol_fp

    @classmethod
    def _get_mol_name_and_smiles(cls, tag_token_pairs: [(str, str)], smileses: [str]):
        for tag_token_pair, smiles in zip(tag_token_pairs, smileses):
            if tag_token_pair[0] == 'OSCARCM':
                smiles = smiles if len(smiles) > 0 else None
                yield tag_token_pair[1], smiles

    @classmethod
    def _query(cls, df: pd.DataFrame, query_col: str, target_col: str, query_value: str):
        if any([c in query_value for c in ['\\', "'"]]):
            for _, row in df.iterrows():
                if row[query_col] == query_value:
                    return row[target_col]
        else:
            query_df = df.query(f"{query_col}=='{query_value}'")
            if len(query_df) > 0:
                query_df = query_df.reset_index()
                return query_df[target_col][0]
        return None

    @classmethod
    def add_to_unique_mol_df(cls, unique_mol_df: pd.DataFrame, name: str, smiles: str):
        if smiles is not None:
            umid = cls._query(unique_mol_df, 'smiles', 'umid', smiles)
            if umid is not None:
                return unique_mol_df, umid
            # query_df = unique_mol_df.query(f"smiles=='{smiles}'")
            # if len(query_df) > 0:
            #     query_df = query_df.reset_index()
            #     return unique_mol_df, query_df.umid[0]
        umid = cls._query(unique_mol_df, 'name', 'umid', name)
        if umid is not None:
            return unique_mol_df, umid
        # query_df = unique_mol_df.query(f"name=='{name}'")
        # if len(query_df) > 0:
        #     query_df = query_df.reset_index()
        #     return unique_mol_df, query_df.umid[0]

        umid = len(unique_mol_df)
        unique_mol_df = unique_mol_df.append({'umid': umid, 'name': name, 'smiles': smiles}, ignore_index=True)
        return unique_mol_df, umid

    def process(self, need_smiles: bool):
        mols_df = pd.DataFrame(columns=['mid', 'umid', 'name', 'smiles'])
        unique_mols_df = pd.DataFrame(columns=['umid', 'name', 'smiles'])
        paper_and_mol_df = pd.DataFrame(columns=['pid', 'tid', 'name', 'umid'])
        tag_token_pairs_df = pd.read_csv(self._tag_token_pairs_fp, sep='\t', encoding='utf-8')

        mid = 0
        with tqdm(total=len(tag_token_pairs_df))as pbar:
            for _, row in tag_token_pairs_df.iterrows():
                pbar.update(1)
                for name, smiles in self._get_mol_name_and_smiles(eval(row.tag_token_pairs), eval(row.attr)['smiles']):
                    if need_smiles and smiles is None:
                        continue
                    unique_mols_df, umid = self.add_to_unique_mol_df(unique_mols_df, name, smiles)
                    mols_df = mols_df.append({'mid': mid, 'umid': umid, 'name': name, 'smiles': smiles}, ignore_index=True)
                    mid += 1
                    paper_and_mol_df = paper_and_mol_df.append({'pid': row.pid, 'tid': row.tid,
                                                                'umid': umid, 'name': name}, ignore_index=True)
        unique_mols_df.to_csv(self._unique_mols_fp, sep='\t', encoding='utf-8', index=False)
        mols_df.to_csv(self._mols_fp, sep='\t', encoding='utf-8', index=False)
        paper_and_mol_df.to_csv(self._paper_and_mol_fp, sep='\t', encoding='utf-8', index=False)


if __name__ == "__main__":
    me = MolExtractor(ConfigUtils.load_config('./_config.json').proj_config)
    me.process(True)
