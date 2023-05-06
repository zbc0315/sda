#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/10/10 16:59
# @Author  : zhangbc0315@outlook.com
# @File    : org_mol_draw.py
# @Software: PyCharm

import os

import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from cuspy import ConfigUtils
from utils.draw_utils import DrawUtils


class OrgMolDraw:

    def __init__(self, config):
        self._mols_fp = config.filter_mols_fp
        self._org_mols_fp = config.org_mols_fp
        self._org_mols_img_fp = config.org_mols_img_fp
        self._min_distance = config.min_distance
        self._fig_dp = config.fig_dp
        if not os.path.exists(self._fig_dp):
            os.mkdir(self._fig_dp)

    @classmethod
    def _find_name_and_sent(cls, smiles: str, mols_df: pd.DataFrame):
        for _, row in mols_df.iterrows():
            if row.smiles == smiles:
                return row['name'], row['sentences']
        return None, None

    @classmethod
    def _get_sent(cls, smiles: str, mols_df: pd.DataFrame) -> str:
        try:
            query_df = mols_df.query(f"smiles=='{smiles}'")
            if len(query_df) == 0:
                return ''
            query_df = query_df.reset_index()
            name = query_df['name'][0]
            sent = query_df['sentences'][0]
        except:
            name, sent = cls._find_name_and_sent(smiles, mols_df)
            if name is None:
                return ''
        start = sent.find(name)
        end = start + len(name)
        res = sent[:start] + ' 【 ' + name + ' 】 ' + sent[end:]
        return res

    @classmethod
    def _is_right_smiles(cls, smiles: str) -> bool:
        mol = AllChem.MolFromSmiles(smiles)
        return mol is not None

    @classmethod
    def _standard_smiles(cls, smiles: str):
        mol = AllChem.MolFromSmiles(smiles)
        return AllChem.MolToSmiles(mol)

    @classmethod
    def _clean_error_mols(cls, mols_df: pd.DataFrame):
        res_df = pd.DataFrame()
        for _, row in mols_df.iterrows():
            if row['name'] in ['exciton']:
                continue
            else:
                res_df = res_df.append(row, ignore_index=True)
        return res_df

    def process(self):
        mols_df = pd.read_csv(self._mols_fp, sep='\t', encoding='utf-8')
        mols_df = mols_df.loc[mols_df.smiles.map(self._is_right_smiles)]
        mols_df = mols_df.query(f"distance<={self._min_distance}")
        mols_df.smiles = mols_df.smiles.map(self._standard_smiles)
        smiles_count = mols_df.smiles.value_counts()
        mol_img_df = pd.DataFrame()
        mol_img_df['smiles'] = smiles_count.index
        mol_img_df['count'] = smiles_count.values
        mol_img_df['name'] = mol_img_df.smiles.map(lambda x: self._find_name_and_sent(x, mols_df)[0])
        mol_img_df['sentence'] = mol_img_df.smiles.map(lambda x: self._get_sent(x, mols_df))
        mol_img_df.to_csv(self._org_mols_fp, sep='\t', encoding='utf-8', index=False)
        PandasTools.AddMoleculeColumnToFrame(mol_img_df, 'smiles', 'mol')
        # mol_img_df = mol_img_df.drop(['smiles'], axis=1)
        PandasTools.SaveXlsxFromFrame(mol_img_df, self._org_mols_img_fp, molCol='mol')
        error_names = ['ozone', 'hydrogen', 'ethyl acetate', 'diethyl ether', 'oleylamine', 'methylammonium', 'formamidine', 'Formamidinium', 'carbon dioxide']
        # error_names = ['exciton', 'toluene', 'DMF', 'ethanol', 'hydrogen', 'hexane', 'dimethyl sulfoxide',
        #                'formamidinium', 'methanol', 'acetate', 'hexane', 'isopropanol', 'chloroform', 'chlorobenzene']
        # right_names = ['dimethylformamide', 'dimethyl sulfoxide', 'ethanol', 'chlorobenzene', 'isopropanol',
        #                'toluene', '']
        right_names = []
        name_to_simple = {'dimethylformamide': 'DMF'}
        names = list(mol_img_df['name'][:30])
        names = [name_to_simple[name] if name in name_to_simple.keys() else name for name in names]
        counts = list(mol_img_df['count'][:30])
        colors = []
        patterns = []
        c = '#4B778D'
        # c = '#8FD9A8'
        for name in names:
            if name in error_names:
                # colors.append('#F92F60')
                # colors.append('#4B778D')
                # colors.append('#BCD969')
                colors.append('white')
                patterns.append('-')
            else:
                colors.append(c)
                patterns.append(None)
        # DrawUtils.draw_hbar(names, counts, colors, self._fig_dp, fig_name='mol_count', ylabel='Ligand',
        #                     edgecolor=c, )
        DrawUtils.draw_hbar_intext(names, counts, colors, self._fig_dp, fig_name='mol_count', ylabel='Solvent',
                                   edgecolor=c, )


if __name__ == "__main__":
    omd = OrgMolDraw(ConfigUtils.load_config('../_config.json').proj_config)
    omd.process()
