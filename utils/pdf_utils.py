#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/6 9:29
# @Author  : zhangbc0315@outlook.com
# @File    : pdf_utils.py
# @Software: PyCharm

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal, LAParams
# from pdfminer.pdfinterp import PDFTextExtractionNotAllowed


class PDFUtils:

    @classmethod
    def clean_text(cls, text: str):
        for c in text:
            if ord(c) >= 127 or ord(c) <= 31:
                text = text.replace(c, ' ')
            # if not c.isascii():
            #     text = text.replace(c, " ")
        return text

    @classmethod
    def parse_paragraph(cls, text: str):
        text = text.replace('ﬂ', 'fi')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2217', '*')
        text = text.replace('\u2019', "'")
        text = text.replace('', 'SOH')
        text = text.replace(' ', ' ')
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.endswith('-'):
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line + ' ')
        text = ''.join(cleaned_lines)
        return cls.clean_text(text)

    @classmethod
    def pdf_to_text(cls, pdf_fp: str):
        paras = []
        pdf = open(pdf_fp, 'rb')
        parser = PDFParser(pdf)
        doc = PDFDocument(parser)
        # parser.set_document(doc)
        # doc.set_parser(parser)
        # doc.initialize()
        if not doc.is_extractable:
            raise ValueError("doc is not extractable")
        else:
            rsrcmagr = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmagr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmagr, device)

            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                layout = device.get_result()
                for x in layout:
                    try:
                        if (isinstance(x, LTTextBoxHorizontal)):
                            paras.append(cls.parse_paragraph(x.get_text()))
                    except Exception as e:
                        print("Failed")
        pdf.close()
        return paras


if __name__ == "__main__":
    pass
