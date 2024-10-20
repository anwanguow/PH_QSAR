#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

def csv_to_txt(csv_file, txt_file):
    with open(csv_file, mode='r', newline='', encoding='utf-8') as infile, open(txt_file, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        for row in reader:
            # 将每一行转换为用空格分隔的字符串，并写入txt文件
            outfile.write(" ".join(row) + '\n')

# 示例用法
csv_file_path = 'reg.csv'  # 替换为你的CSV文件路径
txt_file_path = 'reg.txt'  # 替换为你想要保存的TXT文件路径

csv_to_txt(csv_file_path, txt_file_path)
