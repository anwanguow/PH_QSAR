import pandas as pd

list_cc = pd.read_csv('list_cc', delim_whitespace=True)
code_df = pd.read_excel('code.xlsx')
compound_to_smile = dict(zip(code_df['Compound'], code_df['Smiles']))
list_cc['SMILE'] = list_cc['Name'].map(compound_to_smile)
list_cc.to_csv('list_new', sep=' ', index=False)

