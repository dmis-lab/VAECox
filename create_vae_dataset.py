import pandas as pd 
import pickle
import numpy as np
from tqdm import tqdm

DATA_VERSION = '200115'

imputed_list = []
masking_list = []

OUTPUT_PATH = './data/ember_libfm_{}_811_mRNA@.tsv'.format(DATA_VERSION)
BINARY_PATH = './data/ember_libfm_{}_mRNA@_binary.csv'.format(DATA_VERSION)
CANCERS = ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD', 
'UCEC', 'THCA', 'SKCM', 'PRAD', 'PAAD', 'LAML', 'KIRP', 'GBM', 'COAD', 'CESC']

print("Combining all Cancer Datasets...")
for c in tqdm(CANCERS):
	PICKLE_PATH = './data/imputed_and_binary_{}.pickle'.format(c)
	PANDAS_PATH = '/hdd1/juneseok/imbernomics/{}_imber.tsv'.format(c)
	with open(PICKLE_PATH, "rb") as handle:
		package = pickle.load(handle)
	imputed_df = package[0]
	masking_df = package[1]

	# NEW EMBERNOMICS!
	days2death = []
	days2follow = []
	for row in imputed_df.index.values:
		if imputed_df.loc[row, 'censored'] == 0:
			days2death.append(imputed_df.loc[row, 'survival'])
			days2follow.append(0.0)
		elif imputed_df.loc[row, 'censored'] == 1:
			days2follow.append(imputed_df.loc[row, 'survival'])
			days2death.append(0.0)
		else:
			raise Exception("what the hell?")
	imputed_df['Cli@Days2Death'] = pd.Series(np.array(days2death), index=imputed_df.index)
	imputed_df['Cli@Days2FollowUp'] = pd.Series(np.array(days2follow), index=imputed_df.index)
	imputed_df['Cli@Censored'] = imputed_df['censored']
	imputed_df = imputed_df.drop(columns=['censored', 'survival'])
	old_cols = imputed_df.columns.values
	new_cols = []
	for i in old_cols:
		if 'Cli@' not in i:
			new_cols.append('mRNA@' + i)
		else:
			new_cols.append(i)
	imputed_df.set_axis(new_cols, axis='columns', inplace=True)
	imputed_df.to_csv(PANDAS_PATH, sep="\t",  encoding="utf-8", index_label='Samples')

	imputed_list.append(imputed_df)
	masking_list.append(masking_df)

df = pd.concat(masking_list, axis=0, sort=True)
print(df.shape)
df.to_csv(BINARY_PATH, sep=",", index_label='Samples')


print("Building the VAE Dataset...")
df = pd.concat(imputed_list, axis=0, sort=True)
print(df.shape)

try:
    df = df.drop(columns=['Cli@Days2Death', 'Cli@Days2FollowUp', 'Cli@Censored', 'Fold@CV'])
except:
    df = df.drop(columns=['Cli@Days2Death', 'Cli@Days2FollowUp', 'Cli@Censored'])

print(df.columns)
print('Num of Dups', df.duplicated().sum().sum())
df = df.drop_duplicates()
print('Num of Dups', df.duplicated().sum().sum())
indices = df.index.values
print('Count Zeros', df[df == 0.0].sum().sum())
df_train, df_valid, df_test = np.split(df.sample(frac=1.0, random_state=np.random.seed()), [int(.8*len(df)), int(.9*len(df))])

train_samples = set(df_train.index.values)
valid_samples = set(df_valid.index.values)
test_samples = set(df_test.index.values)

tv = train_samples.intersection(valid_samples)
vs = valid_samples.intersection(test_samples)
st = test_samples.intersection(train_samples)
print(len(tv))
print(len(vs))
print(len(st))
assert tv == vs and vs == st and tv == st

df_test['Fold@811'] = pd.Series([2 for _ in df_test.index], index=df_test.index)
df_train['Fold@811'] = pd.Series([0 for _ in df_train.index], index=df_train.index)
df_valid['Fold@811'] = pd.Series([1 for _ in df_valid.index], index=df_valid.index)

print(df_test.shape)
print(df_valid.shape)
print(df_train.shape)

df = pd.concat([df_train, df_valid, df_test])
print(df.shape)
df.to_csv(path_or_buf=OUTPUT_PATH, sep="\t", index_label='Samples', encoding="utf8", na_rep='NA')