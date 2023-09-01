import re
import pandas as pd

def text_preprocess(text: str):
    text = text.lower()
    text = text.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    url_pattern = re.compile(r"http[s]?://\S+")
    text = url_pattern.sub(r"<URL>", text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    return text


soft_df = pd.read_csv('raw_datasets/train.csv')
soft_df = soft_df.drop(['post_id'], axis=1)
soft_df = soft_df[['issue','post_text','Inappropriateness','fold0.0']]
soft_df = soft_df.rename(columns={'post_text':'txt','Inappropriateness':'style','fold0.0':'split'})
soft_df['style'] = soft_df['style'].apply(lambda x: 'inapp' if x > 0.5 else 'app')
soft_df['split'] = soft_df['split'].replace(['TEST', 'TRAIN', 'VALID'], ['test', 'train', 'val'])
soft_df = soft_df[['issue','txt', 'style', 'split']]


df = pd.read_csv('raw_datasets/appropriateness_corpus_conservative_w_folds.csv')
selected_df = df[['issue','post_text','Inappropriateness','fold0.0']]
selected_df = selected_df.rename(columns={'post_text': 'txt', 'Inappropriateness': 'style', 'fold0.0': 'split'})
selected_df['style'] = selected_df['style'].replace([1, 0], ['inapp', 'app'])
selected_df['split'] = selected_df['split'].replace(['TEST', 'TRAIN', 'VALID'], ['test', 'train', 'val'])
selected_df = selected_df[['issue','txt','split', 'style']]

merged_df = pd.concat([soft_df, selected_df], ignore_index=True)


# number of duplicates
print('Number of duplicates: ', merged_df.duplicated().sum())
print('Number of duplicates in text column: ', merged_df.duplicated(subset=['txt']).sum())
print('Number of duplicates in both text and issue columns: ', merged_df.duplicated(subset=['txt', 'issue']).sum())

# drop duplicates
merged_df = merged_df.drop_duplicates()
merged_df = merged_df.drop_duplicates(subset=['txt'])
merged_df = merged_df.drop_duplicates(subset=['txt', 'issue'])

# number of duplicates
print('Number of duplicates: ', merged_df.duplicated().sum())
print('Number of duplicates in text column: ', merged_df.duplicated(subset=['txt']).sum())
print('Number of duplicates in both text and issue columns: ', merged_df.duplicated(subset=['txt', 'issue']).sum())

# number of classes per split in merged_df
print('Number of classes per split in merged_df:', merged_df.groupby('split')['style'].value_counts())

# randomly drop 20000 rows where style is 1 and split is TRAIN 
merged_df = merged_df.drop(merged_df[(merged_df['style'] == 'inapp') & (merged_df['split'] == 'train')].sample(n=23000, random_state=1).index)

# number of classes per split in merged_df
print('Number of classes per split in merged_df:', merged_df.groupby('split')['style'].value_counts())

merged_df['txt'] = merged_df['txt'].apply(text_preprocess)

merged_df.to_csv('data/Inappropriateness.tsv', sep='\t', index=False, header=True)