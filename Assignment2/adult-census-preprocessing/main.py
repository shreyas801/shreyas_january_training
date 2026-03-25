from preprocessing import *
from categorical_encoding import *
from scaling import *

df = load_data("data/adult.csv")

df = handle_missing_values(df)
df = fix_data_types(df)
df = remove_duplicates(df)
df = treat_outliers(df)
df = drop_irrelevant_features(df)

df = label_encoding(df, 'sex')
df = one_hot_encoding(df, 'workclass')
df = frequency_encoding(df, 'occupation')

# convert target to numeric (IMPORTANT)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

# target encoding
df = target_encoding(df, 'education', 'income')

df = z_score_scaling(df, 'age')

print(df.head())
