import pandas as pd

JUDGED  = "data/judged_600.jsonl"
SAMPLE  = "data/judge_sample_600.csv"

judged = pd.read_json(JUDGED, lines=True)
sample = pd.read_csv(SAMPLE)

# print("judged shape:", judged.shape)
# print("sample shape:", sample.shape)
# print(judged.head())
# print(sample.head())

Columns = ['name_a', 'name_b']
joined = pd.merge(judged, sample, left_on=Columns, right_on=Columns)
# print(joined.columns.tolist())
# print("Joined shape:", joined.shape)

value_counts = joined['label'].value_counts()
print("Value counts:", value_counts)