# convert /home/rnorouzini/JointLearning/datasets/pseudo_annotate_data/llama3_8b_processed.jsonl to a csv file
import pandas as pd

df_llama3 = pd.read_json('/home/rnorouzini/JointLearning/datasets/pseudo_annotate_data/llama3_8b_processed.jsonl', lines=True)
# save the dataframe to a csv file
df_llama3.to_csv('/home/rnorouzini/JointLearning/datasets/pseudo_annotate_data/llama3_8b_processed.csv', index=False)
# read the csv file and print the first 5 rows