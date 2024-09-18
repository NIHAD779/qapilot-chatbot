import pandas as pd
import json

df = pd.read_csv('vector-db.csv')


payloads = []

for index, row in df.iterrows():
    payload = {
        "matching_query": row['Matching Query'],
        "intent": row['Intent']
    }
    payloads.append(payload)

with open('payloads.json', 'w') as json_file:
    json.dump(payloads, json_file, indent=4)

print("JSON file has been created successfully.")