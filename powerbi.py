from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb+srv://admin:admin@cluster0.ybfhrsg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["fraud_detection"]
collection = db["predictions"]

# Fetch all documents
cursor = collection.find()

# Convert to DataFrame
df = pd.DataFrame(list(cursor))

# Optional: flatten nested dictionary if needed
if 'input' in df.columns:
    input_df = pd.json_normalize(df['input'])
    df = pd.concat([df.drop(columns=['input']), input_df], axis=1)
    
# Optional: drop MongoDB ID
df.drop(columns=["_id"], inplace=True)

df
