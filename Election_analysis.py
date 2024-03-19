# -*- coding: utf-8 -*-


#Required installations
!pip install wandb -qU
!pip install transformers

"""# Data Preprocessing"""

#Imports
import os
import email
import numpy as np
import pandas as pd
from nltk.tokenize.regexp import RegexpTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

df_modi = pd.read_csv('/content/ModiRelatedTweetsWithSentiment.csv', encoding='unicode_escape')
df_modi

df_modi = df_modi.dropna(subset = ['Emotion'])
df_modi

df_modi['Emotion'].unique()

df_modi['Emotion'] = df_modi['Emotion'].replace({'neg': 1, 'pos': 0})

df_modi

df_rahul = pd.read_csv('/content/RahulRelatedTweetsWithSentiment.csv', encoding='unicode_escape')
df_rahul

#!pip install wandb -qU
#!pip install transformers
'''
import wandb
#wandb.login()
wandb.init(
    # set the wandb project where this run will be logged
    project="Membership Inference attacks Via Machine Unlearning",
    name = f"Bert-Unlearning-20%-depressrion",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-5,
    "architecture": "Bert",
    "epochs": 20,
    }
)
'''
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

train_data, test_data = tts(df_modi, test_size = 0.2)




def tokenize_data(data, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []

    for text in data["Tweet"]:
        encoded_data = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_data["input_ids"])
        attention_masks.append(encoded_data["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(data["Emotion"].values)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
        dataset, sampler=RandomSampler(dataset), batch_size=16
    )

    return dataloader

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 2)

'''
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("publichealthsurveillance/PHS-BERT")
model = AutoModel.from_pretrained("publichealthsurveillance/PHS-BERT")
'''
train_dataloader = tokenize_data(train_data, tokenizer) #Full Training Set
test_dataloader = tokenize_data(test_data, tokenizer) # Full testing Set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
num_epochs = 20  # You can adjust the number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    #wandb.log({"Avg train Loss": avg_train_loss})

    print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss}")

# Evaluation on test data
model.eval()
test_labels = []
predicted_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}

        outputs = model(**inputs)
        logits = outputs.logits
        predicted_batch = np.argmax(logits.detach().cpu().numpy(), axis=1)
        predicted_labels.extend(predicted_batch)
        test_labels.extend(batch[2].cpu().numpy())

# Specify a path
#PATH = "Bert-Depr-20%.pt"

# Save
#torch.save(model, PATH)

# Load
#model = torch.load(PATH)
#model.eval()

print("Classification Report:")
print(classification_report(test_labels, predicted_labels))



#constructing the attack accuracy

#wandb.log({"Classification Report": classification_report(test_labels, predicted_labels)})

#wandb.finish()