import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *

import sklearn
from sklearn.decomposition import PCA

from datetime import datetime
from dateutil.parser import parse

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def convert_time_to_time_of_day(date:datetime):
  if 0 <= date.hour  < 6:
    return "Midnight to Morning"
  elif 6 <= date.hour  < 12:
    return "Morning to Noon"
  elif 12 <= date.hour  < 18:
    return "Noon to Afternoon"
  elif 18 <= date.hour  < 24:
    return "Afternoon to Midnight"

def convert_textual_series_to_indexes(series:pd.Series)->Tuple[pd.Series,dict]:
  items_list = list(series.unique())
  items_index = {crime:index for index,crime in enumerate(items_list)}
  new_series = series.apply(lambda x:items_index[x])
  return new_series, items_list

def update_df_textual_series_to_indexes(df:pd.DataFrame,col_name:str)->dict:
  df[col_name], items_list =  convert_textual_series_to_indexes(df[col_name])
  return items_list

### Merging Similar Classes
def merge_classes(x):
    if x in ['OTHER NARCOTIC VIOLATION','NARCOTICS']:
        return 'NARCOTICS'
    elif x in ['PROSTITUTION','CRIM SEXUAL ASSAULT','SEX OFFENSE']:
        return 'SEX OFFENSE'
    elif x in ['RITUALISM','LIQUOR LAW VIOLATION','GAMBLING']:
        return 'GAMBLING'
    elif x in ['CRIMINAL TRESPASS','ROBBERY']:
        return 'ROBBERY or TRESPASS'
    elif x in ['INTERFERENCE WITH PUBLIC OFFICER','PUBLIC PEACE VIOLATION']:
        return 'PUBLIC PEACE VIOLATION'
    return x


def bert_encode(text):
    if text in bert_cache:
        return bert_cache[text]
    # Get BERT embeddings
    inputs = tokenizer.encode(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        # Tokenize the text and convert to tensor
        outputs = model(inputs)
    # Retrieve the embeddings (taking the first token's embeddings)
    embeddings = outputs[0][0, 0, :].numpy()
    bert_cache[text] = embeddings
    
    return embeddings

def text_to_2d(text,pca):
    return pca.transform(bert_encode(text).reshape(1, -1))[0]


class CSVDataset(Dataset):
    def __init__(self, data:pd.DataFrame, target_col:str, feature_cols:str, transform=None):
        #remove rows with missing values
        self.data = data.dropna().reset_index()
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = self.data.iloc[idx][self.target_col]
        features = self.data.iloc[idx][self.feature_cols]        
        return int(features), int(target)
    
class EmbedingTransformer(nn.Module):
    def __init__(self, input_size:int, embed_size:int, output_size):
        super().__init__()
        self.embeding = nn.Embedding(int(input_size), embed_size)
        self.linear = nn.Linear(embed_size,output_size )
    def forward(self, x):
        return self.linear(self.embeding(x))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def get_embedder(data, embedding_column:str, epochs:int=3):  
    
    dataset = CSVDataset(data, "Primary Type", embedding_column)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    model = EmbedingTransformer(max(data_for_classifier[embedding_column].unique())+1000, 2,len(data_for_classifier["Primary Type"].unique())+1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            features, target = batch
            optimizer.zero_grad()
            loss = loss_fn(model(features.to(device)),target.to(device))
            loss.backward()
            optimizer.step()
    return model.embeding

def embed_features(x, embedder):
    return embedder(torch.tensor(int(x),device =device)).detach().cpu().numpy()



if __name__ == "__main__":
    #load data
    ## set seeds    
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    target_loc = "crime_data.csv"
    crime_data = pd.read_csv(target_loc)
    unrealted_featurse = ['ID','Case Number','IUCR', 'Arrest','Longitude','Domestic','Beat','FBI Code','Updated On','Latitude','Historical Wards 2003-2015', 'Boundaries - ZIP Codes','Location','Wards 2023-']
    related_feature_before_the_dispatch = set(crime_data.columns) - set(unrealted_featurse)
    crime_data = crime_data[list(related_feature_before_the_dispatch)]
    useless_types = ['NON-CRIMINAL (SUBJECT SPECIFIED)','NON-CRIMINAL','NON - CRIMINAL','CONCEALED CARRY LICENSE VIOLATION','DOMESTIC VIOLENCE','PUBLIC INDECENCY','OBSCENITY','RITUALISM']
    crime_data = crime_data[~crime_data['Primary Type'].isin(useless_types)]
    crime_data['Primary Type'] = crime_data['Primary Type'].map(merge_classes)
    crime_data = crime_data.dropna()

    crime_data["Date"] = pd.to_datetime(crime_data["Date"],format='%m/%d/%Y %I:%M:%S %p')
    crime_data["Weekday"] = crime_data["Date"].map(lambda x:x.weekday())
    crime_data["Time of Day"] = crime_data["Date"].map(convert_time_to_time_of_day)
    crime_data["Month"] = crime_data["Date"].map(lambda x:x.month)
    crime_data = crime_data.drop("Date",axis=1)

    primary_crime_type_to_index = update_df_textual_series_to_indexes(crime_data,"Primary Type")
    time_of_day_to_index = update_df_textual_series_to_indexes(crime_data,"Time of Day")
    time_of_day_to_index = update_df_textual_series_to_indexes(crime_data,"Block")

    data_for_classifier = crime_data



    categorical_features = ["Block","Police Districts","District","Zip Codes",'Census Tracts', 'Wards','Ward', 'Community Areas']
    for categorical_feature in tqdm(categorical_features):
        embedder = get_embedder(data_for_classifier[data_for_classifier["split"] == 0], categorical_feature, epochs=3)
        data_for_classifier[categorical_feature] = data_for_classifier[categorical_feature].apply(lambda x:embed_features(x, embedder))
        data_for_classifier[f"{categorical_feature}_1"] = data_for_classifier[categorical_feature].apply(lambda x: x[0])
        data_for_classifier[f"{categorical_feature}_2"] = data_for_classifier[categorical_feature].apply(lambda x: x[1])
        data_for_classifier = data_for_classifier.drop(categorical_feature, axis=1)

    bert_cache = {}
    # Apply the encoding function to the DataFrame column
    for text_columns in tqdm(['Description','Location Description']):
        data_for_classifier[text_columns].apply(bert_encode)
        pca = PCA(n_components=4)
        pca.fit(np.array(list(bert_cache.values())))
        data_for_classifier[text_columns] = data_for_classifier[text_columns].apply(lambda x:text_to_2d(x,pca))
        for i in range(4):
            data_for_classifier[f"{text_columns}_{i}"] = data_for_classifier[text_columns].apply(lambda x:x[i])

        data_for_classifier = data_for_classifier.drop(text_columns, axis=1)
        bert_cache = {}
    
    
    data_for_classifier.to_csv("data_for_classifier.csv",index=False)





    
