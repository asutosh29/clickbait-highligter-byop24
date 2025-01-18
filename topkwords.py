# !pip install transformers -q

from transformers import AutoTokenizer, TFRobertaModel,pipeline, AutoModel
import matplotlib.pyplot as plt
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# PLEASE COMMENT IF AFFTER DOWNLOADING
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')



'''Trying out different models'''
# Pretrained on Clickbait corupus
# tokenizer = AutoTokenizer.from_pretrained("caush/Clickbait1",is_split_into_words=True)
# model = AutoModel.from_pretrained("caush/Clickbait1",output_hidden_states=True,output_attentions=True)

# Pretrained on Clickbait corupus
tokenizer = AutoTokenizer.from_pretrained("valurank/distilroberta-clickbait",is_split_into_words=True,add_prefix_space=True)
model = AutoModel.from_pretrained("valurank/distilroberta-clickbait",output_hidden_states=True,output_attentions=True)

# Not pretrained on clickbait corpus Roberta model
# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base",is_split_into_words=True,add_prefix_space=True)
# model = AutoModel.from_pretrained("FacebookAI/roberta-base",output_hidden_states=True,output_attentions=True)

model.eval()

def tokenize_and_remove_stopwords(text):
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

  return filtered_tokens

def word_space_tokenizer(text):
    return text.split()
  
def word_to_vectors(text):
  
  return 

def predict(text,top_k = None,plot_graph=False):
  # Playing with different tokenizers
  tokens = tokenize_and_remove_stopwords(text)

  # Hugging face tokenizer
  inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

  # inputs = tokenizer("You Won't Believe what happend next!", return_tensors="pt")
  outputs = model(**inputs)
  tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
  
  attentions = outputs.attentions
  # print(attentions)
  layer_x = -2  # 2nd last layer
  # Attention from layer pervious to next
  attention_layer_x = attentions[layer_x]

  # CLS token attention from layer X to X+1
  cls_attention = attention_layer_x[0, :, 0, :]
  avg_cls_attention_to_tokens = cls_attention.mean(dim=0)

  tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

  sorted_indices = torch.argsort(avg_cls_attention_to_tokens, descending=True)
  
  if top_k == None:
    top_k = len(sorted_indices)
    k=top_k
  else:
    k=top_k
    
  top_K_words_list = [(tokens[i], avg_cls_attention_to_tokens[i].item()) for i in sorted_indices[:top_k]]


  # avg attention plot
  if plot_graph:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(tokens)), avg_cls_attention_to_tokens.detach().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.xlabel("Tokens")
    plt.ylabel("Attention Weight")
    plt.title(f"CLS Token Attention (Layer {layer_x} to Layer {layer_x+1})")
    plt.tight_layout()
    plt.show()
  
  return top_K_words_list

# last_hidden_states = outputs.last_hidden_state

text = "How to choose the best college for you"
text = "An Open Letter to Jerry Seinfeld from a 'Politically Correct' College Student"
text = "Kids runs away from house to become the greatest Gamer of all time!"
text = "100 Fast ways to make quick quick"
text = "German killed by elephant"
text = "Wow, there are things that you can never put in the freezer!"
text = "How to increase your profit using the same content marketing strategy as Spotify?"
text = "President announce free PS4 for all kids"

print(predict(text))

print(text)

# print(len(outputs.hidden_states))
# print(len(outputs.hidden_states[-1].shape))
# print(outputs.hidden_states[-1].shape)
# print(outputs.hidden_states[-2])

"""Using the pretrained model on the Webis Clickbait data set"""

# pipe = pipeline("text-classification", model="caush/Clickbait1")




