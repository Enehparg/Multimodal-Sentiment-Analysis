import pandas as pd
import numpy as np
import torch
from PIL import Image

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel, BertTokenizer,AutoFeatureExtractor ,ResNetModel
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import resnet50
from torch import optim
from torch.utils.data import Dataset
from models import Weight_model, resnet50_image, bert_text, Concat_model
from train import training,predicting
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fusion", type= str, default = "text_and_image")
parser.add_argument("--model", type = str, default = "weight")
parser.add_argument("--epoch",type = int, default = 6)
args = parser.parse_args()

fusion_control = args.fusion
model_control = args.model
epoches = args.epoch

#读取图片和文本信息
def read_txt(path):
    with open(path, 'r', encoding='gb18030') as file:
        text = file.readline()
        file.close()
        return text

def read_jpg(path):
    image = Image.open(path)
    return image

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pretrained_model = BertModel.from_pretrained("bert-base-uncased")

train_source = pd.read_csv('train.txt')
test_source = pd.read_csv("test_without_label.txt")
classes_dict={"positive":0, "neutral":1, "negative": 2}
train_re = train_source.replace({'tag': classes_dict})
labels = list(train_re['tag'])#即y，预测变量

train_texts =[]
train_images = []
test_texts = []
test_images = []

#由于图像的大小不一，这里要统一格式后张量化
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
])

#数据集读入
for guid,tag in train_source.values:
    temp_text = read_txt(f'./data/{int(guid)}.txt')
    temp_image = read_jpg(f'./data/{int(guid)}.jpg')
    train_texts.append(temp_text)
    train_images.append(temp_image)

for guid,tag in test_source.values:
    temp_text = read_txt(f'./data/{int(guid)}.txt')
    temp_image = read_jpg(f'./data/{int(guid)}.jpg')
    test_texts.append(temp_text)
    test_images.append(temp_image)

#建立dataset，以构建dataloader迭代
class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, image, tokenized_texts, labels,transform):
        self.image = image     
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        labels = torch.tensor(self.labels[index])
        image = self.transform(self.image[index])
        
        return image ,input_ids, attention_mask, labels
    
    
#划分训练集
image_train, image_val, texts_train, texts_val, labels_train, labels_val = train_test_split(
    train_images, train_texts, labels, test_size=0.125, random_state=6657)

#tokenized
tokenized_texts_train = [tokenizer(text,padding='max_length',max_length=150,truncation=True,return_tensors="pt") for text in texts_train]
tokenized_texts_val = [tokenizer(text,padding='max_length',max_length=150,truncation=True,return_tensors="pt") for text in texts_val]

#dataset实例化和dataloader构建
train_dataset = CreateDataset(image_train, tokenized_texts_train, labels_train,transform)
val_dataset = CreateDataset(image_val,tokenized_texts_val, labels_val,transform)

train_loader = DataLoader(train_dataset, batch_size = 48, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 48, shuffle = False)
    
#不同模型，通过argparse选用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if (fusion_control == "text_and_image") and (model_control == "weight"):
    print('采用加权的融合模型')
    model = Weight_model()
    model = model.to(device)
elif (fusion_control == "text_and_image") and (model_control == "concat"):
    print('采用向量拼接的融合模型')
    model = Concat_model()
    model = model.to(device)
elif (fusion_control == "image"):
    print('图像消融模型，注意此时的model参数不起效')
    model = resnet50_image()
    model = model.to(device)
elif (fusion_control == "text"):
    print('文本消融模型，注意此时的model参数不起效')
    model = bert_text()
    model = model.to(device)
else:
    print("参数无效")
    exit(0)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-5, weight_decay=2e-6)

best = 0

#训练过程
for epoch in range(epoches):
    train_acc = training(model,train_loader,loss_function, optimizer,device)
    val_result = predicting(model, val_loader, device)
    
    val_labels = np.array(labels_val)
    val_acc = (val_result == val_labels).sum() / len(val_labels)
    
    if (val_acc > best):
        best_acc = val_acc
        torch.save(model, "best_model.pt")
    print("训练进度:",epoch+1,"/6","训练集准确率:",train_acc,"验证集准确率:",val_acc)
    
#测试集的读入以及标签的生成
test_input = pd.read_csv("test_without_label.txt")
labels_test = np.array(test_input['tag'])
tokenized_test = [tokenizer(text,padding='max_length',max_length=150,truncation=True,return_tensors="pt") for text in test_texts]

test_dataset = CreateDataset(test_images, tokenized_test, labels_test,transform)
test_loader = DataLoader(test_dataset, batch_size = 48, shuffle= False)

best_model = torch.load('best_model.pt').to(device)
test_result = np.array(predicting(best_model, test_loader, device))
test_input['tag'] = test_result

classes_trans={0:"positive", 1:"neutral", 2:"negative"}

test_output = test_input.replace({"tag": classes_trans})
test_output.to_csv("predict1.txt",sep=",",index= False)

print("END")