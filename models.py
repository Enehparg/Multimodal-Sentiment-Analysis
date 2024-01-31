from transformers import AutoTokenizer, BertModel, BertTokenizer,AutoFeatureExtractor ,ResNetModel
from torch import nn
from torchvision.models import resnet50
from torch import optim
import torch

class resnet50_image(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = resnet50(pretrained = True)
        self.fc1 = nn.Linear(1000,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,3)
        self.acti = nn.ReLU(inplace = True)
        
    def forward(self, input_ids, attention_mask, image):
        x = self.image_model(image)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = self.acti(x)
        x = self.fc3(x)
        return x

class bert_text(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,3)
        self.acti = nn.ReLU(inplace = True)
    
    def forward(self, input_ids, attention_mask, image):
        x = self.text_model(input_ids = input_ids, attention_mask = attention_mask)
        x = x.last_hidden_state[:,0,:]
        x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = self.acti(x)
        x = self.fc3(x)
        
        return x

class Weight_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = resnet50(pretrained= True)
        
        self.fc_text = nn.Linear(768,256)
        self.fc_image = nn.Linear(1000,256)
        self.fc_mid = nn.Linear(256,128)
        self.weight = nn.Linear(128,1)
        self.output = nn.Linear(128,3)
        self.acti = nn.ReLU(inplace = True)
    
    def forward(self,input_ids, attention_mask, image):
        x = self.text_model(input_ids = input_ids, attention_mask = attention_mask)
        x = x.last_hidden_state[:,0,:]
        x.view(x.shape[0],-1)
        x_text = self.fc_text(x)
        x_text = self.acti(x_text)
        x_text = self.fc_mid(x_text)
        x_text = self.acti(x_text)
        weight_t = self.weight(x_text)
        
        x_image = self.image_model(image)
        x_image = self.fc_image(x_image)
        x_image = self.acti(x_image)
        x_image = self.fc_mid(x_image)
        x_image = self.acti(x_image)
        weight_i = self.weight(x_image)
        
        x_out = x_image * weight_i + x_text * weight_t
        x_out = self.output(x_out)
        return x_out

class Concat_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = resnet50(pretrained= True)
        
        self.fc1 = nn.Linear(1768,1024)
        self.do = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,128)
        self.output = nn.Linear(128,3)
        self.acti = nn.ReLU(inplace = True)
    
    def forward(self,input_ids,attention_mask, image):
        x = self.text_model(input_ids = input_ids,attention_mask = attention_mask)
        x = x.last_hidden_state[:,0,:]
        x.view(x.shape[0],-1)
        
        x_image = self.image_model(image)
        
        x_concat = torch.cat((x,x_image), dim = -1)
        x_concat = self.fc1(x_concat)
        x_concat = self.acti(x_concat)
        x_concat = self.do(x_concat)
        x_concat = self.fc2(x_concat)
        x_concat = self.acti(x_concat)
        x_concat = self.fc3(x_concat)
        x_concat = self.acti(x_concat)
        x_concat = self.output(x_concat)
        
        return x_concat