from torch import optim
import torch

# 训练过程
def training(model, train_loader, criterion, optimizer, device):
    model.train()  
    loss = 0
    iters = 0
    correct_sum = 0 
    for image, input_ids, attention_mask, labels in train_loader:
        iters += 1
        image = image.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)     
        labels = labels.to(device)
        
        output =  model(input_ids, attention_mask,image)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        temp, pred = torch.max(output,1)
        correct_sum += torch.sum(pred == labels)
        
        if (iters % 10 == 0):
            print("迭代次数:",iters,"loss:",loss.item())
    epoch_acc = correct_sum.item() / len(train_loader.dataset)
    return epoch_acc

def predicting(model, test_loader, device):
    model.eval()
    result = []
    for image,input_ids, attention_mask,  labels in test_loader:
        image = image.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids,attention_mask,image)
            temp, preds = torch.max(outputs, 1)
        result.extend(preds.cpu().numpy())
    return result