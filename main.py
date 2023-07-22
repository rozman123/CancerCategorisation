import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split


data=pd.read_csv('cancer.csv') # we are loading the data
#print(data.head())
x=data.drop(['diagnosis(1=m, 0=b)'],axis=1) # data on which we are teaching the model
#print(x.head())
y=data['diagnosis(1=m, 0=b)'] # column with the output (what we are traying to predict cancer malignant or not)
#print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3)

# print(x_train)
# print(x_test)
x_train_tensor=torch.tensor(x_train.values,dtype=torch.float32)

y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32)

Input_layer=x_train_tensor.values
model = nn.Sequential
(
    nn.Linear(Input_layer, 128),
    nn.Sigmoid(),
    nn.Linear(128, 128),
    nn.Sigmoid(),
    nn.Linear(128,1),
    nn.Sigmoid()
)
lossf=nn.BCELoss()  # Funkcja Binary Cross Entropy

optimalizer=torch.optim.Adam(model.parameters())

for i in range(0,501):

    output=model(x_train_tensor)

    Loss=lossf(output,y_train_tensor.view(-1,1))


    optimalizer.zero_grad()
    Loss.backward()
    optimalizer.step()

    # print progres
    if i % 100==0:
        print(f'{i} Skuteczność: {Loss.item():.4f}')



model.eval()
with torch.no_grad():
    x_test_tensor=torch.tensor(x_test.values,dtype=torch.float32)
    y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32)
    

