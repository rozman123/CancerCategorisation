import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split


if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')



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
x_train_tensor=x_train_tensor.to(device) # przełancza na gpu tensory z danymi

y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32)
y_train_tensor=y_train_tensor.to(device) # przełancza na gpu tensory z danymi

imput_size=x_train_tensor.shape[1]
#print(Input_layer)

model_NN = nn.Sequential
(
    nn.Linear(imput_size, 128),
    nn.Sigmoid(),
    nn.Linear(128, 128),
    nn.Sigmoid(),
    nn.Linear(128,1),
    nn.Sigmoid()
)

lossf=nn.BCELoss()  # Funkcja Binary Cross Entropy

optimizer=torch.optim.Adam(model_NN.parameters())

model_NN.to(device) # przełancza na gpu model ML

for i in range(0,501):

    output=model_NN(x_train_tensor)

    Loss=lossf(output,y_train_tensor.view(-1,1))


    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()

    # print progres
    if i % 100==0:
        print(f'{i} Skuteczność: {Loss.item():.4f}')



model_NN.eval()  # przełącza model na tryb oceniania
with torch.no_grad():
    # torch.tensor zmienia vektor z pandas na tensor z pytorcha
    x_test_tensor=torch.tensor(x_test.values,dtype=torch.float32).to(device)
    y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32).to(device)

    y_wynik=model_NN(x_test_tensor)
    wyniki=(y_test_tensor>=0.5).squeeze().long()
    skutecznosc=(y_test_tensor==wyniki).float().mean()
    print(f'Skuteczność: {skutecznosc}')


