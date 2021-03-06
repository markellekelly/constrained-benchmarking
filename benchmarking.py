import pandas as pd
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
from torch.utils.data.sampler import SubsetRandomSampler
from collections import Counter
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time
import math

def get_device():

    if torch.cuda.is_available():      
        return torch.device("cuda")

    return torch.device("cpu")

class MyDataset(Dataset):
    def __init__(self, X, Y):
        X = X.copy()
        self.X = np.array(X.values).astype(np.float32)
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x1 = self.X[idx]
        y1 = self.y[idx]
        return x1, y1

class ClassificationNetwork(nn.Module):
    def __init__(self, input_size, n_classes, layers):
        super().__init__()
        self.input_size = input_size
        self.drops = nn.Dropout(0.3)
        self.nlayers = len(layers)
        self.lin1 = nn.Linear(self.input_size, layers[0])
        self.bn2 = nn.BatchNorm1d(layers[0])
        if self.nlayers==1:
            self.lin2 = nn.Linear(layers[0], n_classes)
        elif self.nlayers==2:
            self.lin2 = nn.Linear(layers[0], layers[1])
            self.bn3 = nn.BatchNorm1d(layers[1])
            self.lin3 = nn.Linear(layers[1], n_classes)
        elif self.nlayers==3:
            self.lin2 = nn.Linear(layers[0], layers[1])
            self.bn3 = nn.BatchNorm1d(layers[1])
            self.lin3 = nn.Linear(layers[1], layers[2])
            self.bn4 = nn.BatchNorm1d(layers[2])
            self.lin4 = nn.Linear(layers[2], n_classes)
        else:
            print("number of layers not supported.")
        
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        if self.nlayers==1:
            return self.lin2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        if self.nlayers==2:
            return self.lin3(x)
        x = F.relu(self.lin3(x))
        x = self.drops(x)
        x = self.bn4(x)
        return self.lin4(x)


class RegressionNetwork(nn.Module):
    def __init__(self, input_size, layers):
        super().__init__()
        self.input_size = input_size
        self.nlayers = len(layers)
        self.lin1 = nn.Linear(self.input_size, layers[0])
        if self.nlayers==1:
            self.lin2 = nn.Linear(layers[0], 1)
        elif self.nlayers==2:
            self.lin2 = nn.Linear(layers[0], layers[1])
            self.lin3 = nn.Linear(layers[1], 1)
        elif self.nlayers==3:
            self.lin2 = nn.Linear(layers[0], layers[1])
            self.lin3 = nn.Linear(layers[1], layers[2])
            self.lin4 = nn.Linear(layers[2], 1)
        
    def forward(self, x):
        x = F.relu(self.lin1(x))
        if self.nlayers==1:
            return self.lin2(x)
        x = F.relu(self.lin2(x))
        if self.nlayers==2:
            return self.lin3(x)
        x = F.relu(self.lin3(x))
        return self.lin4(x)

def get_optimizer(model, opt, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = opt(parameters, lr=lr, weight_decay=wd)
    return optim

def train_model(model, c, optim, train_dl, device):
    model.train()
    total = 0
    sum_loss = 0
    for x, y in train_dl:
        batch = y.to(device).shape[0]
        output = model(x.to(device))
        if c:
            loss = F.cross_entropy(output.float(), y.to(device).long())
        else:
            y = y.float().reshape((y.shape[0], 1))
            loss = F.mse_loss(output, y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
    return sum_loss/total

def val_classification_loss(model, valid_dl, device):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y in valid_dl:
        current_batch_size = y.to(device).shape[0]
        out = model(x.to(device))
        loss = F.cross_entropy(out.float(), y.to(device).long())
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        correct += (pred == y.to(device)).float().sum().item()
    return sum_loss/total

def val_regression_loss(model, valid_dl, device):
    model.eval()
    total = 0
    mse = 0
    sum_loss=0
    for x, y in valid_dl:
        current_batch_size = y.to(device).shape[0]
        out = model(x.to(device))
        y = y.to(device).float().reshape((y.shape[0], 1))
        loss = F.mse_loss(out.float(), y)
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        mse += (y - out).float().sum().item() ** 2
    return sum_loss/total

def classification_train_loop(model, optim, epochs, train_dl, valid_dl, device):
    if epochs <1:
        return -1*math.inf
    losses = []
    model=model.cuda()
    for i in range(epochs): 
        loss = train_model(model, True, optim, train_dl, device)
        val_loss = val_classification_loss(model, valid_dl, device)
        losses.append(val_loss)
        if len(losses)>30 and losses[-30] <= val_loss: #early stopping
            return val_loss
    return val_loss
    
def regression_train_loop(model, optim, epochs, train_dl, valid_dl, device):
    if epochs<1:
        return math.inf
    losses = []
    model=model.cuda()
    for i in range(epochs): 
        loss = train_model(model, False, optim, train_dl, device)
        val_loss = val_regression_loss(model, valid_dl, device)
        losses.append(val_loss)
        if len(losses)>30 and losses[-30] <= val_loss: #early stopping
            return val_loss
    return val_loss

class Evaluator:
    def __init__(self, attr, device, layers, lrs, opts):
        self.attr = attr
        self.device = device
        self.layers = layers
        self.lrs = lrs
        self.opts = opts

    def get_data(self, row):
        fname = row['train_file']
        tfname = row['test_file']
        # get delimiter, whether there is a header, and NA symbol
        delim = "," if pd.isnull(row['delimiter']) else row['delimiter']
        header = 0 if row['header'] == 1 else None
        na = []
        if row['is_missing_values'] == 1:
            na.append(row['missing_values'])
        #read in train file, and test if applicable
        df = pd.read_csv("data-files/" + fname, delimiter=delim, header=header, na_values=na)
        if pd.isnull(tfname):
            return df, False, None
        df_t = pd.read_csv("data-files/" + tfname, delimiter=delim, header=header, na_values=na)
        return df, True, df_t

    def prep_data(self, row):
        train, test_file, test = self.get_data(row)
        attr = self.attr[self.attr['dataset_id'] == row['dataset_id']].reset_index(drop=True)
        # iterate through the column roles, recording the target and columns to drop
        targ_col = None
        drop_cols = []
        for col, role in zip(train.columns, list(attr['role'])):
            if role == "Target":
                targ_col = col
            elif role != "Feature":
                drop_cols.append(col)
        # combine train and test if necessary for get_dummies, etc.
        break_off = None
        if test_file:
            break_off = len(train)
            full = pd.concat([train,test])
        else:
            full = train
        full = full.dropna(subset=[targ_col])
        full[targ_col] = full[targ_col].astype('category')
        full[targ_col] = full[targ_col].cat.codes
        X = full.drop(targ_col, axis=1)
        X = X.drop(drop_cols, axis=1)
        y = full[targ_col]
        try:
            X = pd.get_dummies(X, drop_first=True)
        except:
            pass
        X = X.fillna(X.mean())
        # split back into train and test
        if break_off:
            # follow the original split if there was one
            X_tv, y_tv = X.iloc[:break_off], y.iloc[:break_off]
            X_test, y_test = X.iloc[break_off:], y.iloc[break_off:]
        else:
            X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.25)
        X_train, X_val, y_train, y_val = train_test_split(X_tv,y_tv, test_size=0.1, random_state=0)
        for dset in [X_train, X_test, X_val, y_val, y_train, y_test]:
            dset.reset_index(drop=True, inplace=True)
        return (X_train, X_val, X_test, y_train, y_val, y_test)

    def choose_n_models(self, input_size, n_classes, c, n):
        models = {}
        for i in range(n):
            layer = self.layers[random.randint(0, len(self.layers)-1)]
            lr = self.lrs[random.randint(0, len(self.lrs)-1)]
            opt = self.opts[random.randint(0, len(self.opts)-1)]
            if c:
                mod = ClassificationNetwork(input_size,n_classes,layer)
            else:
                mod = RegressionNetwork(input_size,layer)
            optim = get_optimizer(mod, opt, lr = lr, wd = 0.0000)
            models[(mod, optim)] = None
        return models

    def enumerate_models(self,input_size, n_classes, c):
        comp =0; highest_complexity = None
        models = {}
        for layer in self.layers:
            for lr in self.lrs:
                for opt in self.opts:
                    if c:
                        mod = ClassificationNetwork(input_size,n_classes,layer)
                    else:
                        mod = RegressionNetwork(input_size,layer)
                    optim = get_optimizer(mod, opt, lr = lr, wd = 0.0000)
                    models[(mod, optim)] = None
                    if len(layer) * max(layer) > comp:
                        comp = len(layer) * max(layer)
                        highest_complexity = (mod, optim)
        return models, highest_complexity

    def choose_budget(self, train_dl, valid_dl, hc, c):
        t1 = time.time()
        if c:
            classification_train_loop(hc[0],hc[1], 1, train_dl, valid_dl, self.device)
        else:
            regression_train_loop(hc[0],hc[1], 1, train_dl, valid_dl, self.device)
        t2 = time.time()
        return (30*60)/(t2-t1)

    def successive_halving(self, budget, models, train_dl, valid_dl, c):
        '''jamieson et al. (2015)'''
        t1 = time.time()
        n = len(models.keys())
        while len(models.keys()) > 1:
            e = math.floor(budget/(len(models.keys())*math.ceil(math.log(n,2))))
            for model in models.keys():
                if c:
                    acc = -1*classification_train_loop(model[0],model[1], e, train_dl, valid_dl, self.device)
                else:
                    acc = -1* regression_train_loop(model[0],model[1], e, train_dl, valid_dl, self.device)
                models[model] = acc
            for i in range(int(len(models.keys())/2)):
                models.pop(min(models, key=models.get))
        t2 = time.time()
        f = open("output.txt", "a");f.write("elapsed time:"+str(t2-t1)+'\n'); f.close()
        return models

    def hyperband(self, R, train_dl, valid_dl,input_size, n_classes, c, nu=3):
        '''li et al. (2016)'''
        t1 =time.time()
        smax = math.floor(math.log(R,nu))
        B = R * (smax+1)
        best_performance=-1*math.inf; best_model=None
        for s in range(smax,-1,-1):
            n = math.ceil((B*(nu**s))/(R*(s+1)))
            r = R*(n**(-1*s))
            models = self.choose_n_models(input_size, n_classes, c, n)
            for i in range(0,s+1):
                ni = math.floor(n*(nu**(-1*i)))
                ri = int(r*(nu**i))
            if ri > 0:
                for model in models.keys():
                    if c:
                        acc = classification_train_loop(model[0],model[1], ri, train_dl, valid_dl, self.device)
                    else:
                        acc = -1* regression_train_loop(model[0],model[1], ri, train_dl, valid_dl, self.device)
                    models[model] = acc
                remove = len(models) - math.floor(ni/nu)
                for i in range(remove):
                    models.pop(min(models, key=models.get))
        t2 = time.time()
        f = open("output.txt", "a");f.write("elapsed time:"+str(t2-t1)+'\n'); f.close()
        return models

    def run_model(self, row, HB=False, budget=None):

        c = row['task'] == "Classification"

        X_train, X_val, X_test, y_train, y_val, y_test = self.prep_data(row)
        input_size = X_train.shape[1]
        n_classes = len(y_train.unique())
        train_ds = MyDataset(X_train, y_train)
        valid_ds = MyDataset(X_val, y_val)
        batch_size = 16
        train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)
        test_ds = MyDataset(X_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        models,hc=self.enumerate_models(input_size, n_classes, c)
        if budget is None:
            budget = self.choose_budget(train_dl, valid_dl, hc, c)

        if HB:
            model = self.hyperband(budget, train_dl, valid_dl, input_size, n_classes, c)
        else:
            model = self.successive_halving(budget, models, train_dl, valid_dl, c)
        k = [i for i in model.keys()][0]
        model = k[0]
        f = open("output.txt", "a");f.write(str(model)+'\n');f.write(str(k[1])+'\n'); f.close()

        model.eval()
        model = model.cuda()
        total=0; correct=0; mse=0
        with torch.no_grad():
            for x,y in test_dl:
                out = model(x.to(self.device))
                total+=y.to(self.device).shape[0]
                if c:
                    prob = F.softmax(out, dim=1)
                    pred = torch.max(out, 1)[1]
                    correct += (pred == y.to(self.device)).float().sum().item()
                else:
                    y = y.to(self.device).float().reshape((y.shape[0], 1))
                    mse += (y-out).float().sum().item() ** 2

        if c:
            acc = correct/total
            f = open("output.txt", "a");f.write("accuracy:"+str(acc)+'\n\n'); f.close()
        else:
            mse=mse/total
            f = open("output.txt", "a");f.write("MSE:"+str(mse)+'\n\n'); f.close()


def main():
    f = open("output.txt", "w"); f.write("");f.close()
    
    device = get_device()
    s=0
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

    datasets = pd.read_csv('schema_conversion/datasets_update.csv',
                      usecols=["name","old_id", "new_id", "task", "is_tabular"])
    attr = pd.read_csv("schema_conversion/Attributes.csv")
    tab = pd.read_csv("schema_conversion/DatasetTabular.csv")
    sub = pd.DataFrame(datasets[['name','is_tabular','task','old_id']])
    sub = sub[sub["is_tabular"]==1]
    df = sub.merge(tab, how='inner',left_on='old_id',right_on='dataset_id').drop('old_id', axis=1)

    layers = [[50],[100],[100,50], [200,100,50]]
    lrs = [0.0001, 0.0005]
    opts = [torch_optim.AdamW, torch_optim.SGD]
    ev = Evaluator(attr, device, layers, lrs, opts)
   
    for i in range(10,len(df)):
        f = open("output.txt", "a");f.write(df.iloc[i]['name']+'\n'); f.close()
        ev.run_model(df.iloc[i], HB=True)

if __name__=="__main__":
    main()
