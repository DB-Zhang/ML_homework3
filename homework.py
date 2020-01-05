import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
import csv
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

def getWaterMelonOneHot(pandascsv, rate=0.8, shuffle=True):
    if shuffle:
        tmp_data = pandascsv.sample(frac=1).reset_index(drop=True)
    else:
        tmp_data = pandascsv
    tmp_data = pd.get_dummies(tmp_data)
    tmp_data = tmp_data.drop(['编号','好瓜_否'],axis=1)
    
    shape = tmp_data.shape
    
#     print(int(shape[0]*rate))
    trainset = np.array(tmp_data[:int(shape[0]*rate)],dtype=np.float32)
    testset = np.array(tmp_data[int(shape[0]*rate):],dtype=np.float32)
    
    trainset_x = trainset[:,:-1]
    trainset_y = trainset[:,-1:]
    
    testset_x = testset[:,:-1]
    testset_y = testset[:,-1:]
    
    trainset = (trainset_x,trainset_y)
    testset = (testset_x,testset_y)
    
    return trainset, testset


def getWaterMelonDe(pandascsv, rate=0.8, shuffle=True):
    if shuffle:
        tmp_data = pandascsv.sample(frac=1).reset_index(drop=True)
    else:
        tmp_data = pandascsv
    
    tmp_data = tmp_data.drop(['编号','密度','含糖率'],axis=1)
    
    shape = tmp_data.shape
    
# #     print(int(shape[0]*rate))
#     trainset = np.array(tmp_data[:int(shape[0]*rate)],dtype=np.float32)
#     testset = np.array(tmp_data[int(shape[0]*rate):],dtype=np.float32)
    
#     trainset_x = trainset[:,:-1]
#     trainset_y = trainset[:,-1:]
    
#     testset_x = testset[:,:-1]
#     testset_y = testset[:,-1:]
    
#     trainset = (trainset_x,trainset_y)
#     testset = (testset_x,testset_y)
    trainset = tmp_data[:int(shape[0]*rate)]
    testset = tmp_data[int(shape[0]*rate):]
    
    return trainset,testset

def getMnistData():
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = torchvision.datasets.MNIST('mnist', train=True, transform=x_transforms, download=False)
    test_dataset = torchvision.datasets.MNIST('mnist', train=False, transform=x_transforms, download=False)
    
    return train_dataset,test_dataset   


def compute_squared(X):
    X = X.T
    m,n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n,1))
    return (H + H.T - 2*G).T

def MaxKVector(mat,k):
    e_vals,e_vecs = np.linalg.eig(mat)
    sorted_indices = np.argsort(e_vals)
    return e_vals[sorted_indices[:-k-1:-1]],e_vecs[:,sorted_indices[:-k-1:-1]]

def getAcc(Types):
    acc1 = (sum(Types[0])+sum(Types[1]))/(len(Types[0])+len(Types[1]))
    acc2 = (len(Types[0])-sum(Types[0])+len(Types[1])-sum(Types[1]))/(len(Types[0])+len(Types[1]))
    acc = max(acc1,acc2)
    return acc

def AvgEuclideanDistance(SetA,SetB):
    if len(SetA)==0 or len(SetB)==0:
        return False
    Dist_Sum = 0
    for itemA in SetA:
        for itemB in SetB:
            Dist_Sum += np.sum(np.square(itemA-itemB))
            
    Dist_Sum /= len(SetA) * len(SetB)
    return Dist_Sum




def Mnist_MDS():
    # watermalon = pd.read_csv("watermalon.csv",engine="python")
    # trainset, testset = getWaterMelonOneHot(watermalon)
    # trainset_x, trainset_y = trainset
    # testset_x, testset_y = testset
    # Set,_ = getWaterMelonOneHot(watermalon,rate=1)
    # Data_watermelon, Label_watermelon = Set
    MnistTrain,MnistTest = getMnistData()
    # example = MnistTrain[0][0][0].detach().numpy()
    # example_label = MnistTrain[0][1].item()
    Mnist12_vector = []
    Mnist12_label = []
    Mnist12_img = []
    for item in MnistTrain:
        label = item[1]
        if label >2 or label==0:
            continue
        Mnist12_vector.append(item[0][0].detach().numpy().reshape(-1,))
        Mnist12_img.append(item[0][0].detach().numpy())
        Mnist12_label.append(label)
    for item in MnistTest:
        label = item[1]
        if label >2 or label==0:
            continue
        Mnist12_vector.append(item[0][0].detach().numpy().reshape(-1,))
        Mnist12_img.append(item[0][0].detach().numpy())
        Mnist12_label.append(label)
        
    Mnist12_vector = np.array(Mnist12_vector)
    Mnist12_label = np.array(Mnist12_label)
    Mnist12_img = np.array(Mnist12_img)
    Mnist12_label = Mnist12_label.reshape(-1,1)
    Try = Mnist12_vector[:1000]
    Label = Mnist12_label[:1000]
    MatD_real = compute_squared(Try)
    DistI = np.sum(MatD_real,axis=0)
    DistJ = np.sum(MatD_real,axis=1)
    DistDD = np.sum(MatD_real)
    MatB = np.zeros_like(MatD_real)
    n = MatB.shape[0]
    m = MatB.shape[1]
    for i in range(n):
        for j in range(m):
            MatB[i][j] = (DistI[i] + DistJ[j] - DistDD/n - MatD_real[i][j]*MatD_real[i][j]*n)/(2*n)

    e_vals,e_vecs = np.linalg.eig(MatB)

    need_vals2 , need_vector2 = MaxKVector(MatB,2)
    Dim_to_2 = need_vector2.dot(np.sqrt(np.diag(need_vals2)))
    need_vals3 , need_vector3 = MaxKVector(MatB,3)
    Dim_to_3 = need_vector3.dot(np.sqrt(np.diag(need_vals3)))
    font = {'family': 'MicroSoft Yahei',
        'weight': 'bold',
        'size': 12}
    
    matplotlib.rc("font", **font)


    df = pd.DataFrame(np.hstack((Dim_to_2,Label)),columns = ['x1','x2','label'])

    ax = plt.figure()
    colors = ['b','g']
    Label_Com = [1,2]
    plt.title('MDS_To_dim2')
    for label in range(1,-1,-1):
        x1 = df[df['label']==Label_Com[label]]['x1']
        y1 = df[df['label']==Label_Com[label]]['x2']
        plt.scatter(x1,y1,c=colors[label])

    plt.legend(['1','2'])
    plt.show()

    df3 = pd.DataFrame(np.hstack((Dim_to_3,Label)),columns = ['x1','x2','x3','label'])

    fig = plt.figure(figsize=(16, 12))  #参数为图片大小

    ax = fig.gca(projection='3d')  
    # ax.set_aspect('equal')
    colors = ['b','g']
    Label_Com = [1,2]
    plt.title('MDS_To_dim3')
    for label in range(1,-1,-1):
        x = df3[df3['label']==Label_Com[label]]['x1']
        y = df3[df3['label']==Label_Com[label]]['x2']
        z = df3[df3['label']==Label_Com[label]]['x3']
        c = colors[label]
        ax.scatter(x, y, z, c=c, label=str(label))

    plt.legend(['1','2'])
    plt.show()

def ElurFunc():
    watermalon = pd.read_csv("watermalon.csv",engine="python")
    trainset, testset = getWaterMelonOneHot(watermalon)
    trainset_x, trainset_y = trainset
    testset_x, testset_y = testset
    Set,_ = getWaterMelonOneHot(watermalon,rate=1)
    Data_watermelon, Label_watermelon = Set
    KindList = {}
    numlist ={}  
    num = 0
    for item in Data_watermelon:
        KindList[num] = [item]
        numlist[num] = [num]
        num+= 1

    recorder = []
    while len(KindList)>2:
        mat = np.ones((len(KindList),len(KindList)))
        mat *= 9999999
        for i in range(len(KindList)):
            for j in range(i+1,len(KindList)):
                mat[i,j] = AvgEuclideanDistance(KindList[list(KindList.keys())[i]],KindList[list(KindList.keys())[j]])

        row = np.argmin(mat)// len(KindList)
        colom =  np.argmin(mat)% len(KindList)
        KindList[list(KindList.keys())[row]] += KindList[list(KindList.keys())[colom]]
        numlist[list(KindList.keys())[row]] += numlist[list(KindList.keys())[colom]]
        print(numlist)
        recorder.append(numlist.copy())
        del numlist[list(KindList.keys())[colom]]  
        del KindList[list(KindList.keys())[colom]]  

        
    print(numlist)
    recorder.append(numlist)
        
    Types = []
    for label in list(KindList.keys()):
        Type = KindList[label]
        labels = []
        for item in Type:
            for i in range(len(Data_watermelon)):
                if np.sum(Data_watermelon[i]-item)==0:
                    labels.append(int(Label_watermelon[i][0]))
                    continue
        Types.append(labels)        
        
    acc = getAcc(Types) 

    print("accuracy AvgEuclideanDistance: ",acc)
    print(Types)

def NcutDistance(SetA,SetB,sigma = 3):

    Set_all = np.array(SetA+SetB)
    W_All = np.exp(-compute_squared(Set_all)/2/np.square(sigma))
    times = np.zeros_like(W_All)
    SetA_one = np.zeros((W_All.shape[0],1))
    SetA_one[:len(SetA)]=1

    SetB_one = np.zeros((W_All.shape[0],1))
    SetB_one[-len(SetB):]=1

    Cut = np.sum(W_All*SetA_one.dot(SetB_one.T))

    D_All = np.diag(np.sum(W_All,axis=1))
    VolA = np.sum(D_All.dot(SetA_one))
    VolB = np.sum(D_All.dot(SetB_one))
    L_All = W_All - D_All
    L_All_Norm =  np.sqrt(np.linalg.inv(D_All)).dot(L_All).dot(np.sqrt(np.linalg.inv(D_All)))
    NCut = Cut/VolA + Cut/VolB
#     print(W_All[0])
    return NCut

def NCutFunc():
    KindList = {}
    watermalon = pd.read_csv("watermalon.csv",engine="python")
    trainset, testset = getWaterMelonOneHot(watermalon)
    trainset_x, trainset_y = trainset
    testset_x, testset_y = testset
    Set,_ = getWaterMelonOneHot(watermalon,rate=1)
    Data_watermelon, Label_watermelon = Set
    numlist ={}  
    num = 0
    for item in Data_watermelon:
        KindList[num] = [item]
        numlist[num] = [num]
        num+= 1

    # print(KindList.keys())
    # print(numlist.keys())
    recorder = []
    while len(KindList)>2:
        mat = np.ones((len(KindList),len(KindList)))
        mat *= -9999999
        for i in range(len(KindList)):
            for j in range(i+1,len(KindList)):
                mat[i,j] = NcutDistance(KindList[list(KindList.keys())[i]],KindList[list(KindList.keys())[j]],8)
        
    #     print(mat)

        row = np.argmax(mat)// len(KindList)
        colom =  np.argmax(mat)% len(KindList)
        KindList[list(KindList.keys())[row]] += KindList[list(KindList.keys())[colom]]
        numlist[list(KindList.keys())[row]] += numlist[list(KindList.keys())[colom]]
        print(numlist)
        recorder.append(numlist.copy())
        del numlist[list(KindList.keys())[colom]]  
        del KindList[list(KindList.keys())[colom]]  
    print(numlist)

    recorder.append(numlist)
        
    Types = []
    for label in list(KindList.keys()):
        Type = KindList[label]
        labels = []
        for item in Type:
            for i in range(len(Data_watermelon)):
                if np.sum(Data_watermelon[i]-item)==0:
                    labels.append(int(Label_watermelon[i][0]))
                    continue
        Types.append(labels)        
        
    acc = getAcc(Types) 

    print("accuarcy of NcutDistance: ",acc)
    print(Types)



class InformationTree():
    def __init__(self):
        self.root = 0
        self.flag = False
        self.tree = {}
        
    def train(self,Root_Table):
        self.tree = self.TreeNode(Root_Table)
    
    
    def TreeNode(self,Table_Root):
        classify_gain = []
        for name in list(Table_Root.columns)[:-1]:
            classify = Table_Root.groupby(name).size()
            Good = Table_Root[Table_Root['好瓜']=='是']
            Bad = Table_Root[Table_Root['好瓜']=='否']
            if len(Good)==0 or len(Bad)==0:
                break
            Good_p = len(Good)/len(Table_Root)
            Bad_p = len(Good)/len(Table_Root)
            EntD =  -Good_p*np.log2(Good_p) - Bad_p*np.log2(Bad_p)
            Ent = 0
            IV = 0
            new = 0
            for item in classify.index:
                new = Table_Root[Table_Root[name]==item]
                Good = new[new['好瓜']=='是']
                Bad = new[new['好瓜']=='否']
                Good_p = (len(Good)+1e-3)/(len(new))
                Bad_p = (len(Bad)+1e-3)/(len(new))
                Ent += (-Good_p*np.log2(Good_p) - Bad_p*np.log2(Bad_p))*len(new)/len(Table_Root)
                rate = classify[item]/len(Table_Root)
                IV += -rate*np.log2(rate)
            Gain = EntD - Ent
            Gain_ratio = Gain/(IV+1e-6)
            classify_gain.append(Gain_ratio)
    #         print(name,Gain_ratio)
        if classify_gain:
            if max(classify_gain)>0:
                classify_name = list(Table_Root.columns)[classify_gain.index(max(classify_gain))]
        
        if self.flag==False:
            self.root = classify_name
            self.flag = True
        sonTree = {}
        classify = Table_Root.groupby(classify_name).size()
        print(classify_name)
        for item in classify.index:
            new = Table_Root[Table_Root[classify_name]==item]
    #         print(new)

            Good = new[new['好瓜']=='是']
            Bad = new[new['好瓜']=='否']    
            if len(Bad) == 0:
                print(item,'Good')
                sonTree[item] = 'Good'

                continue
            elif len(Good) == 0:
                sonTree[item] = 'Bad'
                print(item,'Bad')
                continue
            else:
#                 print(item)
                if len(new)>0:
                    sonTree[item] = self.TreeNode(new.drop([classify_name],axis=1))
#         print(sonTree)
        return {classify_name:sonTree}
    
    def show(self):
        pass
    
    def validate(self,testData):
        
        newTable = pd.DataFrame(columns=list(testData.columns)+['预测'])
        for num,row in testData.iterrows():
            tree = self.tree
            predict = 'Good'
            Flag = True
            while Flag: 
                for key in tree.keys():
                    try:
                        diffType = tree[key]
                        Type = row[key]
    #                     print(diffType[Type])
                        if type(diffType[Type]) == type({}):
                            tree = diffType[Type]
                            continue
                        else:
                            predict = diffType[Type]
                            Flag = False
                    except:
                        Flag = False
            if predict=='Bad':
                row['预测'] = '否'
            else:
                row['预测'] = '是'
            newTable.loc[num] = row
        
        a = np.array(newTable['好瓜']==newTable['预测'])
        acc = np.sum(a)/len(a)
        print(acc)
        
        return newTable

        


def TreeFunc():
    np.random.seed(11)
    watermalon = pd.read_csv("watermalon.csv",engine="python")
    Treetrainset, Treetestset = getWaterMelonDe(watermalon,rate=0.7)
    tree = InformationTree()
    tree.train(Treetrainset)
    print(tree.validate(Treetestset))

if __name__ == "__main__":
    Mnist_MDS()
    # ElurFunc()
    # NCutFunc()
    # TreeFunc()