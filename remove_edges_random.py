from libraries import *
from geo import Geo
from load_functions import *
from eval_functions import *


dataset = 'geotext'
path = "C:\\Users\\61484\\Graph_Convolutional_Networks\\data\\geo"
dataset = Geo(path, dataset, transform=None)
data = dataset[0]

A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, vocab = get_geo_data(dataset.raw_dir, 'dump.pkl')

U = U_train + U_dev + U_test
locs = np.array([userLocation[u] for u in U])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = Sequential(Linear(dataset.num_features, 300))
        self.conv1 = GCNConv(300, dataset.num_classes)
        #self.conv2 = GCNConv(300, 300)
        #self.lin2 = Sequential(Linear(300, dataset.num_features))

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        #x = self.conv2(x, edge_index)
        #x = self.lin2(x)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
x, edge_index = data.x, data.edge_index

model_path = osp.join(dataset.raw_dir, 'model.pth')
print(f"model path:{model_path}")
if osp.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

#returns the non_zero edges based on the explained priority:
def priority_edges(edge_index,user):
    aa = np.where(np.array(edge_index[:, edge_mask.argsort()[:]][0])==user)
    bb = edge_index[:, edge_mask.argsort()[:]][:,aa]
    
    return bb[:,0]

#index of the same element:
#cat = explained edge_indexes
def self_connection_index(cat,aa):
    bb = cat[:, edge_mask.argsort()[:]]
    xx = bb[:,aa[0]]
    vv = np.array(xx)
    for i in range(len(vv[0])):
        if vv[0][i]==vv[1][i]:
            return i
#deleting self indexes from the indexes to be removed:
def delete_self(aa,index):
    aa3 = np.delete(aa,index,1)
    return aa3

#sorting the other connection index, based on the intial indexes(in the future,it would to easier to remove both edges at the same time)
def sort2edge(cat,aa3,aa4):
    bb = cat[:, edge_mask.argsort()[:]]
    ee = bb[:,aa3[0]]
    ff = bb[:,aa4[0]]
    ls = []
    for i in ff[0]:
        cnt = 0
        for j in ee[1]:
            if i==j:
                ls.append(cnt)
            cnt +=1
    r = np.arange(len(aa4[0]))
    np.put(r,ls,list(aa4[0]))
    return r

#revering indexes:
def revere_indexes(r):
    return r[::-1]

def find_indx(ls1,ls2):
    ls= []
    for i in ls1:
        cnt=0
        for j in ls2:
            if i==j:
                ls.append(cnt)
            cnt+=1 
    return ls


explainer = GNNExplainer(model, epochs=200)

#random removal of edges:
test_index = np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test)).tolist()
hav_distance = [] #distance between the true and predcited labels for each user
latlon_tr = [] #true latitutde and longitude of the users
latlon_pre = []# predicted latitude and longitude of the users
accuracy = []
num_us = []
user_id = []
user_add = 0

rand_edges = np.arange(211451)

for user in tqdm(test_index[0:10]):
    #explaining the node
    node_feat_mask, edge_mask = explainer.explain_node(user, x, edge_index)

    #priotizig the edges based on explaination
    prio_edge = priority_edges(edge_index,user)    

    aa1 = np.where(np.array(edge_index[:, edge_mask.argsort()[:]][0])==user)
    aa2 = np.where(np.array(edge_index[:, edge_mask.argsort()[:]][1])==user)
    
    for num_users in [perc(prio_edge[0],0),perc(prio_edge[0],0.05)-1,perc(prio_edge[0],0.10)-1,perc(prio_edge[0],0.20)-1,perc(prio_edge[0],0.40)-1,perc(prio_edge[0],0.60)-1,perc(prio_edge[0],0.80)-1,perc(prio_edge[0],1)-1]:
    #------------------------------------------------------------
        Adj_mat = A.copy()
        Adj_mat.setdiag(1)
        Adj_mat[Adj_mat>0] = 1

        Adj_mat = Adj_mat.tocoo()
        
        cat = torch.tensor([Adj_mat.row, Adj_mat.col], dtype=torch.long)
        bb = cat[:, edge_mask.argsort()[:]]
        
        self_indx1,self_indx2 = self_connection_index(cat,aa1),self_connection_index(cat,aa2)
        conn1,conn2 = delete_self(aa1,self_indx1),delete_self(aa2,self_indx2)
        sorted_conn2 = sort2edge(cat,conn1,conn2)
        
        asd = sample(list(conn1[0]),num_users) #random sampling
        qwe = find_indx(asd,conn1[0]) 
        fgh = sorted_conn2[qwe] #indexes of the edges from the other side of the connection.
        
        edge_index_new = torch.tensor(np.delete(np.array(bb),np.append(asd,list(fgh)),1)) # removing the edges

    #using this features to predict the class:
        log_logists_new = model(x,edge_index_new)
        y_pred_test_new = torch.argmax(log_logists_new, dim=1)[np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test))][user_add]
        distances, acc_at_161, latlon_true, latlon_pred = geo_eval_trail(Y_test[user_add], np.array(y_pred_test_new), U_test[user_add], classLatMedian, classLonMedian, userLocation)
        hav_distance.append(distances[0])
        latlon_tr.append(latlon_true[0])
        latlon_pre.append(latlon_pred[0])
        accuracy.append(acc_at_161)
        num_us.append(num_users)
        user_id.append(U_test[user_add])
    user_add += 1

df1 = pd.DataFrame(list(zip(user_id,num_us,latlon_tr,latlon_pre,hav_distance,accuracy)),columns =['user','num_users','latlon_tru','latlon_pred','haversine_distance',"acc_at_161"])


percent = [0,5,10,20,40,60,80,100]
df1['percent'] = percent *10

mean_pts = [np.mean(df1[df1['percent']==i]['haversine_distance']) for i in percent]
median_pts = [np.median(df1[df1['percent']==i]['haversine_distance']) for i in percent]

acc_pts = []
for i in percent:
    dist = list(df1[df1['percent']==i]['haversine_distance'])
    accuracy= len([d for d in dist if d < 161]) / float(len(dist))
    acc_pts.append(accuracy)


df1.to_csv('C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_edges_random.csv',index=False)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\mean_pts_edges_rm_random.txt", "wb") as fp: 
    pickle.dump(mean_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\median_pts_edges_rm_random.txt", "wb") as fp: 
    pickle.dump(median_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\acc_pts_edges_rm_random.txt", "wb") as fp: 
    pickle.dump(acc_pts, fp)