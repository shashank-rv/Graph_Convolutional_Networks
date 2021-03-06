from libraries import *
from geo import Geo
from load_functions import *
from eval_functions import *
from utilities import *


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

explainer = GNNExplainer(model, epochs=200)

def sorted_indexes(nz_indexes):
    
    indices = [node_feat_mask.argsort().tolist().index(i) for i in tqdm(nz_indexes)]
    prio_nz_indices = [i[1] for i in sorted(list(zip(indices,nz_indexes)))]
    
    return prio_nz_indices

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

#for both features and edges:

test_index = np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test)).tolist()
hav_distance = [] #distance between the true and predcited labels for each user
latlon_tr = [] #true latitutde and longitude of the users
latlon_pre = []# predicted latitude and longitude of the users
accuracy = []
num_us = []
num_feat = []
user_id = []
user_add = 0

for user in test_index[0:100]:
    #explaining the node
    node_feat_mask, edge_mask = explainer.explain_node(user, x, edge_index)
    
    #retrieving the non-zero indexes of the features
    nz_indexes = np.array(x[user]).nonzero()[0]
    
    # retreving the prio. non zero indexes for features
    top_nz_indexes = sorted_indexes(nz_indexes)
    
    #prio. edges based on explaination and edges related to the particular user
    prio_edge = priority_edges(edge_index,user)

    aa1 = np.where(np.array(edge_index[:, edge_mask.argsort()[:]][0])==user)
    aa2 = np.where(np.array(edge_index[:, edge_mask.argsort()[:]][1])==user)
    
    
    for num_features,num_users in list(zip([perc(nz_indexes,0),perc(nz_indexes,0.05),perc(nz_indexes,0.10),perc(nz_indexes,0.20),perc(nz_indexes,0.40),perc(nz_indexes,0.60),perc(nz_indexes,0.80),perc(nz_indexes,1)],[perc(prio_edge[0],0),perc(prio_edge[0],0.05),perc(prio_edge[0],0.10),perc(prio_edge[0],0.20),perc(prio_edge[0],0.40),perc(prio_edge[0],0.60),perc(prio_edge[0],0.80),perc(prio_edge[0],1)])):
        
        x_feature_rm = x.detach().clone()
        if  num_features == 0:
            top_features = []
        else:
            top_features = top_nz_indexes[-num_features:]
        x_feature_rm[user][top_features]=0
        
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
        conn1_rev,sorted_conn2_rev = revere_indexes(conn1[0]),revere_indexes(sorted_conn2)
        
        edge_index_new = torch.tensor(np.delete(np.array(bb),np.append(conn1_rev[:num_users],sorted_conn2_rev[:num_users]),1)) # removing the edges

    #using this features to predict the class:
        log_logists_new = model(x_feature_rm,edge_index_new)
        y_pred_test_new = torch.argmax(log_logists_new, dim=1)[np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test))][user_add]
        distances, acc_at_161, latlon_true, latlon_pred = geo_eval_trail(Y_test[user_add], np.array(y_pred_test_new), U_test[user_add], classLatMedian, classLonMedian, userLocation)
        hav_distance.append(distances[0])
        latlon_tr.append(latlon_true[0])
        latlon_pre.append(latlon_pred[0])
        accuracy.append(acc_at_161)
        num_us.append(num_users)
        num_feat.append(num_features)
        user_id.append(U_test[user_add])
    user_add += 1


df4 = pd.DataFrame(list(zip(user_id,num_feat,num_us,latlon_tr,latlon_pre,hav_distance,accuracy)),columns =['user','num_features','num_edges','latlon_tru','latlon_pred','haversine_distance',"acc_at_161"])

percent = [0,5,10,20,40,60,80,100]
df4['percent'] = percent *100

mean_pts = [np.mean(df4[df4['percent']==i]['haversine_distance']) for i in percent]
median_pts = [np.median(df4[df4['percent']==i]['haversine_distance']) for i in percent]

acc_pts = []
for i in percent:
    dist = list(df4[df4['percent']==i]['haversine_distance'])
    accuracy= len([d for d in dist if d < 161]) / float(len(dist))
    acc_pts.append(accuracy)

df4.to_csv('C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_both.csv',index=False)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\mean_pts_both_rm.txt", "wb") as fp: 
    pickle.dump(mean_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\median_pts_both_rm.txt", "wb") as fp: 
    pickle.dump(median_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\acc_pts_both_rm.txt", "wb") as fp: 
    pickle.dump(acc_pts, fp)






