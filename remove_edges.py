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

explainer = GNNExplainer(model, epochs=200)

test_index = np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test)).tolist()
hav_distance = [] #distance between the true and predcited labels for each user
latlon_tr = [] #true latitutde and longitude of the users
latlon_pre = []# predicted latitude and longitude of the users
accuracy = []
num_us = []
user_id = []
user_add = 0

for user in test_index[0:100]:
    #explaining the node
    node_feat_mask, edge_mask = explainer.explain_node(user, x, edge_index)

    #priotizig the edges based on explaination
    prio_edge = priority_edges(edge_index,user)
    
    
    for num_users in [perc(prio_edge[0],0),perc(prio_edge[0],0.05),perc(prio_edge[0],0.10),perc(prio_edge[0],0.20),perc(prio_edge[0],0.40),perc(prio_edge[0],0.60),perc(prio_edge[0],0.80),perc(prio_edge[0],1)]:
    #------------------------------------------------------------
        Adj_mat = A.copy()
        Adj_mat.setdiag(1)
        Adj_mat[Adj_mat>0] = 1
    #------------------------------------------------------------        
        # for indexes in range(num_users):
        #     Adj_mat[prio_edge[0][indexes],prio_edge[1][indexes]]= 0
        Adj_mat = Adj_mat.tocoo()

        cat = torch.tensor([Adj_mat.row, Adj_mat.col], dtype=torch.long)
        
        bb = cat[:, edge_mask.argsort()[:]]
        
        edge_index_new = torch.tensor(np.delete(np.array(bb),aa[0][:num_users],1),dtype=torch.long) # removing the edges(one-side removal)
        #edge_index_new = torch.tensor([Adj_mat.row, Adj_mat.col], dtype=torch.long)

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
df1['percent'] = percent *100

mean_pts = [np.mean(df1[df1['percent']==i]['haversine_distance']) for i in percent]
median_pts = [np.median(df1[df1['percent']==i]['haversine_distance']) for i in percent]

acc_pts = []
for i in percent:
    dist = list(df1[df1['percent']==i]['haversine_distance'])
    accuracy= len([d for d in dist if d < 161]) / float(len(dist))
    acc_pts.append(accuracy)


df1.to_csv('C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_edges.csv',index=False)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\mean_pts_edges_rm.txt", "wb") as fp: 
    pickle.dump(mean_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\median_pts_edges_rm.txt", "wb") as fp: 
    pickle.dump(median_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\acc_pts_edges_rm.txt", "wb") as fp: 
    pickle.dump(acc_pts, fp)