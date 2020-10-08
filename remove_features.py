from libraries import *
from geo import Geo
from load_functions import *
from eval_functions import *
from utilities import *

dataset = 'geotext'
path = "C:\\Users\\61484\\Graph_Convolutional_networks\\data\\geo"
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

print(model)

def sorted_indexes(nz_indexes):
    
    indices = [node_feat_mask.argsort().tolist().index(i) for i in tqdm(nz_indexes)]
    prio_nz_indices = [i[1] for i in sorted(list(zip(indices,nz_indexes)))]
    
    return prio_nz_indices

explainer = GNNExplainer(model, epochs=200)



test_index = np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test)).tolist()
hav_distance = [] #distance between the true and predcited labels for each user
latlon_tr = [] #true latitutde and longitude of the users
latlon_pre = []# predicted latitude and longitude of the users
accuracy = []
num_us = []
user_id = []
pred_act = []
pred_pred = []
percen = []
count = 0


for user in tqdm(test_index[0:10]):
   
    log_logists = model(x, edge_index)
    y_pred_test = torch.argmax(log_logists, dim=1)[np.arange(len(U_train + U_dev), len(U_train + U_dev + U_test))]
    y_pred_test = y_pred_test.detach().numpy()[count]
    pred_act.append(y_pred_test)
    
    #explaining the node
    node_feat_mask, edge_mask = explainer.explain_node(user, x, edge_index)
    
    #getting the non-zero indexes
    nz_indexes = np.array(x[user]).nonzero()[0]
    
    #sorting the non-zero indexes based on explained priority
    top_nz_indexes = sorted_indexes(nz_indexes)
           
    #looping throught the number of features removed:
    for num_features in tqdm([perc(top_nz_indexes,0),perc(top_nz_indexes,0.05),perc(top_nz_indexes,0.10),perc(top_nz_indexes,0.20),perc(top_nz_indexes,0.40),perc(top_nz_indexes,0.60),perc(top_nz_indexes,0.80),perc(top_nz_indexes,1)]):
        x_feature_rm = x.detach().clone()
        top_features = top_nz_indexes[-num_features:]
        x_feature_rm[user][top_features]=0
        log_logists_new = model(x_feature_rm, edge_index)
        y_pred_test_new = torch.argmax(log_logists_new, dim=1)[user]
        pred_pred.append(y_pred_test_new)
        distances, acc_at_161, latlon_true, latlon_pred = geo_eval_trail(Y_test[count], np.array(y_pred_test_new), U_test[count], classLatMedian, classLonMedian, userLocation)
        hav_distance.append(distances[0])
        latlon_tr.append(latlon_true[0])
        latlon_pre.append(latlon_pred[0])
        accuracy.append(acc_at_161)
        num_us.append(num_features)
        user_id.append(U_test[count])
    count += 1
    #print(f"mean:{mean} median: {median} acc: {acc}")
    #after.append(y_pred_test_new)
    #print(f"after {y_pred_test_new}")

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


df1.to_csv('C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_features.csv',index=False)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\mean_pts_feature_rm.txt", "wb") as fp: 
    pickle.dump(mean_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\median_pts_feature_rm.txt", "wb") as fp: 
    pickle.dump(median_pts, fp)

with open("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\acc_pts_feature_rm.txt", "wb") as fp: 
    pickle.dump(acc_pts, fp)



