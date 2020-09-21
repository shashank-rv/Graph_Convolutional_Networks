from libraries import *

def load_geotext(raw_dir, name):
    filename = osp.join(raw_dir, name)
    #print(raw_dir, name)
    #geo_data = load_obj(filename)
    #A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = geo_data
    geo_data = preprocess_data(raw_dir, builddata=False)
    #vocab = None
    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, vocab = geo_data
    A.setdiag(1)
    A[A>0] = 1
    A = A.tocoo()
    edge_index = torch.tensor([A.row, A.col], dtype=torch.long)
    #A is the normalised laplacian matrix as A_hat in Kipf et al. (2016).
    #The X_? and Y_? should be concatenated to be feed to GCN.

    X = sp.sparse.vstack([X_train, X_dev, X_test])
    X = X.todense().astype(np.float32)
    X = torch.from_numpy(X)
    '''
    X = X.tocoo()
    values = X.data
    indices = np.vstack((X.row, X.col))
    X = torch.sparse_coo_tensor(indices = torch.tensor(indices), values = torch.tensor(values), size=X.shape)
    '''

    if len(Y_train.shape) == 1:
        y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        y = np.vstack((Y_train, Y_dev, Y_test))
    #print(A.shape, X.shape, y.shape)
    y = y.astype(np.int64)
    y = torch.from_numpy(y)
    
    #get train/dev/test indices in X, Y, and A.
    train_index = torch.arange(0, X_train.shape[0], dtype=torch.long)
    val_index = torch.arange(X_train.shape[0], X_train.shape[0] + X_dev.shape[0], dtype=torch.long)
    test_index = torch.arange(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0], dtype=torch.long)
    
    #print(val_index, y.size(0))
    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    data = Data(x=X, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, geo_data


def load_obj(filename, serializer=pickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin, encoding='latin1')
    return obj



def get_geo_data(raw_dir, name):
    filename = osp.join(raw_dir, name)
    #print(raw_dir, name)
    geo_data = load_obj(filename)
    #A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = geo_data
    return geo_data


def perc(indexes,num):
    return int(np.ceil(len(indexes)*num))