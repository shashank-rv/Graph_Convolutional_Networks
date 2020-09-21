from libraries import *

def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
        
    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred

def get_distance(user1, user2, userLocation):
    lat1, lon1 = userLocation[user1].split(',')
    lat2, lon2 = userLocation[user2].split(',')
    lat1, lon1 = float(lat1), float(lon1)
    lat2, lon2 = float(lat2), float(lon2)
    distance = haversine((lat1, lon1), (lat2, lon2))
    return distance


def geo_eval_trail(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    #assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    user = U_eval
    location = userLocation[user].split(',')
    lat, lon = float(location[0]), float(location[1])
    latlon_true.append([lat, lon])
    prediction = str(y_pred)
    lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
    latlon_pred.append([lat_pred, lon_pred])  
    distance = haversine((lat, lon), (lat_pred, lon_pred))
    distances.append(distance)

    acc_at_161 = len([d for d in distances if d < 161]) / float(len(distances))

    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
        
    return distances,acc_at_161, latlon_true, latlon_pred