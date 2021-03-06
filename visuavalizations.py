from libraries import *

percent = [0,5,10,20,40,60,80,100]

# df1 = pd.read_csv("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_features.csv")

# df1['percent'] = percent *100

# mean_pts = [np.mean(df1[df1['percent']==i]['haversine_distance']) for i in percent]
# median_pts = [np.median(df1[df1['percent']==i]['haversine_distance']) for i in percent]

# acc_pts = []
# for i in percent:
#     dist = list(df1[df1['percent']==i]['haversine_distance'])
#     accuracy= len([d for d in dist if d < 161]) / float(len(dist))
#     acc_pts.append(accuracy)

# f, axs = plt.subplots(2,2,figsize=(15,15))
# axs[0,0].set_title("mean")
# axs[0,0].plot(mean_pts)
# axs[0,1].set_title("median")
# axs[0,1].plot(median_pts)
# axs[1,0].set_title("accuracy")
# axs[1,0].plot(acc_pts)
# plt.show()


# df2 = pd.read_csv("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_edges.csv")
# df2['percent'] = percent *100

# mean_pts_2 = [np.mean(df2[df2['percent']==i]['haversine_distance']) for i in percent]
# median_pts_2 = [np.median(df2[df2['percent']==i]['haversine_distance']) for i in percent]

# acc_pts_2 = []
# for i in percent:
#     dist = list(df2[df2['percent']==i]['haversine_distance'])
#     accuracy= len([d for d in dist if d < 161]) / float(len(dist))
#     acc_pts_2.append(accuracy)

# f, axs = plt.subplots(2,2,figsize=(15,15))
# axs[0,0].set_title('mean')
# axs[0,0].plot(mean_pts_2)
# axs[0,1].set_title("median")
# axs[0,1].plot(median_pts_2)
# axs[1,0].set_title("acc")
# axs[1,0].plot(acc_pts_2)
# plt.show()




# df3 = pd.read_csv("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_both.csv")
# df3['percent'] = percent *100

# mean_pts_3 = [np.mean(df3[df3['percent']==i]['haversine_distance']) for i in percent]
# median_pts_3 = [np.median(df3[df3['percent']==i]['haversine_distance']) for i in percent]

# acc_pts_3 = []
# for i in percent:
#     dist = list(df3[df3['percent']==i]['haversine_distance'])
#     accuracy= len([d for d in dist if d < 161]) / float(len(dist))
#     acc_pts_3.append(accuracy)

# f, axs = plt.subplots(2,2,figsize=(15,15))
# axs[0,0].set_title("mean")
# axs[0,0].plot(mean_pts_3)
# axs[0,1].set_title("median")
# axs[0,1].plot(median_pts_3)
# axs[1,0].set_title("acc")
# axs[1,0].plot(acc_pts_3)
# plt.show()


# #random removal visuavalizations:


# df4 = pd.read_csv("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_features_random.csv")
# df4['percent'] = percent *100

# mean_pts_4 = [np.mean(df4[df4['percent']==i]['haversine_distance']) for i in percent]
# median_pts_4 = [np.median(df4[df4['percent']==i]['haversine_distance']) for i in percent]

# acc_pts_4 = []
# for i in percent:
#     dist = list(df4[df4['percent']==i]['haversine_distance'])
#     accuracy= len([d for d in dist if d < 161]) / float(len(dist))
#     acc_pts_4.append(accuracy)

# f, axs = plt.subplots(2,2,figsize=(15,15))
# axs[0,0].set_title("mean")
# axs[0,0].plot(mean_pts_4)
# axs[0,1].set_title("median")
# axs[0,1].plot(median_pts_4)
# axs[1,0].set_title("accuracy")
# axs[1,0].plot(acc_pts_4)
# plt.show()


# df5 = pd.read_csv("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_edges_random.csv")
# df5['percent'] = percent *100

# mean_pts_5 = [np.mean(df5[df5['percent']==i]['haversine_distance']) for i in percent]
# median_pts_5 = [np.median(df5[df5['percent']==i]['haversine_distance']) for i in percent]

# acc_pts_5 = []
# for i in percent:
#     dist = list(df5[df5['percent']==i]['haversine_distance'])
#     accuracy= len([d for d in dist if d < 161]) / float(len(dist))
#     acc_pts_5.append(accuracy)

# f, axs = plt.subplots(2,2,figsize=(15,15))
# axs[0,0].set_title('mean')
# axs[0,0].plot(mean_pts_5)
# axs[0,1].set_title("median")
# axs[0,1].plot(median_pts_5)
# axs[1,0].set_title("acc")
# axs[1,0].plot(acc_pts_5)
# plt.show()


df6 = pd.read_csv("C:\\Users\\61484\\Graph_Convolutional_Networks\\saved_files\\remove_both_random.csv")
df6['percent'] = percent *100

mean_pts_6 = [np.mean(df6[df6['percent']==i]['haversine_distance']) for i in percent]
median_pts_6 = [np.median(df6[df6['percent']==i]['haversine_distance']) for i in percent]

acc_pts_6 = []
for i in percent:
    dist = list(df6[df6['percent']==i]['haversine_distance'])
    accuracy= len([d for d in dist if d < 161]) / float(len(dist))
    acc_pts_6.append(accuracy)

f, axs = plt.subplots(2,2,figsize=(15,15))
axs[0,0].set_title("mean")
axs[0,0].plot(mean_pts_6)
axs[0,1].set_title("median")
axs[0,1].plot(median_pts_6)
axs[1,0].set_title("acc")
axs[1,0].plot(acc_pts_6)
plt.show()