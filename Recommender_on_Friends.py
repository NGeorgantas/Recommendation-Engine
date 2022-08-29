import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import os
os.chdir(r'C:\Users\Nikos\Desktop\DATA_SPOUDES\Lessons\Fall\Introduction to Big Data\Project\Final')

tags = pd.read_csv('user_friends.dat', sep='\t')
artists=pd.read_csv('artists.dat', sep='\t',usecols=['id','name'])
plays = pd.read_csv('user_artists.dat', sep='\t')

df=pd.merge(artists, plays, how='inner', left_on='id', right_on='artistID')
df=df.rename(columns={"weight": "playCount"})

def pivot_maker_for_user(x):
    f=tags[tags['userID']==x]
    teliko1=df[df.userID.isin(f.friendID)]
    teliko2=df[df.userID.isin(f.userID)]
    teliko3=pd.concat([teliko1,teliko2])

    target=np.log(df.playCount)
    teliko3['logged']=target

    pc = teliko3.logged
    logged_scaled = (pc - pc.min()) / (pc.max() - pc.min())

    teliko3 = teliko3.assign(Scaled=logged_scaled)

    pivoterino=teliko3.pivot(index = 'userID', columns = 'name', values = 'Scaled')

    pivoterino=pivoterino.fillna(0)
    
    return pivoterino

            

def calculator(x):
    wide_artist_data=x.fillna(0)                                                           # Values of zero mean that the user has not listened to this artist 

    distance_matrix = cosine_similarity(wide_artist_data)                                  # We do cosine similarity for the users here to find the similarities between them 


    distDF = pd.DataFrame(distance_matrix, columns=(wide_artist_data.index), index=(wide_artist_data.index)) # We put the distance matrix values in a dataframe to have a better view since and also se the dataframe later to find neighbours

    def calc_neighbourhood(s, k):
        return [[x for x in np.argsort(s[i]) if x != i][len(s) - 1: len(s) - k - 2: -1] for i in range(len(s))] # This functions finds neighbours

    s=distDF.values
    k=2                                                                                                           # The number of neighbours to find
    nb=calc_neighbourhood(s, k)                                                                                   # In nb are stored the neighbours

    wide_artist_array=wide_artist_data.values                                                                     # We make an array of the matrix we made before with the user id and artists and ratings of users 


    masked = np.ma.masked_equal(wide_artist_array, 0)                                                               # We mask the zeros and find the mean rating for each user in order to use it with collaborative filtering
    user_mean_rating=masked.mean(axis=1)

    dummy=wide_artist_array.copy()                                                                      # We make a copy of wide_artist_array and we are going to do the predictions on this.. All the values in this array will change

    for i in range(len(wide_artist_array)):
        
        
        umr=user_mean_rating[i]
        
        nbs1=nb[i][0]   
        nbs2=nb[i][1]   

        unr1=user_mean_rating[nbs1]
        unr2=user_mean_rating[nbs2]

        dist1=distance_matrix[i][nbs1]                                                                  
        dist2=distance_matrix[i][nbs2]

        for j in range(len(wide_artist_array[i])):
            
            dummy[i][j] = umr + (   (dist1 * ( wide_artist_array[nbs1][j] - unr1)   + dist2 * (wide_artist_array[nbs2][j] - unr2)) / ( dist1 + dist2 )   )   #These 2 for loop make the predictions and print them on the dummy array


    dummy=np.nan_to_num(dummy)

    # Next we define some functions to find MAE and RMSE between the dummy and wide_artist_array which are the predictions and the starting ratings

    def mae(p, a):
        return sum(map(lambda x: abs(x[0] - x[1]), zip(p, a))) / len(p)

    def flatten(l):
        return [x for r in l for x in r]

    def rmse(p, a):
        return math.sqrt(sum(map(lambda x: (x[0] - x[1]) ** 2, zip(p, a))) / len(p))

    print('\nMAE: {:.4f}'.format(mae(flatten(wide_artist_array), flatten(dummy))))
    print('\nRMSE: {:.4f}'.format(rmse(flatten(wide_artist_array), flatten(dummy))))


    masked_wide = np.ma.masked_equal(wide_artist_array, 0)                                      # Here we mask the starting array with the ratings of users in the places that there were not values  
    masked_dummy=np.ma.masked_where(np.ma.getmask(masked_wide), dummy)                          # We mask the same way the dummy array in order to only compare the prdicted and actual ratings 

    actual=masked_wide.compressed()
    predicted=masked_dummy.compressed()

    threshol=np.mean(user_mean_rating)                                        # As a threshold we put the mean rating of all users



    z=zip(actual,predicted)
    z=list(z)

    threshold=threshol
    tp=0
    tn=0
    fp=0
    fn=0
    for i,j in z:
        if i>=threshold and j>=threshold:
            tp=tp+1
        elif i>=threshold and j<threshold:
            fn=fn+1
        elif i<threshold and j>=threshold:
            fp=fp+1

    precision=tp/(tp+fp)
    recall = tp/(tp+fn)
    f1=2*precision*recall/(precision+recall)
    print("\n")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1 = {f1}")  




    recommend=np.ma.masked_where(wide_artist_array != 0, wide_artist_array)     # We mask the the dummy list in the places where our starting array had values and we get this way only the predicted movies
    recommend_dummy=np.ma.masked_where(np.ma.getmask(recommend), dummy)         # We will use this to propose some movies to the users that they have no seen before
    recommend_dummy=recommend_dummy.filled(0)

    sor=np.argsort(recommend_dummy,axis=1)

    column=wide_artist_data.columns.tolist()
    row=wide_artist_data.index.tolist()

    j=0
    for i in range(len(sor)):
        if j==1:
            break
        j=j+1
        first_match=sor[i][-1] 
        second_match=sor[i][-2] 
        third_match=sor[i][-3]
        fourth_match=sor[i][-4]
        fifth_match=sor[i][-5]

        
        print("\n")
        print(f'Arists recommended for User{row[i]}: {column[first_match]}, with predicted rating {recommend_dummy[i][first_match]}\n\
                                {column[second_match]}, with predicted rating {recommend_dummy[i][second_match]}\n\
                                {column[third_match]}, with predicted rating {recommend_dummy[i][third_match]}\n\
                                {column[fourth_match]}, with predicted rating {recommend_dummy[i][fourth_match]}\n\
                                {column[fifth_match]}, with predicted rating {recommend_dummy[i][fifth_match]}\n')



L=df.userID.unique()
L=sorted(L)

u=0
for k in L:
    if u==5:
        break
    u=u+1
    x=pivot_maker_for_user(k)          # This proposes some movies for users based on the similarity with their friends. THis is done for the first 5 users, since u==5 and it breaks there
    calculator(x)                      # the problem is that if a user has no 1 or less friends the programm brakes so bellow if you call the fuction manually you can see the propositions for each user

#x=pivot_maker_for_user(2)             # If you want to use this way comment out the above and use these 2 lines here. The number 2 gives propositions for user 2. IF you change the number you can get propositions 
#calculator(x)                         # for other users 