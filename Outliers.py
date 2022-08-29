import pandas as pd
from matplotlib import pyplot as plt
import os
os.chdir(r'C:\Users\Nikos\Desktop\DATA_SPOUDES\Lessons\Fall\Introduction to Big Data\Project\hetrec2011-lastfm-2k')

tags = pd.read_csv('user_taggedartists.dat', sep='\t',usecols=['userID','artistID','tagID'])
plays = pd.read_csv('user_artists.dat', sep='\t')


weights=plays.groupby(['artistID']).agg({"weight":"sum"})
weights=weights.reset_index()

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(weights.artistID,weights.weight) 
plt.xlabel("Artist ID")
plt.ylabel("Times Users Listened to an Artist")
plt.title("1. Listening Frequency of Artists")
plt.grid()

#======================================================================================================================================================

art2 = tags.groupby(["userID"]).agg({"tagID":"sum"}) 
art2=art2.reset_index()
art2

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(art2.userID,art2.tagID) 
plt.xlabel("User ID")
plt.ylabel("How many times a user Tagged")
plt.title("2. Frequency of Tags per User")
plt.grid()

#=======================================================================================================================================================

art = tags.groupby(["artistID"]).agg({"tagID":"sum"})
art=art.reset_index()
art

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(art.artistID,art.tagID) 

plt.xlabel("Artist ID")
plt.ylabel("How many times an artist got tagged")
plt.title("3. Frequency of Tags per Artist")
plt.grid()

#=======================================================================================================================================================



art21= tags.groupby(["userID"]).agg({"tagID":"sum"})
art21=art21.reset_index()

target=art21.tagID                                  
art21['zscore']=(target-target.mean())/target.std()
art21.sort_values(['zscore'],ascending=False) 

work=art21[(art21['zscore']<3) & (art21['zscore']>-3)]

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(work.userID,work.tagID) # tags per user
plt.xlabel("User ID")
plt.ylabel("How many times a user Tagged")
plt.title("4. Without Outliers Frequency of Tags per User")
plt.grid()

#=======================================================================================================================================================

art4 = tags.groupby(["tagID"]).agg({"userID":"sum"}) # graph 5 gia OUTLIERS
art4=art4.reset_index()

target2=art4.tagID                                  
art4['zscore']=(target2-target2.mean())/target2.std()
art4.sort_values(['zscore'],ascending=False)

work2=art4[(art4['zscore']<3) & (art4['zscore']>-3)]

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(work2.tagID,work2.userID)
plt.xlabel("tagID")
plt.ylabel("User Count")
plt.title("5. How many times Tags were used by users")
plt.grid()


plt.show()