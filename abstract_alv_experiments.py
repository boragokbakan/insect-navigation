import numpy as np
import matplotlib.pyplot as plt

'''ABSTRACT ALV SIMULATION CODE, RUNS STANDALONE'''

size_y, size_x  = (25,25)
nest_y,nest_x = (12,3) #target coordinates

maparr = np.zeros((size_y,size_x))
LANDMARK_PROBABILITY = 0.92 #the probability of existence of a landmark in any grid square, negative for manual setting

if LANDMARK_PROBABILITY>0:
    maparr = np.random.rand(size_y,size_x) #comment out to disable random landmarks
    maparr[maparr>LANDMARK_PROBABILITY] = 1
    maparr[maparr<1] = 0
else:
    #manual landmark positioning
    maparr[7][10] = 1
    maparr[7][16] = 1
    maparr[17][13] = 1

landmarks_y,landmarks_x=np.where(maparr==1)

def alv(nest_y,nest_x,target_y=0,target_x=0):
    #we get the relative position of each landmark from the agent location
    relative_y = landmarks_y-nest_y
    relative_x = landmarks_x-nest_x

    #maparr[nest_y][nest_x]=5
    print(maparr)
    print(relative_y, relative_x)

    #We get the length of each landmark vector, and normalise x and y values such that their hypothenus is equal to 1 (unit vector).
    c = np.sqrt(relative_y**2+relative_x**2) #x^2+y^2 = c^2
    unit_y, unit_x = relative_y/c, relative_x/c #normalise relative vectors to 1 (unit vectors)

    print(unit_y,unit_x, np.sqrt(unit_y**2+unit_x**2))

    unit_y, unit_x = np.sum(unit_y), np.sum(unit_x) #we sum all y and x coordinates of unit landmark vectors...
    unit_y, unit_x = unit_y/len(landmarks_y), unit_x/len(landmarks_x) #...and average them.

    #we get the vector sum
    relative_y = np.sum(relative_y)/len(relative_y)
    relative_x = np.sum(relative_x)/len(relative_x)



    print(relative_y,relative_x)
    print(unit_y,unit_x, np.sqrt(unit_y**2+unit_x**2))
    print("unit angles : ",np.arctan2(unit_y,unit_x))
    print(landmarks_x.shape,landmarks_y.shape)

    return (unit_y-target_y,unit_x-target_x)

alv(nest_y,nest_x)
py,px=np.arange(size_y),np.arange(size_x) #AL vector origins
vy, vx = np.zeros((size_y,size_x)), np.zeros((size_y,size_x)) #AL vectors themselves


#Compute ALV_target
target_y,target_x = alv(nest_y,nest_x)

#Computing home vector?
diff = False

for i in py:
    for j in px:
        if diff:
            vy[i][j],vx[i][j] = alv(i,j,target_y=target_y,target_x=target_x)
        else:
            vy[i][j],vx[i][j] = alv(i,j)

fig, ax = plt.subplots()

#Plot AL vectors
d = np.sqrt(vx**2+vy**2)
q = ax.quiver(px, py, vx, vy, units='xy',scale=1.1)
plt.scatter(landmarks_x,landmarks_y,s=20)
plt.title("Average Landmark Vector Field")
plt.show()

#Plot AL_current-AL_target vectors (i.e. home vectors)
d = np.sqrt((vx-target_y)**2+(vy-target_y)**2)
fig, ax = plt.subplots()
q = ax.quiver(px, py, vx-target_x, vy-target_y,  units='xy',scale=1.1)
plt.scatter(landmarks_x,landmarks_y,s=20)
plt.scatter(nest_x,nest_y, s=50, c='orange', marker='X')
plt.title("Homing Vector Field")
plt.show()
