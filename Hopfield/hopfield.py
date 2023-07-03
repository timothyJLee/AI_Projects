#!/usr/bin/python3


# made using this site
# http://www.cs.ucla.edu/~rosen/161/notes/hopfield.html


import numpy as np

def calcEngy(w,IN):

  energy=0
  size = len(IN)
  for i in range(size):
    for j in range(1,size-i):
      energy += w[i][j+i]*IN[i]*IN[j+i] 
 
  return -.5*energy


def makeADJMat(size,lists):
 

  w = np.zeros((size,size))
 

  #only useful for indexing corners of weight matrix
  #iu = np.triu_indices(size,1)
 
  # example of two networks
  # l1 = lists[0]
  # l2 = lists[1]
  # for i in range(size):
  #   for j in range(1,size-i):
  #     w[i][j+i] = ((2*l1[i]-1)*(2*l1[j+i]-1))+((2*l2[i]-1)*(2*l2[j+i]-1))
  #     w[j+i][i] = w[i][j+i]
  # print(w)
  # w = np.zeros((size,size))
 
  for l in lists:
    for i in range(size):
      for j in range(1,size-i):
        w[i][j+i] += ((2*l[i]-1)*(2*l[j+i]-1))
        w[j+i][i] = w[i][j+i]
 
  #print(w.shape)
  return w

##Hackish oneliner to make adjacency list not used yet. Although will be once my code is optimized :)
def makeAdjList(size,l1,l2):
  i,u = np.triu_indices(size,1)
  #print(i,u)
  return [(2*l1[x]-1)*(2*l1[y]-1)+(2*l2[x]-1)*(2*l2[y]-1) for x,y in zip(i,u)]

class HF:
  def __init__(self, NI, patterns):

    patters = np.array(patterns)

    self.w = makeADJMat(NI,patterns)


  def update(self,IN):


    #create random permutation of length of input
    order= np.random.permutation(len(IN))
    change = 1
    count = 0
    changes=[]
    while change != 0:
      change = 0

     

      print("Energy",calcEngy(self.w,IN))
     
      for x in order:


        energy= np.dot(self.w[x],IN)

        if energy < 0 and IN[x] != 0:
          change+=1
          IN[x]=0
        elif energy >= 0 and IN[x] !=1:
           change+=1
           IN[x]=1
      changes.append(change)
      count+=1
    return (IN,count,changes)

  def compare(self,p1,p2):
    return True if (p1==p2).all()  else False

