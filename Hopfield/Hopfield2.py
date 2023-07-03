
#!/usr/bin/python3
#some code help from http://pmatigakis.wordpress.com/2014/01/18/character-recognition-using-hopfield-networks/
import numpy as np
import hopfield


def rowstr(row,row2):
  return ''.join('+' if e > 0 else '-' for e in row) + (' '*5) + ''.join('+' if e > 0 else '-' for e in row2)


def view(test,result):
  print("input","   ","output")
  img = (test > 0).astype(int).reshape((10,10))
  img2 = (result > 0).astype(int).reshape((10,10))
  print ('\n'.join([rowstr(row,row2) for row,row2 in zip(img,img2)]))
  print()

def test():
  #Some Test data
  #Create the training patterns
  a = np.array([[0,0,0,0,1,1,0,0,0,0],
                [0,0,0,1,0,0,1,0,0,0],
                [0,0,1,0,0,0,0,1,0,0],
                [0,1,0,0,0,0,0,0,1,0],
                [1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1]])

  
  b = np.array([[1,1,1,1,1,1,1,1,1,0],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,0],
                #[0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,0],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,0]])
  
  c = np.array([
                [1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1]
                ])
  #really a Z
  # d = np.array([[1,1,1,1,1,1,1,1,1,1],
  #               [0,0,0,0,0,0,0,0,1,1],
  #               [0,0,0,0,0,0,0,1,1,0],
  #               [0,0,0,0,0,1,1,0,0,0],
  #               [0,0,0,0,0,1,0,0,0,0],
  #               [0,0,0,0,1,0,0,0,0,0],
  #               [0,0,0,1,1,0,0,0,0,0],
  #               [0,0,1,1,0,0,0,0,0,0],
  #               [0,1,1,0,0,0,0,0,0,0],
  #               [1,1,1,1,1,1,1,1,1,1]])

  d = np.array([[0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,1],
                [0,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1]])

  ro = 10
  col = 10

  network = hopfield.HF(100,[a.flatten(),b.flatten(),c.flatten()])
    #d.flatten()]) 
  a_test =  a.flatten()#(a.flatten()*2)-1


  for i in range(5):
      p = np.random.randint(0, (ro*col))
      a_test[p] *= 0

  a_result,_,_ = network.update(a_test.copy())


  a_result.shape =(ro,col)
  a_test.shape = (ro,col)
  print("  ### A ###")
  view(a_test,a_result)

  b_test =  (b.flatten()*2)-1
  
  for i in range(4):
      p = np.random.randint(0, (ro*col))
      b_test[p] *= 0
      
  b_result,_,_ = network.update(b_test.copy())
  
  b_result.shape = (ro,col)
  b_test.shape = (ro,col)
 
  print("  ### B ###")
  view(b_test,b_result)

 
  c_test =  c.flatten()#(c.flatten()*2)-1
  
  for i in range(4):
      p = np.random.randint(0, (ro*col))
      c_test[p] *= 0
      
  c_result,_,_ = network.update(c_test.copy())
  
  c_result.shape = (ro,col)
  c_test.shape = (ro,col)
 
  print("  ### C ###")
  view(c_test,c_result)


  d_test = d.flatten()  #(d.flatten()*2)-1
  

  for i in range(4):
      p = np.random.randint(0, (ro*col))
      d_test[p] *= 0
      
  d_result,_,_ = network.update(d_test.copy())
  
  d_result.shape = (ro,col)
  d_test.shape = (ro,col)
 
  print("  ### D ###")
  view(d_test,d_result)


if __name__ == "__main__":
    test()





