
import numpy as np
import matplotlib.pyplot as plt 
import networkx as nx 



class MNIST_Diffuse : 

    def __init__(self , X_train , Alpha  = 0.8) : 
        self.X_train = X_train 
        self.n = X_train.shape[0] 
        self.Alpha = Alpha
        self.I = np.eye(self.n*self.n)
    

    def establish_edges(self) : 
        edges = []
        for i in range(28) : 
            for j in range(28) :
                
                current_node = str(self.X_train[i][j])+ "_"+str(i*28+j)
                if (i - 1)>=0 and (i-1) <=27 and j >=0 and j<=27 : 
                    nearby_node = str(self.X_train[i-1][j])+ "_"+str((i-1)*28 + j)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))

                if (i - 1)>=0 and (i-1) <=27 and j-1 >=0 and j-1<=27 : 
                    nearby_node = str(self.X_train[i-1][j-1])+ "_"+str((i-1)*28 + j-1)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))


                if (i-1 )>=0 and (i-1) <=27 and j +1 >=0 and j+1 <=27 : 
                    nearby_node = str(self.X_train[i-1][j+1])+ "_"+str((i-1)*28 + j+1)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))


                if (i)>=0 and (i) <=27 and j+1 >=0 and j+1<=27 : 
                    nearby_node = str(self.X_train[i][j+1])+ "_"+str(i*28 + j+1)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))


                if (i )>=0 and (i) <=27 and j -1>=0 and j-1<=27 : 
                    nearby_node = str(self.X_train[i][j-1])+ "_"+str(i*28 + j-1)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))


                if (i +1)>=0 and (i+1) <=27 and j >=0 and j<=27 : 
                    nearby_node = str(self.X_train[i+1][j])+ "_"+str((i+1)*28 + j)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))


                if (i + 1)>=0 and (i+1) <=27 and j+1 >=0 and j+1<=27 : 
                    nearby_node = str(self.X_train[i+1][j+1])+ "_"+str((i+1)*28 + j+1)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))


                if (i + 1)>=0 and (i+1) <=27 and j -1>=0 and j-1<=27 : 
                    nearby_node = str(self.X_train[i+1][j-1])+ "_"+str((i+1)*28 + j-1)
                    edges.append ((current_node,nearby_node,{'weight' : 1 }))
        return edges 


        
    def establish_laplacien(self) : 
        G = nx.Graph()
        edges = self.establish_edges()
        G.add_edges_from(edges) 
        L = nx.laplacian_matrix(G).toarray()
        return L
    
    

    def plat(self) : 
        X_train_plat = [] 
        for i in range(28) : 
            for j in range(28) : 
                X_train_plat .append(self.X_train[i][j])
        X_train_plat = np.array(X_train_plat)
        return X_train_plat

    def Calculate_X_diff_plat(self) :
        edges = self.establish_edges()
        L = self.establish_laplacien(edges) 
        X_train_plat = self.plat()
        X_diff_plat = np.dot(np.linalg.inv(self.I + self.Alpha*L), X_train_plat )
        return X_diff_plat
 


    def Calculate_X_diff(self) : 
        L = []
        Q = [] 
        X_diff_plat = self.Calculate_X_diff_plat()
        for i in range(28):
            Q = []
            for j in range(28) :
                Q.append (X_diff_plat[i*28+j] )
            L.append(Q)

        X_diff = np.array(L)
        return X_diff 



    def show_results(self) : 
        plt.subplot(1,2,1)
        plt.imshow(self.X_train,cmap = plt.get_cmap('gray'))
        plt.title("The initial hand-written digit ")

        X_diff = self.Calculate_X_diff()
        plt.subplot(1,2,2) 
        plt.imshow(X_diff,cmap= plt.get_cmap('gray'))
        plt.title("the image after diffusion ")

        plt.show()


