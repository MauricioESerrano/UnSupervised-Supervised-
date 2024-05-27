import numpy as np

# Approach 1:
# Run PCA once to go directly from R100 to R25 constructing new data set Z1

# Approach 2:
# First run PCA with k = 50 to create intermediate data set Z' of points in R50 , then run PCA with k = 25 on Z' 
# to create new data set Z2.

# Z1 = Z2 Ideally


def covMatrix(data):
    mu = data.mean(axis=0)         #calculate mu
    Z = data - mu                  #Calculate Z by centering columns of data      
    C = 1 / len(data) * Z.T @ Z    #Calculate C = (1/n)( Z^T * Z )  
    return C


def approach1(data):
    C = covMatrix(data)
    eigenvalues,eigenvectors = np.linalg.eigh(C) 
    top25 = eigenvectors[:,:25]
    dataset_1 = np.dot(data,top25)

    return dataset_1

    
def approach2(data):
    C = covMatrix(data)
    dataset_2 = []
    dataset_intermediate = []
    eigenvalues,eigenvectors = np.linalg.eigh(C)

    print(eigenvectors[0])

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    print(eigenvectors[0])

    
    top50 = eigenvectors[:,:50]

    dataset_intermediate = np.dot(data,top50)
    
    Cprime = covMatrix(dataset_intermediate)

    eigenvalues_,eigenvectors_ = np.linalg.eigh(Cprime)
    
    top25 = eigenvectors_[:,:25]
    dataset_2 = np.dot(dataset_intermediate,top25)

    return dataset_2



# ------------------------------------------------------------------
#                         Test Function
# ------------------------------------------------------------------

np.random.seed(42)
X = np.random.randint(1, 100, size=(1000, 100))

def test_pca_approaches(dataset_1, dataset_2):
    dataset_1 = abs(dataset_1)
    dataset_2 = abs(dataset_2)
    if np.allclose(dataset_1, dataset_2):
        print("Test Passed!")
    else:
        print("Test Failed!")

dataset_1 = approach1(X)
dataset_2 = approach2(X)

print(test_pca_approaches(dataset_1,dataset_2))