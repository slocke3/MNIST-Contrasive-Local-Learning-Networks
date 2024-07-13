from scipy.sparse import csc_matrix
from scipy.sparse import spdiags, diags
from scipy.sparse.linalg import spsolve, splu, minres 

from numpy import r_, ones, arange


# Task type contains all individual tasks that have the same input and output nodes\edges

# Make sparse incidence and constraint matrices for one type of tasks
def SparseIncidenceConstraintMatrix(SourceNodes, SourceEdges, TargetNodes, TargetEdges, GroundNodes, NN, EI, EJ):
    NE = len(EI)
    dF = []
    xF = []
    yF = []
    dC = []
    xC = []
    yC = []
    nc = NN
    nc2 = NN
    for i in range(len(GroundNodes)):
        dF.append(1.)
        xF.append(GroundNodes[i])
        yF.append(nc+i)
        dF.append(1.)
        xF.append(nc+i)
        yF.append(GroundNodes[i])

        dC.append(1.)
        xC.append(GroundNodes[i])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(GroundNodes[i])
    nc += len(GroundNodes)
    nc2 += len(GroundNodes)

    for i in range(len(SourceNodes)):
        dF.append(1.)
        xF.append(SourceNodes[i])
        yF.append(nc+i)
        dF.append(1.)
        xF.append(nc+i)
        yF.append(SourceNodes[i])

        dC.append(1.)
        xC.append(SourceNodes[i])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(SourceNodes[i])
    nc += len(SourceNodes)
    nc2 += len(SourceNodes)

    for i in range(len(SourceEdges)):
        dF.append(1.)
        xF.append(EI[SourceEdges[i]])
        yF.append(nc+i)
        dF.append(1.)
        xF.append(nc+i)
        yF.append(EI[SourceEdges[i]])

        dF.append(-1.)
        xF.append(EJ[SourceEdges[i]])
        yF.append(nc+i)
        dF.append(-1.)
        xF.append(nc+i)
        yF.append(EJ[SourceEdges[i]])

        dC.append(1.)
        xC.append(EI[SourceEdges[i]])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(EI[SourceEdges[i]])

        dC.append(-1.)
        xC.append(EJ[SourceEdges[i]])
        yC.append(nc2+i)
        dC.append(-1.)
        xC.append(nc2+i)
        yC.append(EJ[SourceEdges[i]])
    nc += len(SourceEdges)
    nc2 += len(SourceEdges)

    for i in range(len(TargetNodes)):
        dC.append(1.)
        xC.append(TargetNodes[i])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(TargetNodes[i])
    nc2 += len(TargetNodes)

    for i in range(len(TargetEdges)):
        dC.append(1.)
        xC.append(EI[TargetEdges[i]])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(EI[TargetEdges[i]])

        dC.append(-1.)
        xC.append(EJ[TargetEdges[i]])
        yC.append(nc2+i)
        dC.append(-1.)
        xC.append(nc2+i)
        yC.append(EJ[TargetEdges[i]])
    nc2 += len(TargetEdges)

    # Incidence matrix templates
    sDMF = csc_matrix((r_[ones(NE),-ones(NE)], (r_[arange(NE),arange(NE)], r_[EI,EJ])), shape=(NE, nc))
    sDMC = csc_matrix((r_[ones(NE),-ones(NE)], (r_[arange(NE),arange(NE)], r_[EI,EJ])), shape=(NE, nc2))

    # Constraint border Laplacian matrices
    sBLF = csc_matrix((dF,                   # data
                      (xF, yF)),             # coordinates
                      shape=(nc, nc))
    sBLC = csc_matrix((dC,                   # data
                      (xC, yC)),             # coordinates
                      shape=(nc2, nc2))
    
    # Matrix for cost computation
    sDot = sBLC[nc:,:nc]
    
    return (sDMF, sDMC, sBLF, sBLC, sDot)

def SquareGrid(a, b, Periodic=False, SecondNeighbor=False): # construct a square grid
    NN = a*b
    ys = arange(b) - b//2 + 0.5
    xs = arange(a) - a//2 + 0.5
    xs = -xs
    Y, X = meshgrid(xs, ys)
    Pos = c_[X.flatten(), Y.flatten()]


    EI = []
    EJ = []
    for i in range(b-1):
        for j in range(a-1):
            EI.append(i*a + j)
            EJ.append(i*a + j + 1)
            EI.append(i*a + j)
            EJ.append((i+1)*a + j)
        EI.append(i*a + a-1)
        EJ.append((i+1)*a + a-1)
    for j in range(a-1):
            EI.append((b-1)*a + j)
            EJ.append((b-1)*a + j + 1)
    NE0 = len(EI)

    if Periodic and not SecondNeighbor:
        for j in range(a):
            EI.append(j)
            EJ.append((b-1)*a + j)     

        for i in range(b):
            EI.append(i*a)
            EJ.append(i*a + a-1) 
            
    if SecondNeighbor and not Periodic:
        for i in range(b-1):
            EI.append(i*a + 0)
            EJ.append((i+1)*a + 0 + 1)
            for j in range(1, a-1):
                EI.append(i*a + j)
                EJ.append((i+1)*a + j + 1)
                EI.append(i*a + j)
                EJ.append((i+1)*a + j - 1)
            EI.append(i*a + a - 1)
            EJ.append((i+1)*a + a - 2)
        

    EI = array(EI)
    EJ = array(EJ)
    NE = len(EI)

    print(NN,NE)
    return NN, NE, EI, EJ, Pos
