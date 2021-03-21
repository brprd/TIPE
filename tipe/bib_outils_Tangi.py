def obstacle(R,o):
    n = len(R)
    i, j = o[0][0], o[0][1]
    k = o[1]
    rm = o[2]
    R[i][j] = R[i][j] + rm
    for r in range(1,k):
        s = _carre_centre((i,j),r)
        for c in s:
            if in_matrice(c,n):
                R[c[0]][c[1]] = R[c[0]][c[1]] + (rm//(r+1))
    return R

def retire_obstacle(R,o):
    n = len(R)
    i, j = o[0][0], o[0][1]
    k = o[1]
    rm = o[2]
    R[i][j] = R[i][j] - rm
    for r in range(1,k):
        s = _carre_centre((i,j),r)
        for c in s:
            if in_matrice(c,n):
                R[c[0]][c[1]] = R[c[0]][c[1]] - (rm//(r+1))
    return R


def absord(L):
    X = []
    Y = []
    for c in L:
        X.append(c[1])
        Y.append(-c[0])
    return X,Y

def in_matrice(c,n):
    i, j = c[0], c[1]
    return (i >= 0 and i < n and j >= 0 and j < n)



def _carre_centre(c,r):
    s = []
    i, j = c[0], c[1]
    s.append((i-r,j))
    s.append((i+r,j))
    s.append((i,j-r))
    s.append((i,j+r))
    for p in range(1,r+1):
        s.append((i-r,j+p))
        s.append((i-r,j-p))
        s.append((i+r,j+p))
        s.append((i+r,j-p))
    for p in range(1,r):
        s.append((i+p,j-r))
        s.append((i-p,j-r))
        s.append((i+p,j+r))
        s.append((i-p,j+r))
    return s


