import numpy as np

def intersect(p, d, k, e=0.00001):
    """ p: xyz point; d: ray direction, default [0,0,1];
    k: 3X3 matrix of xyz coord of triangle [(x,y,z), (x,y,z), (x,y,z)]
    e: epsilon, default 0.00001
    """

    e1 = k[1] - k[0]
    e2 = k[2] - k[0]

    h = np.cross(d, e2)
    a = np.dot(e1, h)

    if ((a > -e) & (a < e)):
        return False

    f = 1/a
    s = p - k[0]

    u = f * np.dot(s,h)

    if((u < 0) | (u > 1)):
        return False

    q = np.cross(s, e1)
    v = f * np.dot(d,q)

    if ((v < 0) | (u+v > 1)):
        return False

    # At this stage we can compute t to find out where 
    # the intersection point is on the line

    t = f * np.dot(e2,q)

    return t > e

def intersect_all(p, d, all_tri, e=0.00001):
    """ p: xyz point; d: ray direction, default [0,0,1];
    k: Nx3x3 matrix of xyz coord of triangle: N*[(x,y,z), (x,y,z), (x,y,z)]
    e: epsilon, default 0.00001
    """

    sumx = np.sum((all_tri[:,:,0] >= p[0]), axis=1)
    sumy = np.sum((all_tri[:,:,1] >= p[1]), axis=1)
    #sumz = np.sum((all_tri[:,:,2] >= p[2]), axis=1)

    tri = all_tri[((sumx == 1) | (sumx == 2)) & ((sumy == 1) | (sumy == 2))]

    # Note: algorithm returns FALSE if point on vertex or edge of triangle

    e1 = tri[:,1] - tri[:,0]
    e2 = tri[:,2] - tri[:,0]

    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)

    a_valid = np.where((a < -e) | (a > e))
    #print(a)

    #f = 1/np.maximum(a,e/10)
    a[np.where(a==0)] = e/10
    #print(a)
    f=1/a
    s = p - tri[:,0]

    u = f * np.einsum('ij,ij->i', s, h)

    u_valid = np.where((u > 0) & (u < 1))

    q = np.cross(s, e1)
    v = f * np.dot(d, q.T)

    v_valid = np.where((v > 0) & (u+v < 1))

    # At this stage we can compute t to find out where 
    # the intersection point is on the line

    t = f * np.einsum('ij,ij->i', e2, q)

    valid_indices = np.intersect1d(np.intersect1d(a_valid, u_valid), v_valid)

    index_intersect = np.intersect1d(valid_indices, np.where(t > e))

    #print(index_intersect.shape[0])
    
    # If odd number of intersections, point within polygon and return True
    return bool(index_intersect.shape[0] % 2)






