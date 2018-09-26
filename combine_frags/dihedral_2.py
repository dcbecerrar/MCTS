import numpy as np
import random
import time

def dihedral(p):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def get_phi_psi_omega(array):
    """ Array of form 3N*1 where N is the number of atoms """
    #array=array.reshape(array.shape[0]/3, 3)
    #mask = np.ones(array.shape[0], dtype=bool)
    #mask[3::4] = 0

    filtered = array # contains only N, CA, C atoms, in that order

    # C - N - CA - C - N ==> phi: C-N-CA-C; psi: N-CA-C-N; omega: CA-C-N-CA

    # For residues with both phi and psi angles
    # start at first C atom (index 2) and end at last N atom (index -2)

    #total_residues=filtered.shape[0]/3 - 2

    phi_psi_omega_table = np.zeros([filtered.shape[0]/3, 3])
    
    # Order: phi, psi, omega
    # First residue missing phi and omega (assigned to next residue by convention)
    # Last residue missing psi

    psi_first = dihedral(filtered[0:4, :])
    phi_last = dihedral(filtered[-4:, :])
    omega_last = dihedral(filtered[-5:-1,:])
    
    phi_psi_omega_table[0, 1] = psi_first
    phi_psi_omega_table[-1, 0] = phi_last
    phi_psi_omega_table[-1, 2] = omega_last
    
    i=1
    residues=1
    
    while(i+4 <= filtered.shape[0]-2):
            omega = dihedral(filtered[i:i+4,:])
            phi = dihedral(filtered[i+1:i+5,:])
            psi = dihedral(filtered[i+2:i+6,:])

    #print(phi,psi,residues)
            phi_psi_omega_table[residues] = [phi,psi,omega]

            residues+=1
            i+=3
            
    return phi_psi_omega_table
    
def get_all_bond_length_angle(array):
    """ Array of form 3N * 1 where N is the number of atoms """
    #array=array.reshape(array.shape[0]/3, 3)
    #mask = np.ones(array.shape[0], dtype=bool)
    #mask[3::4] = 0

    filtered = array # contains only N, CA, C atoms, in that order
    
    atom1 = filtered[:-1, :]
    atom2 = filtered[1:, :]
    
    # Atom 2 always after Atom 1 in PDB
    bond_vector = atom2-atom1
    
    # First missing C-N bond
    all_bond_lengths = np.sqrt(np.sum(bond_vector ** 2, axis=1))
    
    # Normalize vectors for dot product calculations later
    bond_vector_normalized = bond_vector / all_bond_lengths[:,np.newaxis]
    
    # Dot products of two vectors of 3 consecutive atoms. First vector reversed so that
    # two vectors pointing out of central atom
    cosines = np.sum(-bond_vector_normalized[:-1] * bond_vector_normalized[1:],axis=1)
    
    # First missing C-N-CA angle, last missing CA-C-N angle
    all_bond_angles=np.degrees(np.apply_along_axis(np.arccos, axis=0, arr=cosines))
    
    # Order: C-N, N-CA, CA-C
    all_bond_lengths_table = np.zeros([filtered.shape[0]/3,3])
    all_bond_lengths_table[0,1] = all_bond_lengths[0]
    all_bond_lengths_table[0,2] = all_bond_lengths[1]
    all_bond_lengths_table[1:] = all_bond_lengths[2:].reshape(all_bond_lengths[2:].shape[0]/3, 3)
    
    # Order: C-N-CA, N-CA-C, CA-C-N
    all_bond_angles_table = np.zeros([filtered.shape[0]/3,3])
    all_bond_angles_table[0,1:3] = all_bond_angles[0:2]
    all_bond_angles_table[-1,0:2] = all_bond_angles[-2:]
    all_bond_angles_table[1:-1] = all_bond_angles[2:-2].reshape(all_bond_angles[2:-2].shape[0]/3, 3)
    
    return all_bond_lengths_table, all_bond_angles_table
    


def get_rama_score(phi_psi_omega_table, type):
    """ Input of format Nx2 where N is number of residues with phi and psi angles """

    # Normalize input to calculate score
    adjusted_phi_psi = ((phi_psi_omega_table+179) // 2).astype(int)

    # Use general rama score table for score calculations
    # Index is array[phi][psi]
    rama_score = np.load("./rama_tables/rama_{}.npy".format(type))

    scores = rama_score[adjusted_phi_psi[:,0], adjusted_phi_psi[:,1]]

    num_favored = np.sum(scores >= 0.02)
    if type=="general":
        num_allowed = np.sum(scores >= 0.0005)
        num_outliers = np.sum(scores < 0.0005)
    elif type=="glycine":
        num_allowed = np.sum(scores >= 0.001)
        num_outliers = np.sum(scores < 0.001)

    #print(num_favored, num_allowed, num_outliers)
    return (num_favored, num_allowed, num_outliers)


def score_loop(array, type="glycine"):
    """ Input:loop coordinates, output:number of rama favored, allowed, outliers"""

    phi_psi_omega_table = get_phi_psi_omega(array)
    return get_rama_score(phi_psi_omega_table, type)
	
def calc_next_position(p, d, a, l):
    """ Returns next atom position given positions of previous 3 atoms p,
    dihedral d, bond angle to new position a, and bond length l """
    
    v0 = p[0] - p[1]
    v1 = p[2] - p[1]
    
    # Normalize vectors
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    
    # Obtain portion of v0 perpendicular to v1
    v0_perp = v0 - np.dot(v0, v1)*v1
    #print(np.dot(v0_perp,v1))
    
    # Normalize v0_perp
    v0_perp /= np.linalg.norm(v0_perp)
    
    # Rotate along v1 with dihedral d given
    v0_rotated = np.cos(np.radians(d))*v0_perp + np.sin(np.radians(d))*np.cross(v1,v0_perp)
    #print(np.dot(v0_rotated, v1))
    #print(np.degrees(np.arccos(np.dot(v0_perp, v0_rotated))))
    
    # Bond angle a has to be > 90 to work
    if a < 90:
            raise ValueError("bond angle is less than 90")
    
    # Rotate perpendicular portion of v2 by a-90 degrees to get final vector
    v2_norm = np.cos(np.radians(a-90))*v0_rotated + np.sin(np.radians(a-90)) * v1
    #print(np.sum(v2_norm ** 2))
    
    return p[2] + l*v2_norm


def add_oxygens(array):
    ''' Adds oxygens to backbone from residue start to end '''
    d = 120.0       # CA-C-O bond angle
    l = 1.232       # length of carbonyl C=O bond


    # normalize vectors and rotate ca-c vector along cross product vector by 120 degres
    n = array[::3]
    ca = array[1::3]
    c = array[2::3]

    cac = ca - c
    cac = cac / np.linalg.norm(cac, axis=1)[:,None]

    nc = n - c
    nc = nc / np.linalg.norm(nc, axis=1)[:,None]

    perp = np.cross(nc, cac, axis=1)
    perp = perp / np.linalg.norm(perp,axis=1)[:,None]

    o_vec = np.cos(np.radians(d))*cac + np.sin(np.radians(d))*np.cross(perp,cac,axis=1)

    pred_o = c + l*o_vec

    return pred_o

def add_oxygens(array,start,end):
    ''' Adds oxygens to backbone from residue start to end '''
    d = 120.0       # CA-C-O bond angle
    l = 1.232       # length of carbonyl C=O bond


    # normalize vectors and rotate ca-c vector along cross product vector by 120 degres
    n = array[::3][start+1:end+1]
    ca = array[1::3][start:end]
    c = array[2::3][start:end]

    cac = ca - c
    cac = cac / np.linalg.norm(cac, axis=1)[:,None]

    nc = n - c
    nc = nc / np.linalg.norm(nc, axis=1)[:,None]

    perp = np.cross(nc, cac, axis=1)
    perp = perp / np.linalg.norm(perp,axis=1)[:,None]

    o_vec = np.cos(np.radians(d))*cac + np.sin(np.radians(d))*np.cross(perp,cac,axis=1)

    pred_o = c + l*o_vec

    return pred_o

    
