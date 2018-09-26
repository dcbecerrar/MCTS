import numpy as np
import random
import dihedral_2 as d
import pdb_H3_generator as p
from sklearn.neighbors import NearestNeighbors as NN
import time
from Bio.PDB import PDBParser

# Note: function below is basically a part of function calc_value in mcts_frag.py

def build_frag(array, moves, begin, end):

    # array is with oxygen, original does not

    mask=np.ones(array.shape[0]).astype(bool)
    mask[3::4] = 0
    original=array[mask]
    size = len(moves)/3



    dih = np.zeros([(end-begin)/3 + 1, 3])
    angle = np.zeros([(end-begin)/3 + 1, 3])
    length = np.zeros([(end-begin)/3, 3])    # Note length of this is 1 row shorter

    angle[:,0] = 121.7
    angle[:,2] = 116.2
    angle[0,0] = 0
    angle[-1,-1] = 0

    length[:,0] = 1.329
    length[:,1] = 1.458
    length[:,2] = 1.525
    #length[0,0] = 0

    dih[:,0] = 180
    dih[0,0] = 0
    #dih[0,1] = 0
    #dih[-1,2] = 0


    phi_mask = np.zeros(len(moves), dtype=bool)
    phi_mask[2::3] = 1

    psi_mask = np.zeros(len(moves), dtype=bool)
    psi_mask[0::3] = 1

    tau_mask = np.zeros(len(moves), dtype=bool)
    tau_mask[1::3] = 1
    #print(start, size) 

    dih[1:,1] = np.array(moves)[phi_mask]
    dih[:-1,2] = np.array(moves)[psi_mask]
    angle[1:,1] = np.array(moves)[tau_mask]


    #angle[:,1] = actual_angle[:,1]
    dih = dih[dih!=0].flatten()
    length = length[length!=0].flatten()
    angle = angle[angle!=0].flatten()

    constructed = np.zeros([(end-begin) + 3, 3])    # +3 accounts for 2 fixed residues at end
    constructed[0:3] = original[begin:begin+3]

    i = 3

    p = constructed[0:3]

    while(i < constructed.shape[0]):
        curr_d = dih[i-3]
        curr_a = angle[i-3]
        curr_l = length[i-3]

        next_position = d.calc_next_position(p, curr_d, curr_a, curr_l)
        constructed[i] = next_position

        i+=1
        p = constructed[i-3:i]


    #print(constructed)
    return constructed


def recurse_build(original, curr, frag_dict, midpoints, curr_level):
    if curr_level == len(midpoints)-1:
        final = np.zeros([original.shape[0],3])

        mask=np.ones(final.shape[0]).astype(bool)
        mask[3::4] = 0

        final[mask] = curr

        #adds oxygen to every residue except last
        all_oxy = np.zeros([final.shape[0]/4,3])
        all_oxy[:-4] = d.add_oxygens(curr,0,(curr.shape[0]/3)-4)
        all_oxy[-4:] = original[np.invert(mask)][-4:]

        final[np.invert(mask)] = all_oxy

        #print(final)

        score = (calc_internal_clashing(final))
        #print(score)
        if score == 0:
            with open("./5ggs_relaxed_H3_template_gly.pdb", "rt") as template:
                SIDE_TEMP = template.readlines()
            name = "5ggs_relaxed_H3_frag_phe_combo_run1_{}.pdb".format(random.getrandbits(32))
            p.make_pdb(final[16:], "./frag_combos/"+name, SIDE_TEMP)
            squared_diff =((original[mask] - curr)[4*3:-4*3]**2)/float(curr[4*3:-4*3].shape[0])
            
            # Below is only relevant if you're using same midpoint as native
            RMSDs.write(str(np.sqrt(np.sum(squared_diff))) + " " + name + "\n")

    else:
        combos = frag_dict[curr_level]
        num_combos = combos.shape[0]
        for i in range(num_combos):
            temp=curr[:]
            temp[midpoints[curr_level]+3:midpoints[curr_level+1]] = combos[i][3:-3]
        
            recurse_build(original, temp, frag_dict, midpoints, curr_level+1)



def build_whole(original, midpoints):
    ''' midpoints is a list of midpoints used in construction, including startpoint and endpoint '''
    # midpoints for 5ggs relaxed phe: 12, 27, 48 

    frag_dict = dict()

    # assumes file is well formatted, with num lines as first and contents later

    for i in range(len(midpoints)-1):
        f=open("./5ggs_relaxed_frag{}_phe_all_possible_run1.txt".format(i+1))
        size=int(f.readline().strip())
        frag = np.zeros((size, (midpoints[i+1]-midpoints[i])+3, 3))
        for j in range(size):
            line=f.readline()
            nums=line.strip().split("[")[1].strip("]").split(",")
            moves = np.array(nums, dtype=int)
            frag[j]=build_frag(original, moves, midpoints[i], midpoints[i+1])
        frag_dict[i] = frag

    # Cheat if you have original structure. If you don't then raw would be zero matrix
    # with midpoint and pre and post H3 coordinates filled (YYCAR and WGQGT)
    mask=np.ones(original.shape[0]).astype(bool)
    mask[3::4] = 0

    raw = original[mask]

    recurse_build(original, raw, frag_dict, midpoints, 0)

    print("ALL DONE!")



def calc_internal_clashing(array):
    ''' calculates number of nearest clashing atoms in backbone or jumps using knn
    with k=5. Array order is N,CA,C,O. '''

    #num_residues = array.shape[0]/4
    cutoff = 2      # the cutoff determining clashing

    nbrs = NN(n_neighbors=5, algorithm='kd_tree').fit(array)
    dist, ind = nbrs.kneighbors(array)
    n_clash = np.sum(dist[::4,3] < cutoff) + np.sum(dist[::4,:3][1:-1] >= cutoff)
    ca_clash = np.sum(dist[1::4,3] < cutoff) + np.sum(dist[1::4,:3][1:-1] >= cutoff)
    c_clash = np.sum(dist[2::4,4] < cutoff) + np.sum(dist[2::4,:4][1:-1] >= cutoff)
    o_clash = np.sum(dist[3::4,2] < cutoff) + np.sum(dist[3::4,:2][1:-1] >= cutoff)

    return n_clash+ca_clash+c_clash+o_clash

if __name__ == "__main__":
    #SEQUENCE: YYCARRDYRFDMGFDYWGQGT
    #INDEX:    012345678901234567890
    RMSDs = open("./5ggs_relaxed_constructs_frag_phe_disres_rmsds_run1.txt",'w')
    # If no starting structure, original is zero matrix with midpoint, pre, and post
    # H3 coords filled in (with oxygen)
    original = np.load("../mcts/5ggs_original_H3_relaxed.npy")
    midpoints = [12, 27, 48]    #ARG 4, PHE 9, TRP 16, zero numbering

    parser = PDBParser()
    disembodied_residue = parser.get_structure("disres", "../mcts/original_relaxed_docking_phe_0939.pdb")
    MID = np.zeros([4,3])
    count=0

    for atom in disembodied_residue.get_atoms():
        if atom.get_id() in ['N', 'CA', 'C', 'O']:
                MID[count] = atom.get_coord()
                count+=1

    original[4*(midpoints[1]/3):4*(midpoints[1]/3)+4] = MID
    print(original.shape)

    #build_whole(original, midpoints)
    RMSDs.close()
