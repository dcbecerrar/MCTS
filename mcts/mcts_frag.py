#!/usr/bin/env python
import random
import math
import hashlib
import logging
import argparse
import numpy as np
import dihedral_2 as d
import intersect_2 as intersect
from sklearn.neighbors import NearestNeighbors as NN
from multiprocessing.pool import Pool
import multiprocessing
import os, sys
from Bio.PDB import PDBParser

# import backbone_trie as bt
# import pickle
import time


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf

The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.

The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  

In particular there are two models of best child that one can use 
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

original = np.load("./5ggs_original_H3_relaxed.npy")
mask = np.ones(original.shape[0],dtype=bool)
mask[3::4]=0

TARGET = original[mask]

FIRST_RES = 4
LAST_RES = 16

FIXED_RES = 9


START = TARGET[FIRST_RES*3:FIRST_RES*3+3]    # Arginine 4, zero numbered
END = TARGET[LAST_RES*3:LAST_RES*3+3]   # Tryptophan 16


# Choose between disembodied residue or original midpoint below:

#Original Coord
#MID = TARGET[FIXED_RES*3:FIXED_RES*3+3]     # Phenylalanine 8 


#Disembodied Residue Coord
parser = PDBParser()
disembodied_residue = parser.get_structure("disres", "./original_relaxed_docking_phe_0939.pdb")

MID = np.zeros([3,3])
count=0

for atom in disembodied_residue.get_atoms():
    if atom.get_id() in ['N', 'CA', 'C']:
        MID[count] = atom.get_coord()
        count+=1

#print(MID)

# The surface mesh for external collision
MESH = np.load("./5ggs_relaxed_2a_surface_mesh.npy")


# A dictionary of favorable ramachandran angles from Molprobity 8000
GEN_FAV = np.load("rama_general_favored.npy").item()


# Adjust params below to build different fragments

START_RES = FIRST_RES
END_RES = LAST_RES


# Note: START_PHI is used to determine possible values for first psi angle.
# Obtainable if fragment uilds onwards from existing structure (START is START)
# 0 if building from disembodied residue (START is MID)


START_PHI = -115


class State():
    NUM_TURNS = 3*(END_RES-START_RES)
    GOAL = 0
    TAU=[105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    FAV_PHI = GEN_FAV.keys()
    MAX_VALUE= 50

    def __init__(self, value=MAX_VALUE, moves=[], turn=NUM_TURNS, vary="psi"):
        self.value=value
        self.turn=turn
        self.moves=moves
        self.vary = vary

        if self.vary=="phi":
            self.MOVES = self.FAV_PHI
        elif self.vary=="tau":
            self.MOVES = self.TAU
        elif self.vary=="psi":
            if len(self.moves) == 0:
                if START_PHI == 0:
                    self.MOVES = list(set(val for key in GEN_FAV for val in GEN_FAV[key]))
                else:
                    self.MOVES = GEN_FAV[START_PHI]
            else:
                self.MOVES = GEN_FAV[self.moves[-1]]
        else:
            raise ValueError("input vary not a valid state")

        self.num_moves = len(self.MOVES)

    def next_state(self):
        nextmove=random.choice([x for x  in self.MOVES])
        
        if self.vary=="phi":
            next=State(self.MAX_VALUE, self.moves+[nextmove],self.turn-1, "psi")
        elif self.vary=="psi":
            next=State(self.MAX_VALUE, self.moves+[nextmove],self.turn-1, "tau")
        elif self.vary=="tau":
            next=State(self.MAX_VALUE, self.moves+[nextmove],self.turn-1, "phi")

        return next
    def terminal(self):
        if self.turn == 0:
            return True
        return False
    def reward(self):
        self.value = self.calc_value()
        r = 1.0-(self.value-self.GOAL)/self.MAX_VALUE
        return r

    def construct_backbone(self, constructed_size):
        np.random.seed()

        dih = np.zeros([(END_RES - START_RES+1), 3])
        angle = np.zeros([(END_RES - START_RES+1), 3])
        length = np.zeros([END_RES - START_RES, 3])   # Note length of this is 1 row shorter
        
        # Fixed dihedral angles, bond angles, and bond lengths used for loop building

        # Order: C-N-CA, N-CA-C (tau), CA-C-N
        angle[:,0] = 121.7
        angle[:,2] = 116.2
        angle[0,0] = 0
        angle[-1,-1] = 0

        # Order: N-CA, CA-C, C-N 
        length[:,0] = 1.329
        length[:,1] = 1.525
        length[:,2] = 1.458
        #length[0,0] = 0

        # Column 0 is omega, column 1 is phi, column 2 is psi
        dih[:,0] = 180
        dih[0,0] = 0
        #dih[0,1] = 0
        #dih[-1,2] = 0
        
        
        phi_mask = np.zeros(len(self.moves), dtype=bool)
        phi_mask[2::3] = 1

        psi_mask = np.zeros(len(self.moves), dtype=bool)
        psi_mask[0::3] = 1

        tau_mask = np.zeros(len(self.moves), dtype=bool)
        tau_mask[1::3] = 1

        #print(self.moves, self.vary)

        curr_phi = np.array(self.moves)[phi_mask]
        curr_psi = np.array(self.moves)[psi_mask]
        curr_tau = np.array(self.moves)[tau_mask]
        
        #print(curr_tau, curr_phi, curr_psi)

        rand_phi = np.random.choice(self.FAV_PHI, (self.NUM_TURNS/3) - len(curr_phi))   
    
        dih[1:,1] = np.concatenate((np.array(curr_phi), rand_phi))

        # last phi missing corresponding psi angle; fill it in
        if(len(curr_phi) > (len(curr_psi)-1)):
            if len(rand_phi) != 0:
                miss_psi = np.random.choice(GEN_FAV[curr_phi[-1]])

                dih[:-1,2] = np.concatenate((np.array(curr_psi), [miss_psi], \
                        [np.random.choice(GEN_FAV[x]) for x in rand_phi[:-1]]))
            else:
                dih[:-1,2] = np.array(curr_psi)
        else:
            dih[:-1,2] = np.concatenate((np.array(curr_psi), \
                    [np.random.choice(GEN_FAV[x]) for x in rand_phi[:-1]]))

        angle[1:,1] = np.concatenate((np.array(curr_tau), np.random.choice(self.TAU, (self.NUM_TURNS/3) - len(curr_tau))))

        angle = angle[angle!=0].flatten()
        dih = dih[dih!=0].flatten()
        length = length[length!=0].flatten()

        #print(angle, dih, length)

        constructed = np.zeros([constructed_size,3])
        constructed[0:3] = START

        i = 3

        p = START
        mid_coords, end_coords = p, p

        while(i < (constructed_size)):
            curr_d = dih[i-3]
            curr_a = angle[i-3]
            curr_l = length[i-3]

            next_position = d.calc_next_position(p, curr_d, curr_a, curr_l)

            constructed[i] = next_position

            i+=1
            p = constructed[i-3:i]

            if i == 3*(FIXED_RES - START_RES) + 3:
                mid_coords = p
            elif i == constructed_size - 1:
                end_coords = p

        return constructed, mid_coords, end_coords

    def calc_value(self):
        constructed_size = 3*(END_RES - START_RES + 1)   # Build fragment from start fixed to end fixed

        constructed, mid_coords, end_coords = self.construct_backbone(constructed_size)

        con_oxy = np.zeros([constructed_size*4/3,3])
        oxy_mask = np.zeros(con_oxy.shape[0],dtype=bool)
        oxy_mask[3::4] = 1

        con_oxy[np.invert(oxy_mask)] = constructed

        #combined = np.zeros([TARGET.shape[0],3])
        #combined[:constructed_size] = constructed
        #combined[constructed_size:] = TARGET[constructed_size:]

        #all_oxy = d.add_oxygens(con_oxy,0,constructed_size/3)
        
        #con_oxy[oxy_mask] = all_oxy
        
        mesh_penalty = 0
        
        #num_penalty = 0
        #num_penalty = min(3, self.NUM_TURNS - len(self.moves)) + 1     #with oxygen
        num_penalty = min(2, self.NUM_TURNS - len(self.moves)) + 1      # no oxygen

        # can choose to penalize oxygens or not


        '''
        for j in range(num_penalty):
            start_index = len(self.moves) -1 + 4 + j
            if (intersect.intersect_all(con_oxy[start_index], [0,0,-1], MESH)):
                mesh_penalty += self.MAX_VALUE/10
        '''
        
        for j in range(num_penalty):
            start_index = len(self.moves) - 1 + 3 + j
            if (intersect.intersect_all(constructed[start_index], [0,0,-1], MESH)):
                mesh_penalty += self.MAX_VALUE/10
        
        
        num_clash = self.calc_internal_clashing(constructed)
        #num_clash = 0
        clash_penalty = num_clash * self.MAX_VALUE/10

        mid_score = np.sqrt(np.sum((mid_coords - MID) ** 2)/3.0)
        end_score = np.sqrt(np.sum((end_coords - END) ** 2)/3.0)

        num_of_moves = float(len(self.moves))
        len_first_frag = (FIXED_RES - FIRST_RES)*3
        len_whole = (LAST_RES - FIRST_RES)*3

        if num_of_moves <= 15:
            return (num_of_moves/len_first_frag)*mid_score + mesh_penalty + clash_penalty
        else:
            return mid_score + (num_of_moves/len_whole)*end_score + mesh_penalty + clash_penalty


    def calc_internal_clashing(self, array):
        ''' calculates number of nearest clashing atoms in backbone using knn
         with k=5. Array order is N,CA,C,O. '''

        #num_residues = array.shape[0]/4
        cutoff = 2  # the cutoff determining clashing

        nbrs = NN(n_neighbors=5, algorithm='kd_tree').fit(array)
        dist, ind = nbrs.kneighbors(array)


        #Note: clashing no-oxygen backbone

        n_clash = np.sum(dist[::3,3] < cutoff)
        ca_clash = np.sum(dist[1::3,3] < cutoff)
        c_clash = np.sum(dist[2::3,3] < cutoff)
        o_clash = 0
        #o_clash = np.sum(dist[3::4,2] < cutoff)

        return n_clash+ca_clash+c_clash+o_clash


    def __hash__(self):
        return int(hashlib.md5(str(self.moves)).hexdigest(),16)
    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False
    def __repr__(self):
        s="Value: %.5f; Moves: %s"%(self.value,self.moves)
        return s
    

class Node():
    def __init__(self, state, parent=None):
        self.visits=1
        self.reward=0.0 
        self.state=state
        self.children=[]
        self.parent=parent  
    def add_child(self,child_state):
        child=Node(child_state,self)
        self.children.append(child)
    def update(self,reward):
        self.reward+=reward
        self.visits+=1
    def fully_expanded(self):
        if len(self.children)==self.state.num_moves:
            return True
        return False
    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
        return s
        

def UCTSEARCH(budget,root):
    for iter in range(budget):
        if iter%10000==9999:
            logger.info("simulation: %d"%iter)
            logger.info(root)
        front=TREEPOLICY(root)
            
        reward, value=DEFAULTPOLICY(front.state)        

        BACKUP(front,reward, value)
    return BESTCHILD(root,0)

def TREEPOLICY(node):
    while node.state.terminal()==False:
        if node.fully_expanded()==False:    
            return EXPAND(node)
        else:
            node=BESTCHILD(node,SCALAR)
    return node

def EXPAND(node):
    tried_children=[c.state for c in node.children]
    new_state=node.state.next_state()
    while new_state in tried_children:
        new_state=node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]


#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
    bestscore=0.0
    bestchildren=[]
    for c in node.children:

        exploit = c.reward
        explore=math.sqrt(math.log(node.visits)/float(c.visits))  
        score=exploit+scalar*explore
        # print(exploit, scalar, explore, score)
        if score==bestscore:
            bestchildren.append(c)
        if score>bestscore:
            bestchildren=[c]
            bestscore=score
    if len(bestchildren)==0:
        logger.warn("OOPS: no best child found, probably fatal")
        #print("no non-zero reward children, going to parent node")
    #print(bestscore)
    return random.choice(bestchildren)

def DEFAULTPOLICY(state):
    reward=0
    value=50

    for i in range(MAXRAND):
        curr_reward, curr_value= state.reward(), state.value
        if curr_reward > reward:
            reward=curr_reward
            value=curr_value
    
    return reward, value

def BACKUP(node,reward,value):
    while node!=None:
        node.visits+=1
        #node.reward+=reward
        node.reward = max(node.reward, reward)
        node.state.value = min(node.state.value,value)
        node=node.parent
    return

def MCTS(args):

    global MAXRAND
    (num_sims, levels, MAXRAND, walltime) = args

    root = Node(State())

    t_end = time.time() + walltime
    while time.time() < t_end:
        for l in range(levels):
            UCTSEARCH(num_sims*(root.state.num_moves), root)

    current_node = root
    while current_node.state.terminal()==False:
        if len(current_node.children) > 0:
            current_node = BESTCHILD(current_node,0)
        else:
            break

    return [current_node.state.value, current_node.state.moves]


# def create_trie(mcts_outfile, epsilon):
#     backbones = TrieNode([10.07199955, -10.81700039, -17.52499962], epsilon)
#     family_heads = []

#     rosetta_scores = bt.parse_rosetta_scores("score.sc")
#     angles = bt.parse_mcts_output(mcts_outfile)

#     assert len(rosetta_scores) == len(angles)

#     for rank in angles:
#         assert rank in rosetta_scores

#     for rank in angles:
#         moves = angles[rank]

#         test_state = State(50, moves, 0, "psi")

#         backbone = test_state.construct_backbone(len(moves) + 3)[0]

#         if bt.add_backbone(backbones, backbone, epsilon, rosetta_scores[rank]):
#             family_heads.append(rank)

#     print(len(family_heads))

#     outfile = open("test_trie.ds", "w")
#     pickle.dump(backbones, outfile)
#     outfile.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MCTS research code')
    parser.add_argument('--num_sims', action="store", required=True, type=int)
    parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS+1))
    parser.add_argument('--num_rand', action="store", required=True, type=int)
    parser.add_argument('--walltime', action="store", required=True, type=int)

    args=parser.parse_args()

    MAXRAND = args.num_rand
    
    # SLURM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    
    # cpus = SLURM_CPUS

    # all_iter = [[args.num_sims, args.levels, MAXRAND] for i in range(cpus)]

    # pool = Pool(cpus)
    # result = list(pool.map(MCTS, all_iter))
    # pool.close()
    # pool.join()
    # print result
    
    root = Node(State())

    t_end = time.time() + args.walltime

    final_moves = []

    while time.time() < t_end:
        current_node = root
        for l in range(args.levels):
            current_node = UCTSEARCH(args.num_sims*(current_node.state.num_moves), current_node)
        
            print("level %d"%l)
            print("Num Children: %d"%len(current_node.parent.children))
            # for c in (current_node.parent.children):
            #     print(c.state.moves[-1],c)
            print("Best Child: %s"%current_node.state)

            print("--------------------------------")   

    current_node = root
    while current_node.state.terminal()==False:
        if len(current_node.children) > 0:
            current_node = BESTCHILD(current_node,0)
        else:
            break
    
    print(current_node.state)  
