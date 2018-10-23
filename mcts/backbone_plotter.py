import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pyRMSD.RMSDCalculator
import pyRMSD.condensedMatrix

from dihedral_2 import calc_next_position
from mcts_frag import State

def parse_tinker_int_file(filename):
    dihedral_angles, bond_angles, bond_lengths = [], [], []

    with open(filename, 'r') as infile:
        for i in range(4):
            next(infile)

        for line in infile:
            if line != "\n":
                line = line.split(" ")
                bond_lengths.append(float(line[4]))
                bond_angles.append(float(line[6]))
                dihedral_angles.append(float(line[8]))

    return dihedral_angles, bond_angles, bond_lengths

def recreate_backbone_from_int(filename):
    dihedral_angles, bond_angles, bond_lengths = parse_tinker_int_file(filename)

    coordinates = [
        np.array([10.07199955, -10.81700039, -17.52499962]),
        np.array([10.97599983, -11.62199974, -18.32799911]),
        np.array([12.39700031, -11.53999996, -17.77300072])]

    for i in range(len(bond_lengths)):
        print(dihedral_angles[i], bond_angles[i], bond_lengths[i])
        previous = np.array([coordinates[i], coordinates[i + 1], coordinates[i + 2]])
        
        coordinates.append(np.array(calc_next_position(previous, dihedral_angles[i], bond_angles[i], bond_lengths[i])))

    return np.array(coordinates)

def parse_tinker_xyz_file(filename='5ggs_h3_whole_loop_from_relaxed_npy.xyz'):
    coordinates = []

    with open(filename, 'r') as infile:
        lines = [line for line in infile]

        for line in lines[1:]:
            line = [float(c) for c in [c for c in line.split(' ') if c][2:5]]
            coordinates.append(line)

    return np.array(coordinates)

def parse_pdb_xyz_file(filename='5ggs_original.xyz'):
    coordinates = []

    with open(filename, 'r') as infile:
        for line in infile:
            line = [c for c in line.split(' ') if c]

            if line[0] == 'ATOM' and line[2] in ('C', 'CA', 'N'):
                line = [float(c) for c in line[6:9]]
                coordinates.append(line)

    return np.array(coordinates)

def superposition(coordset):
    calculator = pyRMSD.RMSDCalculator.RMSDCalculator("QCP_SERIAL_CALCULATOR", coordset)
    rmsd = calculator.pairwiseRMSDMatrix()
    print(rmsd)
    rmsd_matrix = pyRMSD.condensedMatrix.CondensedMatrix(rmsd).get_data()

    return coordset, rmsd_matrix

def array_of_points_to_arrays_of_points(array_of_points):
    x, y, z = [], [], []

    for point in array_of_points:
        print(point)
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])

    return x, y, z

def plot_and_save_backbone_figs(filename):
    original = np.load("./5ggs_original_H3_relaxed.npy")
    mask = np.ones(original.shape[0],dtype=bool)
    mask[3::4]=0

    TARGET = original[mask]

    with open(filename, "r") as infile:
        line_counter = 1
        coords = []
        for line in infile:
            # print(line)
            moves = eval("[" + line.split(", [")[1])

            # print(moves)

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            test_state = State(50, moves, 0, "psi")
        
            x, y, z = array_of_points_to_arrays_of_points(test_state.construct_backbone(len(moves) + 3)[0])

            ax.plot(x, y, z, label=str(line_counter))

            x, y, z = array_of_points_to_arrays_of_points(TARGET[12:48])

            ax.plot(x, y, z, label='TARGET')

            x, y, z = array_of_points_to_arrays_of_points(parse_pdb_xyz_file("xyz/5ggs_original.xyz"))

            ax.plot(x, y, z, label='pdb')

            plt.title(str(line_counter))
            line_counter+=1

            ax.legend()

            plt.show()



if __name__=="__main__":
    mpl.rcParams['legend.fontsize'] = 10

    # plot_and_save_backbone_figs("scores_below_one.txt")
    plot_and_save_backbone_figs("5ggs_relaxed_phe_whole_all_more_iters.txt")
    # plot_and_save_backbone_figs("5ggs_relaxed_phe_whole_all_weighted.txt")

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    #######################################

    # moves = [137, 107, -143, 119, 108, -57, -17, 110, -67, -25, 111, -79, 147, 107, -167]
    # moves_set = [[159, 107, -143, 137, 108, -57, 119, 110, -67, -17, 111, -79, -25, 107, -167, 147, 110, -71, 95, 109, -115, 35, 114, 79, 161, 106, -85, 93, 113, -85, 5, 109, -161]]
    # moves = [160, 115, -145, 130, 110, -50, 120, 120, -60, -17, 111, -79, -25, 107, -167]
    # moves_set = [[121, 109, 77, 7, 113, -71, 103, 113, 79, 5, 110, -135, -5, 110, -171, 173, 111, -53, -23, 113, 39, 55, 111, -59, -17, 115, -133, -5, 110, -77, 77, 106, -135, 51, 106, -57]
    # , [-51, 114, -47, 129, 108, -105, -53, 112, -111, -55, 111, -135, 29, 106, 71, 31, 109, -155, -171, 105, 49, -129, 107, -69, -17, 108, -83, 165, 114, -141, -171, 112, -77, -179, 107, -129]
    # , [31, 110, -119, -177, 114, 71, -1, 108, -135, -5, 113, -95, 73, 106, 69, 7, 108, 67, 3, 105, -91, -43, 115, -97, 93, 108, -115, 89, 114, -133, 147, 109, -125, -171, 110, -177]
    # , [-37, 114, -81, 69, 107, -103, -31, 111, -141, 87, 109, -55, -23, 105, -177, 171, 113, 67, 37, 114, 71, 33, 108, -99, 151, 115, -139, 153, 108, -121, 161, 110, -163, 123, 113, 41]
    # , [163, 113, -63, 143, 107, -89, 91, 113, 77, 1, 111, 63, 51, 111, -157, 149, 113, 43, 45, 111, 69, 21, 110, -59, 127, 114, -153, 115, 110, -67, 125, 114, -91, 119, 115, 67]
    # , [-175, 114, -165, -175, 107, -87, -169, 115, -77, 63, 106, 73, 5, 107, -67, 161, 110, -53, -39, 105, -85, 125, 109, -145, 95, 114, -75, 113, 110, -133, 17, 112, -103, 87, 113, -99]]
    # moves_set = [[-7, 114, -127, 1, 105, -51, -53, 109, -125, 89, 114, -119, 179, 105, -55, -27, 113, -77, 3, 105, 77, 13, 115, -79, 57, 110, -69, 135, 109, 51, -131, 108, 43, 61, 113, -173],
    # [-3, 105, -71, 141, 112, 79, 7, 112, -71, -47, 109, -141, 87, 112, 79, -3, 110, -93, -3, 111, 51, -123, 109, -49, 123, 105, -61, 153, 113, -159, 165, 115, 73, 27, 113, -79],
    # [119, 107, -45, 135, 113, -45, -59, 110, -137, 7, 108, 75, -3, 107, -133, 129, 111, 67, 9, 115, -165, 135, 110, -131, -3, 107, -105, 139, 108, -153, 159, 109, -63, 149, 108, -167],
    # [-175, 114, 63, 7, 111, -67, -13, 110, -147, 157, 115, 71, 29, 107, -163, -177, 112, -87, -25, 113, -133, 143, 109, -83, 159, 107, -73, 137, 106, -77, -61, 113, -139, 165, 112, 57],
    # [-45, 106, -121, -37, 105, -45, -59, 105, -45, -59, 111, -177, 171, 113, -161, 157, 106, -45, -45, 108, -107, -45, 115, -131, 123, 114, -57, -27, 112, -103, 121, 106, -135, -9, 109, -127],
    # [173, 113, -89, 71, 113, -77, 67, 115, -175, 173, 111, -49, -61, 106, -167, 159, 112, 47, 47, 109, -147, -177, 110, -129, 177, 113, -125, 43, 112, 79, 3, 106, 39, 53, 110, -57]]
    # moves_set = [
    #     [29, 110, -73, -177, 114, 51, 39, 110, -105, -55, 105, -83, 63, 105, 53, 59, 108, 71, 29, 108, -147, 163, 110, 79, -3, 113, -55, 133, 105, -105, 99, 105, -69, 143, 108, -55],
    # [165, 113, -123, 121, 114, -109, 171, 110, 79, -1, 113, 51, 29, 106, -173, 153, 108, 41, 45, 113, -51, 147, 113, -83, -173, 110, -145, 159, 112, 51, 35, 109, -65, -7, 105, -97],
    # [33, 110, -75, -175, 105, 73, -5, 106, -67, 163, 109, 39, 55, 114, 53, 51, 105, 69, 15, 105, -83, -43, 105, -103, 95, 107, -101, 9, 110, -41, 127, 111, -117, -175, 114, -123],
    # [149, 115, -45, -37, 114, 61, 9, 110, -165, 169, 111, -87, 3, 111, -177, 165, 112, 75, 7, 113, 41, 51, 105, -125, 121, 109, -167, 161, 109, -41, 129, 106, -77, -173, 109, -67],
    # [149, 115, -45, -41, 111, 69, 5, 105, -169, 173, 109, -93, -11, 112, -151, 117, 105, 73, 1, 109, -105, 3, 108, -49, -41, 107, -73, 137, 106, -145, 117, 115, -55, 121, 106, -175],
    # [99, 115, -139, 41, 115, -123, -31, 109, -55, -33, 115, -121, 131, 112, -43, 123, 108, 73, 7, 109, -43, 135, 110, -63, 135, 110, -95, 165, 114, 47, 45, 115, -71, 5, 108, -79],
    # [-179, 107, -59, -59, 108, 77, -1, 115, -171, -179, 110, -89, 125, 111, 39, 55, 107, -143, -171, 109, -143, 145, 115, 79, -3, 105, -51, -49, 113, -109, 93, 110, -91, 143, 106, -45],
    # [5, 115, -55, 159, 114, 45, 49, 109, -83, -57, 114, -79, -61, 113, -137, 1, 109, -109, 163, 109, -77, -37, 110, 43, 57, 114, -45, -53, 108, -155, 121, 111, -107, -3, 110, -141],
    # [115, 105, -163, -177, 108, 73, 7, 109, -99, -35, 113, -109, 39, 108, 59, 33, 106, -161, 149, 106, -105, 177, 106, 61, 33, 109, -93, 117, 109, 55, 51, 110, -71, 165, 107, -109],
    # [121, 108, -157, -175, 111, 73, 9, 105, -101, -17, 109, -123, 43, 105, 43, 53, 109, 79, -5, 105, -103, 35, 108, -67, -31, 106, 43, 59, 112, -153, 133, 115, -159, 165, 114, -167],
    # [159, 107, -143, 137, 108, -57, 119, 110, -67, -17, 111, -79, -25, 107, -167, 147, 110, -71, 95, 109, -115, 35, 114, 79, 161, 106, -85, 93, 113, -85, 5, 109, -161],
    # [155, 109, -77, 123, 107, -45, -49, 108, -147, -167, 109, -81, 1, 113, -131, 17, 110, -105, 161, 105, -63, 127, 107, -81, 101, 108, -145, 163, 106, -65, 123, 108, -99, 13, 111, 57]]
#     moves_set = [[157, 106, -65, -49, 106, 71, -1, 110, -157, 147, 115, -73, -7, 113, -149, -171, 112, 53, -129, 115, -167, -173, 105, 49, 31, 110, -79, -1, 113, -175, 157, 112, -165, 139, 105, -47]
# , [31, 109, -67, 165, 111, 65, 41, 111, -101, -11, 106, -135, 49, 105, 67, 3, 105, -127, 161, 109, -105, -177, 110, 57, 19, 106, -63, -45, 105, -123, 95, 108, -89, 135, 114, -173]]
#     for moves in moves_set:
#         test_state = State(50, moves, 0, "psi")
        
#         x, y, z = array_of_points_to_arrays_of_points(test_state.construct_backbone(len(moves) + 3)[0])

#         ax.plot(x, y, z, label='library_function')

    #     print(len(x))

    # ########################################

    # x4, y4, z4 = array_of_points_to_arrays_of_points(recreate_backbone_from_int("xyz/5ggs_h3_whole_loop_from_relaxed_npy.int"))
    # ax.plot(x4, y4, z4, label='library_int')

    # print(len(x4))

    # ########################################

    # moves = [115, 105, -149, -165, 114, 55, 27, 105, -123, -15, 106, -127, 47, 107, 41]
    # moves = [159, 110, -115, 137, 107, -143, 119, 108, -57, -17, 110, -67, -25, 111, -79, 147, 107, -167, 110, -71, 109, -115, 161, 114, 79, 106, -85, 5, 113, -85]
    # moves = [159, 107, -143, 137, 108, -57, 119, 110, -67, -17, 111, -79, -25, 107, -167, 147, 110, -71, 95, 109, -115, 35, 114, 79, 161, 106, -85, 93, 113, -85, 5, 109, -161]
    # moves = [159, 107, -143, 137, 108, -57, 119, 110, -67, -17, 111, -79, -25, 107, -167]   
    # moves = [125, 107, -129, 125, 108, -57, 119, 110, -67, -17, 111, -79, -25, 107, -167]
    # test_state = State(50, moves, 0, "psi")

    # # print(test_state.calc_value())
    
    # x5, y5, z5 = array_of_points_to_arrays_of_points(test_state.construct_backbone(len(moves) + 3)[0])

    # ax.plot(x5, y5, z5, label='high_score_function')

    # print(len(x5))

    # ########################################

    # x6, y6, z6 = array_of_points_to_arrays_of_points(recreate_backbone_from_int("xyz/5ggs_h3_loop_frag1_highest_score.int"))
    # ax.plot(x6, y6, z6, label='high_score_int')

    ##########################################

    # filename = '5ggs_relaxed_frag1_phe_all_possible_run1.txt'
    # counter = 1
    # coordset = []
    # with open(filename, 'r') as infile:
    #     for line in infile:
    #         line = line.split(", [")
    #         if len(line) > 1 and counter < 25:
    #             moves = eval('[' + line[1].rstrip('\n'))

    #             test_state = State(50, moves, 0, "psi")

    #             coord = test_state.construct_backbone(len(moves) + 3)[0]
                
    #             x, y, z = array_of_points_to_arrays_of_points(coord)

    #             ax.plot(x, y, z, label=str(counter))
    #             counter+=1

    #             coordset.append(coord)

    # rmsd_matrix = superposition(np.array(coordset))[1]
    # print(rmsd_matrix)

    ##########################################

    # x2, y2, z2 = array_of_points_to_arrays_of_points(parse_pdb_xyz_file("xyz/5ggs_relaxed_original_full.xyz"))

    # ax.plot(x2, y2, z2, label='pdb')

    # print(len(x2))

    #####################################

    # original = np.load("./5ggs_original_H3_relaxed.npy")
    # mask = np.ones(original.shape[0],dtype=bool)
    # mask[3::4]=0

    # TARGET = original[mask]

    # x3, y3, z3 = array_of_points_to_arrays_of_points(TARGET[12:48])

    # print(len(x3))

    # ax.plot(x3, y3, z3, label='TARGET')

    # ####################################

    # ax.legend()

    # plt.show()