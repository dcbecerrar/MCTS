class TrieNode(object):
    """
    Trie Node for protein backbone structures.
    """
    
    def __init__(self, coord, epsilon):
        """
        coord: List[float, float, float]
        epsilon: int
        """
        x, y, z = coord
        self.x_range = [x - epsilon, x + epsilon]
        self.y_range = [y - epsilon, y + epsilon]
        self.z_range = [z - epsilon, z + epsilon]
        self.children = []
        # Is this the final atom?
        self.is_terminal_atom = False
        # How many times an atom in this range appeared in the addition process. 
        self.counter = 1
        # Score representing energy of the backbone ending at this atom
        self.energy = 0

    def in_range(self, coord):
        """
        coord: List[float, float, float]
        """
        x, y, z = coord
        
        if self.x_range[0] <= x <= self.x_range[1] \
         and self.y_range[0] <= y <= self.y_range[1] \
         and self.z_range[0] <= z <= self.z_range[1]:
            return True
        return False

    def set_energy(self, energy):
        """
        energy: float
        """
        self.energy = energy


def add_backbone(root, backbone, epsilon, energy=0):
    """
    Adds a backbone to the trie.
    
    backbone: List[List[[float, float, float]]
    epsilon: int
    """
    node = root
    new_backbone_cluster = False

    for atom in backbone:
        found_in_child = False

        # Search for the atom in the children of the present `node`.
        for child in node.children:
            if child.in_range(atom):
                # We found it, increase the counter by 1 to keep track that another.
                # backbone has it as well.
                child.counter += 1
                # And point the node to the child that contains this atom.
                node = child
                found_in_child = True
                break
        
        # We did not find it so add a new child.
        if not found_in_child:
            new_node = TrieNode(atom, epsilon)
            node.children.append(new_node)
            # And then point node to the new child.
            node = new_node
            new_backbone_cluster = True

    # Everything finished. Mark it as the end of a backbone.
    node.is_terminal_atom = True

    if new_backbone_cluster:
        node.set_energy(energy)

    return new_backbone_cluster

def in_backbone(root, backbone_fragment):
    """
    Check and return 
      1. If the backbone fragment exists in the trie.

    backbone: List[List[[float, float, float]]
    """
    node = root
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie.
    if not root.children:
        return False, 0
    for atom in backbone_fragment:
        atom_not_found = True
        # Search through all the children of the present `node`.
        for child in node.children:
            if child.in_range(atom):
                # We found the char existing in the child.
                atom_not_found = False
                # Assign node as the child containing the atom and break.
                node = child
                break
        # Return False anyway when we did not find an atom.
        if atom_not_found:
            return False, 0
    # Backbone found.
    return True, node.energy


def parse_rosetta_scores(filename):
    sequence_rank_to_score = {}

    with open(filename, "r") as infile:
        infile.readline()
        infile.readline()

        for line in infile:
            line = [e for e in line.split(" ") if e]
            
            score, rank = float(line[1]), int(line[-1].split("_")[0])
            sequence_rank_to_score[rank] = score

    return sequence_rank_to_score

def parse_mcts_output(filename):
    sequence_rank_to_angles = {}

    with open(filename, "r") as infile:
        rank = 1
        for line in infile:
            angles = eval("[" + line.split(", [")[1])
            sequence_rank_to_angles[rank] = angles
            rank+= 1

    return sequence_rank_to_angles


if __name__ == "__main__":
    pass

                