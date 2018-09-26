from Bio import SeqIO
from Bio.SubsMat import MatrixInfo

def score_match(pair, matrix):
    if pair not in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return matrix[pair]

f=open("score.sc")

#Skips first two lines
f.readline()
f.readline()

#First line with pdb file
line=f.readline()

original="YYCARRDYRFDMGFDYWGQGT"

final=open("5ggs_relaxed_fastrelax_backbone_only_all_pdb_H3_seq.txt", 'w')

blosum=MatrixInfo.blosum62

while line:
    info=line.split()
    score=info[1]
    pdb_name=info[-1]+".pdb"

    with open(pdb_name,'rU') as pdb:
        lst=[]
        for record in SeqIO.parse(pdb, "pdb-atom"):
            lst+=[record.seq]
        sequence=str(lst[0][92:113])

    blosum_score=0
    hamming_score=0

    for i in range(len(original)):
        blosum_score+=score_match((original[i], sequence[i]), blosum)
        hamming_score+=int(original[i]!=sequence[i])



    final.write(pdb_name+"\t"+score+"\t"+str(blosum_score)+"\t"+str(hamming_score)+"\t"+sequence+"\n")

    line=f.readline()

f.close()




