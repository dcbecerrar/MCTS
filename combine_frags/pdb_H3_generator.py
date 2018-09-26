import numpy as np
import os
import sys

def make_pdb(coord, name, template):
  """ Makes pdb using pdb template of H3 loop backbone and
  coordinates in array """


  # Make pdb coordinates round to 3 decimal places
  coord_str = np.char.mod("%7.3f", coord)

  #with open("TIGIT_sidechain_template.pdb", "rt") as data:
  # template = data.readlines()

  #offset=62     # number of LINES in H3 template before design
  offset=35
  
  num_lines = len(template)

  with open("{}".format(name), "wt") as fout:
    for j in range(num_lines):
      if j >= offset and j < (offset+(coord.shape[0])):
        fout.write(template[j].replace("xyz", " ".join(coord_str[j-offset])))
      else:
        fout.write(template[j])


def insert_H3(H3_file, outfile, template):
  """ Inserts H3 loop into template fab with TIGIT without H3 """
  '''
  tempdata="tigit_fab_with_H3_template.pdb"    
        
  temp=open(tempdata)
  template = temp.read()
  temp.close()
  '''

  H3 = open(H3_file)
  H3data = H3.read()
  H3.close()

  insert = template.replace("XYZ\n", H3data) # Omits TER line in H3 file

  out=open(outfile,'w')
  out.write(insert)
  out.close()

if __name__ == "__main__":
    H3_dir = sys.argv[1]
    H3_files = os.listdir(sys.argv[1])
    print(sys.argv[1])
    outdir = sys.argv[2]
    
    temp = open("./5ggs_relaxed_all_with_H3_template.pdb")
    FAB_TEMP = temp.read()
    temp.close()

    for i in H3_files:
        insert_H3(H3_dir+"/"+i, outdir+"/"+i[:-4]+"_fab.pdb", FAB_TEMP)

