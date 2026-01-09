import os
import sys
import string
import random
msa_path=sys.argv[1]
MSA_path=sys.argv[2]
msa_128_path=sys.argv[3]


table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
with open(msa_path, 'r') as f_msa, open(MSA_path, 'w') as f2:
    lines = f_msa.readlines()
    for line in lines:
        if line.startswith('>'):
            continue
        else:
            line1 = line.translate(table)
            f2.write(line1)

with open(MSA_path, 'r') as f1:
    line_1 = f1.readlines()
    N = len(line_1)
    with open(msa_128_path, 'w') as f2:
        if N > 128:
            randomList = random.sample(range(1, N), 127)
            for i, line_64 in enumerate(line_1):
                if i == 0 or i in randomList:
                    f2.write('>')
                    f2.write('\n')
                    f2.write(line_64)

        else:
            for line_64 in line_1:
                f2.write('>')
                f2.write('\n')
                f2.write(line_64)
