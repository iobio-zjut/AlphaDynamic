import sys
import numpy as np
import torch
import torch.nn.functional as F
from Network import Network
from esm_predict import msa_trans

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定编号为0的一张显卡，"0,1,2,3"是4张

msa_128_path = sys.argv[1]
out = sys.argv[2]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    msa_feats, row_att = msa_trans(msa_128_path)
    np.savez_compressed(out,
                        msa_feats=msa_feats,
                        row_att=row_att
                        )

if __name__ == "__main__":
    main()
