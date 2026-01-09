import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class DecoyDataset_for_final(Dataset):

    def __init__(self,
                 targets,
                 msa_transformer=True,
                 root_dir="/share/home/zhanglab/Gly/Database_MD/GPCR_and_ALTAS_feature_NEW",
                 multi_dir=False,
                 root_dirs=["/projects/casp/dldata/", "/projects/casp/dldata_ref/"],
                 lengthmax=500,
                 verbose=False,
                 include_native=True,

                 distance_cutoff=0,
                 features=[],
                 msa_path="/share/home/zhanglab/Gly/Database_MD/GPCR_ATLAS_PDB_MSA_Feature_All",
                 pairs_dic_txt="/share/home/zhanglab/Gly/Database_MD/GPCR_ALTAS_PDB_contains_pairs.txt",
                 template_path = "/share/home/zhanglab/Gly/Database_MD/template_feature_pdb_atlas",
                 use_bfactor=False,
                 ):
                     

        # Properties
        self.targets = targets
        self.use_bfactor = use_bfactor
        self.pairs_dic_txt = pairs_dic_txt
        self.msa_path = msa_path
        self.datadir = root_dir
        self.template_path = template_path

        self.verbose = verbose
        self.include_native = include_native
        self.distance_cutoff = distance_cutoff
        self.lengthmax = lengthmax
        self.multi_dir = multi_dir
        self.root_dirs = root_dirs
        self.features = features
        self.msa_transformer = msa_transformer

        self.n = {}
        self.set_dict = {}
        self.sizes = {}

        self.pairs_dic = {}
        self.proteins = []

        print(f"Initial targets number: {len(self.targets)}")

        # 从文件中读取对照关系
        with open(self.pairs_dic_txt) as f:
            for line in f.readlines():
                lines = line.strip().split("\t")
                primary = lines[0]
                secondary_pairs = lines[1:]
                self.pairs_dic[primary] = secondary_pairs


        temp = []
        msa_not_exist_count = 0
        template_not_exist_count = 0
        processed_primaries = set()

        for line in self.targets:
            group = line.strip().split("\t")
            primary = group[0]
            if primary in processed_primaries:
                continue

            primary_path = os.path.join(self.datadir, primary) + ".npz"
            if not os.path.exists(primary_path):
                continue


            dic = {"x": primary_path}

            msa_primary_path = (os.path.join(self.msa_path, primary, primary + ".npz") if not self.msa_transformer else
                                os.path.join(self.msa_path, primary, primary + "_random.npz"))

            if os.path.exists(msa_primary_path):
                dic["msa_path"] = msa_primary_path
            else:
                msa_not_exist_count += 1
                continue


            template_primary_path = os.path.join(self.template_path, primary + ".npz")

            if os.path.exists(template_primary_path):
                dic["template_path"] = template_primary_path #
            else:
                template_not_exist_count += 1
                continue


            temp.append(primary)
            self.set_dict[primary] = dic
            processed_primaries.add(primary)

        self.proteins = temp
        print("Proteins successfully load Number:", len(temp))
        print("MSA failed load Number:", msa_not_exist_count)
        print("template failed load Number:", template_not_exist_count)
        print("--*Initialization complete*--.")

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx, transform=True):
        try:

            if torch.is_tensor(idx):
                idx = idx.tolist()

            pname = self.proteins[idx]
            npz_file_path = os.path.join(self.datadir, pname + ".npz")

            if not os.path.exists(npz_file_path):
                raise FileNotFoundError(f"{npz_file_path} not found")

            dic = self.set_dict[pname]
            data = np.load(npz_file_path, allow_pickle=True)

            if "template_path" in dic and os.path.exists(dic["template_path"]):
                template_data = np.load(dic["template_path"], allow_pickle=True)

                if "result" in template_data:
                    template = template_data["result"].astype(np.float32)
                else:
                    raise KeyError(f"template文件中未找到 'result': {dic['template_path']}")
            else:
                print(f"template 文件 '{dic['template_path']}' 不存在，跳过 'template_path' ")
                template = None


            data_template = self.min_max_normalize(template)

            if pname not in data:
                raise KeyError(f"数据中缺少主键 '{pname}': {dic['x']}")

            protein_data = data[pname].item()

            if "idx" not in protein_data:
                raise KeyError(f"数据中缺少键 'idx': {dic['x']}")

            idx = protein_data["idx"]
            val = protein_data["val"]

            length = protein_data["tbt"].shape[1]
            psize = length

            if not self.msa_transformer:
                msa = np.load(dic["msa_path"])["msa"]
                msa_length = msa.shape[-1]
            else:
                msa = np.load(dic["msa_path"])
                msa_feats = msa["msa_feats"]
                row_att = msa["row_att"]
                msa_length = row_att.shape[-2]
                msa = {"msa_feats": msa_feats,
                       "row_att": row_att}

            if not (length < self.lengthmax and msa_length == length):
                raise Exception("msa长度不同", "msa_length:", msa_length, "seq_length:", length)

            angles = np.stack([np.sin(protein_data["phi"]),
                               np.cos(protein_data["phi"]),
                               np.sin(protein_data["psi"]),
                               np.cos(protein_data["psi"])], axis=-1)

            obt = protein_data["obt"].T
            prop = protein_data["prop"].T

            if not self.use_bfactor and prop.shape[-1] == 56:
                prop = prop[:, :-1]
            if np.isnan(prop).any():
                raise Exception("prop存在nan值")

            orientations = np.stack([protein_data["omega6d"], protein_data["theta6d"], protein_data["phi6d"]],
                                    axis=-1)
            orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
            euler = np.concatenate([np.sin(protein_data["euler"]), np.cos(protein_data["euler"])],
                                   axis=-1)
            maps = protein_data["maps"]
            tbt = protein_data["tbt"].T
            sep = self.seqsep(psize)

            another_list = []
            if pname in data:
                primary_tbt = data[pname].item()["tbt"][0]
            for primary, secondary_pairs in self.pairs_dic.items():
                if pname == primary:
                    for secondary in secondary_pairs:
                        if secondary in data:
                            another_tbt = data[secondary].item()["tbt"][0]
                            another_list.append(another_tbt)

                        else:
                            print(f"not found secondary '{secondary}': {npz_file_path}")

            if not another_list:
                print(f"not found  primary {pname} for secondary")
                return {'pname': pname}

            count = len(another_list)
            estogram_sum = 0
            for another_tbt in another_list:
                estogram_sum += abs(primary_tbt - another_tbt)

            estogram = estogram_sum / count
            estogram = estogram >= 2


            if transform:
                tbt[:, :, 0] = self.dist_transform(tbt[:, :, 0])
                maps = self.dist_transform(maps, cutoff=self.distance_cutoff)
            _1d = np.concatenate([angles, obt, prop, data_template], axis=-1)
            _2d = np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
            _2d = np.expand_dims(_2d.transpose(2, 0, 1), 0) 
            print(_1d)
            print(_2d)
            exit()

            if len(self.features) > 0:
                inds1d, inds2d = self.getMask(self.features)
                _1d = _1d[:, inds1d]
                _2d = _2d[:, inds2d, :, :]

            sample = {'idx': idx.astype(np.int32),
                      'val': val.astype(np.float32),
                      '1d': _1d.astype(np.float32),
                      '2d': _2d.astype(np.float32),
                      'msa': msa,
                      'estogram': np.expand_dims(estogram.astype(np.float32), 0),
                      'pname': pname,
                      }

            return sample
        except Exception as e:
            print(f"Exception in __getitem__ with idx={idx}: {str(e)}")
            return {'pname': str(pname)}

    def min_max_normalize(self, tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        if max_val == min_val:
            return np.zeros_like(tensor)
        data_template = (tensor - min_val) / (max_val - min_val)
        return data_template

    # VARIANCE REDUCTION
    def dist_transform(self, X, cutoff=4, scaling=3.0):
        X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
        return np.arcsinh(X_prime) / scaling

    def seqsep(self, psize, normalizer=100, axis=-1):
        ret = np.ones((psize, psize))
        for i in range(psize):
            for j in range(psize):
                ret[i, j] = abs(i - j) * 1.0 / normalizer - 1.0
        return np.expand_dims(ret, axis)

    # Getting masks
    def getMask(self, include):
        feature2D = [("distance", 1), ("rosetta", 9), ("distance2", 4), ("orientation", 18), ("seqsep", 1),
                     ("bert", 16)]
        feature1D = [("angles", 10), ("rosetta", 4), ("ss", 4), ("aa", 52)]
        print("2d:",feature2D.shape)
        print("1d:",feature1D.shape)
        for e in include:
            if e not in [i[0] for i in feature2D] and e not in [i[0] for i in feature1D]:
                print("Feature names do not exist.")
                print([i[0] for i in feature1D])
                print([i[0] for i in feature2D])
                return -1
        mask = []
        temp = []
        index = 0
        for f in feature1D:
            for i in range(f[1]):
                if f[0] in include: temp.append(index)
                index += 1
        mask.append(temp)
        temp = []
        index = 0
        for f in feature2D:
            for i in range(f[1]):
                if f[0] in include: temp.append(index)
                index += 1
        mask.append(temp)
        return mask