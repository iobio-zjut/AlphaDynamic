# AlphaDynamic
# Accuracy modeling of protein multiple conformations guided by predicted residue dynamics

AlphaDynamic provides a method to capture protein multiple conformations. AlphaDynamic predicts multiple conformational states through AlphaFold2 by targeted masking of MSA columns corresponding to deep learning-predicted movement residues, to selectively attenuate dominant conformational signals, enhance minor conformational features, and preserve evolutionary constraints in structural core regions. This approach establishes a quantitative relationship among structural dynamics, evolutionary features, and functional conformations, achieving robust multiple conformations prediction while fully retaining AlphaFold2’s original architectural framework.


## 👨‍💻 Developer

**Lingyu Ge**  
College of Information Engineering  
Zhejiang University of Technology, Hangzhou 310023, China  
✉️ Email: [gelingyu@zjut.edu.cn](mailto:gelingyu@zjut.edu.cn)

**Xinyue Cui**  
College of Information Engineering  
Zhejiang University of Technology, Hangzhou 310023, China  
✉️ Email: [Cuixinyue@zjut.edu.cn](mailto:Cuixinyue@zjut.edu.cn)

## 🛠 Installation

### Environment Requirements

Make sure the following software/tools are installed:

- [Python 3.8+]
- [PyRosetta]
- [FoldSeek]

Install the required Python packages:

```bash
absl-py==1.0.0
biopython==1.79
chex==0.0.7
dm-haiku==0.0.12
dm-tree==0.1.6
docker==5.0.0
immutabledict==2.0.0
ml-collections==0.1.0
numpy==1.24
pandas==1.3.4
scipy==1.11
tensorflow-cpu==2.11.0/
```

### 📥 Required Models & Resources
Download and place the following models:

AlphaFold2 parameters/
From: https://github.com/google-deepmind/alphafold/
→ Download: params_model_1_ptm.npz/
→ Place in: ./AlphaFlex/af_multiple_conformation/params/

ESM-MSA-1b model/
From: https://github.com/facebookresearch/esm/
→ Download: esm_msa1b_t12_100M_UR50S.pt/
→ Place in: ./AlphaFlex/flexible_residue/MSA_embedding/model/

TMalign executable/
From: https://zhanggroup.org/TM-score/
→ Download: TMalign/
→ Place in: ./AlphaFlex/scripts/

### 📂 Example Output
```bash
./AlphaFlex/example/4AKE_B/
```

### Running
#### ⚙️ Configuration Parameters (`config.sh`)
```bash
./AlphaFlex/scripts/config.sh

| Parameter          | Description                                              |
|--------------------|----------------------------------------------------------|
| `BASE_DIR`         | Absolute path to the root directory of the project       |
| `pdb_dir`          | Path to the input protein structure files (PDB format)   |
| `fasta_dir`        | Path to the input FASTA files (optional)                 |
| `msa_out_dir`      | Directory where generated MSAs will be saved             |
| `flagfile`         | Path to AlphaFold2 configuration flags                   |
| `target_db`        | FoldSeek-compatible database built from AFDB             |
| `filter_list`      | Text file containing the list of target protein names    |
| `template_dir`     | Directory containing AFDB templates                      |
| `msa_database_dir` | Msa database                                             |
| `num_threads`      | Number of parallel processes to run                      |
| `num_comformations`| Number of multiple conformation to predict               |
```

#### ⚙️ Configuration Parameters (`monomer_full_dbs.flag`)
```bash
./AlphaFlex/af_multiple_conformation/monomer_full_dbs.flag
```

#### 🚀 generate multiple conformations
```bash
bash ./AlphaFlex/scripts/run.sh
```

## 📄 License & Acknowledgement

© 2025 **Intelligent Optimization and Bioinformatics Lab**, Zhejiang University of Technology
