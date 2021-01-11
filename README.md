```bash
git clone https://github.com/chungdz/recommenders.git
cd recommenders
python tools/generate_conda_file.py --gpu
conda env create -f reco_gpu.yaml
conda activate reco_gpu
cd running
./run.sh
```
