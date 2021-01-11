git clone https://github.com/Microsoft/Recommenders
cd Recommenders
python tools/generate_conda_file.py --gpu
conda env create -f reco_gpu.yaml
