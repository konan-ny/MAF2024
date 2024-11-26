#MAF2024/data/co-occurrence-bias 스크립트 실행

mkdir -p "original_LAMA"
cd original_LAMA
wget dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
python preprocess_LAMA_TREx.py
