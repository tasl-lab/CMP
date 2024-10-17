## Environment Setup

```bash
conda env create --name cmp --file=environment.yml
conda activate cmp
pip install -r requirements.txt

python opencood/setup.py develop
python opencood/utils/setup.py build_ext --inplace

cd MTR
python setup.py develop 
cd ..

# In the project root:
export PYTHONPATH="$(pwd)/AB3Dmot:$PYTHONPATH"
export PYTHONPATH="$(pwd)/AB3Dmot/Xinshuo_PyToolbox:$PYTHONPATH"
export PYTHONPATH="$(pwd)/MTR:$PYTHONPATH"
export PYTHONPATH="$(pwd)/opencood:$PYTHONPATH"
```