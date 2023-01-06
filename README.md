# Window-based HOI Coreference Resolution

## Setup
* environment
    1. Build Python 2.7 environment for `get_conll_data.sh` to run correctly. To do it with with conda:
        ```
        conda create -y --name py27 python=2.7 && conda activate py27
        ```
    2. Build python3 environment for this repo. To do it with with conda:
        ```
        conda create -y --name win-coref python=3.7 openjdk perl
        conda activate win-coref
        python -m pip install -r requirements.txt
        ```
* data
    1. Get OntoNotes 5.0 corpus (download [here](https://catalog.ldc.upenn.edu/LDC2013T19), registration needed)
    2. Extract OntoNotes 5.0 arhive. In case it's in the repo's root directory:
        ```
        tar -zxvf ontonotes-release-5.0_LDC2013T19.tgz
        ```
    3. Switch to Python 2.7 environment and Run the conll data preparation scripts
        ```
        conda activate py27
        bash get_conll_data.sh ontonotes-release-5.0 data
        ```
    4. Download conll scorers and Stanford Parser:
        ```
        bash get_third_party.sh
        ```
    5. Switch to Python3 environment and Build the corpus in jsonlines format:
        ```
        conda activate win-coref
        python convert_to_jsonlines.py data/conll-2012/ --out-dir data --keep-tmp-dir
        python convert_to_heads.py
        ```
        
