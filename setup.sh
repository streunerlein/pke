# install EmbedRank project
cd ai-research-keyphrase-extraction/

# install sent2vc for EmbedRank project
git clone https://github.com/epfml/sent2vec.git
cd sent2vec/
git checkout f827d014a473aa22b2fef28d9e29211d50808d48
make
pip uninstall Cython
pip install Cython
cd src/
export MACOSX_DEPLOYMENT_TARGET=10.10 # if you're on mac
python setup.py build_ext
pip install .

cd ../../../ # should be back in repository root
pip install -e ai-research-keyphrase-extraction/

# install PKE
pip install -e .

# install nltk data
python -m nltk.downloader stopwords # PKE
python -m nltk.downloader universal_tagset # PKE
python -m nltk.downloader punkt # EmbedRank

# spacy models for every language you would like to support
python -m spacy download fr # download the french model
python -m spacy download de # download the german model
python -m spacy download en # download the english model
