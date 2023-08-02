Follow the instructions to download and preprocess the data.

### Download GloVe pre-trained word embedding
```
cd data
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip
```

### Download MIND dataset
```
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test
rm MINDsmall_*.zip
```
*Note:MIND Small doesn't have a test set, so we just copy the validation set as test set.* 

### Preprocess data into appropriate format
```
cd data
python data_preprocess.py
```

## Credits
Dataset by MIcrosoft News Dataset (MIND), see https://msnews.github.io/.
