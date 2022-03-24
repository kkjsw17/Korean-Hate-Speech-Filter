import gdown
import os
import shutil

if __name__ == '__main__':
    url = 'https://drive.google.com/u/0/uc?id=1dn9fBXo6r-J8OpvHaOWYeZF238WdSWcK&export=download'
    output = '.model/model.pt'
    
    if os.path.exists('./.model'):
        shutil.rmtree('./.model')
    os.mkdir('./.model')
    gdown.download(url, output)