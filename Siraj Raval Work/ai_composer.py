import urllib
import zipfile
import nottingham_util
import rnn

url = "www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip"
urllib.urlretrieve(url, "dataset.zip")

zip = zipfile.ZipFile(r'dataset.zip')
zip.extractall('data')

nottingham_util.create_model()

rnn.train_model()

#ne compile pas 
# voir https://github.com/yoavz/music_rnn