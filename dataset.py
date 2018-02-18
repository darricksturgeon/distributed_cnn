import os
import pathlib
import tarfile
from urllib.request import urlretrieve

from sklearn.model_selection import train_test_split


class Dataset(object):
    """
    For initial loading in images/segmentations into the tensorflow pipeline.
    Retrieves paths and labels or paths to segmentations, downloads if necessary.
    """

    def __init__(self, name='OxfordFlower',
                 datadir=os.path.expanduser('~/OxFlowers'), **kwargs):
        if name == 'OxfordFlower':
            self.images, self.labels = get_oxford_flower_dataset(datadir, **kwargs)
            # make absolute paths, slightly clunky, but quick for the programmer.
            self.images = list(map(
                lambda x: str(pathlib.Path(x).resolve()), self.images
            ))

        self.train_imgs, self.train_labels, self.test_imgs, self.test_labels \
            = train_test_split(self.images, self.labels,
                               shuffle=True, stratify=self.labels,
                               train_size=.6)

    def get_train_data(self):
        return self.train_imgs, self.train_labels

    def get_test_data(self):
        return self.test_imgs, self.test_labels


def get_oxford_flower_dataset(dir, segs_instead=False, **kwargs):
    # downloads and/or returns paths to the oxford flower dataset.

    url1 = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
    folder1 = _retrieve_dataset(url1, dir, 'flowers.tgz')
    imagedir = os.path.join(folder1, 'jpg')

    with open(os.path.join(imagedir, 'files.txt')) as f:
        image_set = [imagedir + '/' + im for im in f.readlines()]

    # there are 17 classes of flower with 80 images each.  They are sorted in the folder.
    if labeled:
        labels = []
        for i in range(0, 17):
            labels += [i] * 80

        return image_set, labels
    else:
        pass  # download segmentations instead.  Possible goal if training is too fast here.


def _retrieve_dataset(url, dir, name):

    if not os.path.exists(dir):
        os.makedirs(dir)
    target = dir + '/' + name
    if not os.path.exists(target):
        print('downloading...')
        urlretrieve(url, target)
    unzipped = target[:-4]
    if not os.path.exists(unzipped):
        f = tarfile.open(target, 'r:gz')
        f.extractall(unzipped)
        f.close()
    return unzipped


if __name__ == '__main__':
    pass