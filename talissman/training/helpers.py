import gdown
import os

def download_dataset(dataset_id, output_dir, overwrite=False):
    if not isinstance(dataset_id, (list, tuple)):
      dataset_id = [dataset_id]
    dsPath=[]
    for i, id in enumerate(dataset_id):
      p=os.path.join(output_dir, "dataset{}.h5".format(i))
      dsPath.append(p)
      if overwrite or not os.path.exists(p):
          gdown.download('https://drive.google.com/uc?id={}'.format(id), p, quiet=False)
    return dsPath
