import numpy as 

class BatchGenerator(object):
    
    dataset = np.loads(os.path.join(data_dir, "dataset.npy"))
    labels = np.loads(os.path.join(data_dir, "labels.npy"))
