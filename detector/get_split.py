import os
import os.path as osp
import numpy as np

from options import DetectorOptions
from utils import write_to_file

''' python3 get_split.py
'''

if __name__ == "__main__":

    train_frac = 0.7
    val_frac = 0.1
    test_frac = 0.2

    # Load Options
    options = DetectorOptions()
    opt = options.parse()

    traversals = [t for t in os.listdir(opt.data_dir) if '.txt' not in t]

    assert len(set([len(os.listdir(osp.join(opt.data_dir, t, 'images'))) for t in traversals])) == 1, 'All traversals must contain the same number of locations'

    locations = list(set([osp.splitext(f)[0] for f in os.listdir(osp.join(opt.data_dir, traversals[0], 'images'))]))

    for l in locations:
        for t in traversals:
            assert osp.exists(osp.join(osp.join(opt.data_dir, t, 'images', l+'.png'))), 'Path must exists'
    
    np.random.shuffle(locations)

    train_end = int(train_frac*len(locations))
    val_end = train_end+int(val_frac*len(locations))

    train_locs = locations[:train_end]
    val_locs = locations[train_end:val_end]
    test_locs = locations[val_end:]
    assert len(set(train_locs+val_locs+test_locs)) == len(locations)

    print(f'Generated {train_frac}/{val_frac}/{test_frac} split:')
    for name, lst in zip(['train','val','test'], [train_locs,val_locs,test_locs]):
        fpath = osp.join(opt.data_dir, f'{name}.txt')
        write_to_file(lst, fpath)
        print(f'\tNumber of {name:<5s}: {len(lst):<5}', '-> saved to', fpath)
