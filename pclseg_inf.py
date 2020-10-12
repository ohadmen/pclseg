import argparse
import os
from time import sleep

import torch
import yaml
import numpy as np
from common.laserscan import LaserScan
from modules.segmentator import Segmentator



class PclSegInf:

    def get_laserscan_data(self, ls: LaserScan):
        proj_mask = ls.proj_mask
        proj = np.r_[np.expand_dims(ls.proj_range, 0),
                     ls.proj_xyz.transpose([2, 0, 1]),
                     np.expand_dims(ls.proj_remission, 0)]
        proj = proj - np.array(self.model.sensor_img_means).reshape(-1, 1, 1)
        proj = proj / np.array(self.model.sensor_img_stds).reshape(-1, 1, 1)
        proj = proj * proj_mask.astype(float)
        return proj, proj_mask

    @torch.no_grad()
    def __init__(self, modeldir):
        arch = yaml.safe_load(open(modeldir + "/arch_cfg.yaml", 'r'))
        data = yaml.safe_load(open(modeldir + "/data_cfg.yaml", 'r'))
        n_classes = len(data['learning_map_inv'])

        self.class_colormap = np.zeros((n_classes, 3))
        for k, v in data['color_map'].items():
            kp = data['learning_map'][k]
            self.class_colormap[kp] = v

        self.model = Segmentator(arch, n_classes, modeldir)
        self.model.cuda()
        self.model.eval()
        # use knn post processing?
        self.post = None
        if arch["post"]["KNN"]["use"]:
            self.post = KNN(arch["post"]["KNN"]["params"],
                            n_classes)

    @torch.no_grad()
    def __call__(self, ls: LaserScan):
        torch.cuda.empty_cache()
        proj_in, _ = self.get_laserscan_data(ls)
        proj_in = torch.from_numpy(proj_in).unsqueeze(0).float().cuda()
        # compute output
        proj_output = self.model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        return proj_argmax.cpu().detach().numpy()


if __name__ == '__main__':
    from pyzview import Pyzview

    parser = argparse.ArgumentParser("pclseg inference")
    parser.add_argument('--modeldir', '-m', type=str, required=True, default=None,
                        help='Directory to get the trained model.')
    parser.add_argument('--datadir', '-d', type=str, required=True, default=None,
                        help='Directory to get the data.')

    # parser.add_argument('--data', '-d', type=str, required=True, default=None,help='Directory to get the data')
    args, _ = parser.parse_known_args()
    zv = Pyzview()
    files = [os.path.join(args.datadir, x) for x in os.listdir(args.datadir) if
             os.path.isfile(os.path.join(args.datadir, x)) and x[-3:] == 'bin']
    files = np.sort(files)
    ls = LaserScan(project=True, W=1024)
    zv.remove_shape(-1)
    inf = PclSegInf(args.modeldir)
    i = 0
    play_mode = True
    while i != len(files):
        key = zv.get_last_keystroke()
        if key == Pyzview.KEY_SPACE:
            play_mode = not play_mode
        if not play_mode:
            sleep(0.1)
            continue
        else:
            i = i + 1
        f = files[i]
        ls.open_scan(f)
        label = inf(ls)
        label_color = inf.class_colormap[label] / 255
        xyzl = np.concatenate([ls.proj_xyz, label_color], axis=2)
        xyzl[ls.proj_mask] = np.nan

        k = zv.add_points("dbg", xyzl.reshape(-1, 6)) \
            if 'k' not in locals() else \
            zv.update_points(k, xyzl.reshape(-1, 6))
