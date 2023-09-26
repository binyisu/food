import os
import math
import argparse
import numpy as np
from tabulate import tabulate
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='../output/voc_coco/HMWA_2', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[1,2,3,5,10,30], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'result.txt'), 'w')

    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue

            file_paths.append(os.path.join(_dir, 'log.txt'))

        header, results = [], []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            print(fpath)
            # with open (fpath, 'r') as f:
            #     infer_reaults = json.load(f)
            result = []
            for i in range(16):
                i = 16-i
                res_info = lineinfos[-i].strip()
                header.append(res_info.split(':')[-1].split('=')[0])
                result.append(float(res_info.split(':')[-1].split('=')[-1]))
            results.append([fid] + result)


        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        # cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        results.append(['μ'] + avg[1:])
        # results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()
