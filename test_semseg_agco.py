"""
Author: Morten
Date: Dec 2024
"""
import argparse
import os
from data_utils.AGCODataLoader import TractorsAndCombines
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['other', 'tractor', 'combine']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--root_folder', type=str, default='/home/agco/datasets/agco_real/', help='Path to root folder of data.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 1]')
    parser.add_argument('--gpu', type=str, default='0,1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=30000, help='point number [default: 30000]')
    parser.add_argument('--log_dir', type=str, default="size_200_test_300_epochs", help='experiment root')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = '/home/morten/Repos/Pointnet_Pointnet2_pytorch/log/sem_seg_agco_real/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = len(classes)
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = args.root_folder

    TEST_DATASET = TractorsAndCombines(root, split='test')
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # Remove "module." prefix from the checkpoint keys
    state_dict = checkpoint['model_state_dict']
    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    classifier.load_state_dict(state_dict)

###########################################################################
    with torch.no_grad():
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]

        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        classifier = classifier.eval()

        log_string('---- Test Results ----')
        for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, _ = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            testDataLoader.collate_fn

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
        
        # OA
        point_accuracy = total_correct / float(total_seen)
        log_string('eval point accuracy: %f' % (point_accuracy))
        
        # mAcc
        mAcc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))
        log_string('eval point avg class acc: %f' % (mAcc))

        # mIoU logging
        log_string('eval mIoU: %f' % (mIoU))
        
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))

        log_string(iou_per_class_str)


if __name__ == '__main__':
    args = parse_args()

    main(args)
