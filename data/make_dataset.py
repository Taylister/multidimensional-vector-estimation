import os
import sys
import argparse
import imghdr
import random
import shutil
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--split', type=float, default=0.8)


# the structure of data_dir is like below
# /data
# └ images
# └ vectors
# └ information.csv
#  this program copy the files like a below structure
# /data
# └ train
#    └ images
#    └ vectors  
# └ test
#    └ images 
#    └ vectors
# └ information.csv

def main(args):
    args.data_dir = os.path.expanduser(args.data_dir)

    print('loading dataset and proceccing image data')

    src_image_paths = []
    target_dir_path = os.path.join(args.data_dir,"images")

    for file in os.listdir(target_dir_path):
        path = os.path.join(target_dir_path, file)
        if imghdr.what(path) == None:
            continue

        src_image_paths.append(path)
    random.shuffle(src_image_paths)

    src_vector_paths = []
    target_dir_path = os.path.join(args.data_dir,"vectors")

    for path in src_image_paths:
        filename = os.path.basename(path).split(".")[0] + ".npy"
        target_file_path = os.path.join(target_dir_path,filename)
        if not os.path.exists(target_file_path):
            print("Error: Couldn't find the vector file", file=sys.stderr)
            sys.exit()
        src_vector_paths.append(target_file_path)

    # separate the paths
    border = int(args.split * len(src_image_paths))
    train_image_paths = src_image_paths[:border]
    test_image_paths = src_image_paths[border:]
    
    train_vector_paths = src_vector_paths[:border]
    test_vector_paths = src_vector_paths[border:]
    

    print('the number of train images and the vectors : %d' % len(train_image_paths))
    print('the number of test images and the vectors: %d' % len(test_image_paths))

    # create dst directories
    train_image_dir = os.path.join(args.data_dir, 'train/images')
    test_image_dir = os.path.join(args.data_dir, 'test/images')

    train_vector_dir = os.path.join(args.data_dir, 'train/vectors')
    test_vector_dir = os.path.join(args.data_dir, 'test/vectors')

    if os.path.exists(train_image_dir) == False:
        os.makedirs(train_image_dir)
    if os.path.exists(test_image_dir) == False:
        os.makedirs(test_image_dir)

    if os.path.exists(train_vector_dir) == False:
        os.makedirs(train_vector_dir)
    if os.path.exists(test_vector_dir) == False:
        os.makedirs(test_vector_dir)

    # move the image files
    pbar = tqdm.tqdm(total=len(src_image_paths))
    for dset_paths, dset_dir in zip([train_image_paths, test_image_paths, train_vector_paths, test_vector_paths], [train_image_dir, test_image_dir, train_vector_dir, test_vector_dir]):
        for src_path in dset_paths:
            dst_path = os.path.join(dset_dir, os.path.basename(src_path))
            shutil.move(src_path, dst_path)
            pbar.update()
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)