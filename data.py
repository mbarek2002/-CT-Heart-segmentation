import os 
import numpy as np
import cv2
from  glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    """create a directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path , split=0.2):
    """load the data and split it into train and test"""
    images = sorted(glob(os.path.join(path, "*", "image", "*.png")))
    masks = [image_path.replace("\\image\\", "\\mask\\") for image_path in images]

    if not images:
        raise FileNotFoundError(f"Aucune image PNG trouvee dans: {path}")
    if not all(os.path.exists(mask_path) for mask_path in masks):
        raise FileNotFoundError("Au moins un masque correspondant est introuvable")

    """Split the data """
    split_size = int(len(images) * split)

    # Split the data into train and test
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=split, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

def augment_data(images, masks , save_path , augment=True):
    """"Performing data augmentation"""
    H = 512
    W = 512

    for idx, (x,y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """Extracting the dir name and image name"""
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(x)))
        name = dir_name + "_" + os.path.splitext(os.path.basename(x))[0]

        """Reading the image and mask"""
        x= cv2.imread(x , cv2.IMREAD_COLOR)
        y= cv2.imread(y , cv2.IMREAD_GRAYSCALE)

        if augment:
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = Rotate(limit=45, p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']


            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        idx= 0
        for i , m in zip(X,Y):
            """Resizing the image and mask"""
            i = cv2.resize(i, (W,H))
            m = cv2.resize(m, (W,H))
            m = m/255
            m = ((m > 0.5) * 255).astype(np.uint8)

            if len(X) == 1 :
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else :
                tmp_image_name = f"{name}_{idx}.jpg"
                tmp_mask_name = f"{name}_{idx}.jpg"

            image_path = os.path.join(save_path , "image/" , tmp_image_name)
            mask_path = os.path.join(save_path , "mask/" , tmp_mask_name)


            cv2.imwrite(image_path , i)
            cv2.imwrite(mask_path , m)

            idx += 1
if __name__ == "__main__":
    """Load the dataset"""
    dataset_path = r"C:\Users\ihebm\Downloads\data\train"
    (train_x, train_y), (valid_x, valid_y) = load_data(path=dataset_path , split=0.2)

    """Create the directories"""
    create_dir("data/train/image")
    create_dir("data/train/mask")
    create_dir("data/valid/image")
    create_dir("data/valid/mask")

    """Augment the data"""
    augment_data(train_x, train_y , save_path="data/train" , augment=True)
    augment_data(valid_x, valid_y , save_path="data/valid" , augment=False)

