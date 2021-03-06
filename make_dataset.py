from utils.tools import DatasetProcessor
from utils.utils import save_to_pickle, load_from_pickle
import os
# 数据集请在dataset_dir文件夹下按文件夹名进行分类
# (Please categorize the datasets by folder name under the folder DATASET_DIR)
# --------------------------------
# dataset_dir = r'D:\datasets\Facial Expression Recognition\fer2013'  # 数据集所在文件夹
dataset_dir = r'../fer2013'  # 数据集所在文件夹
train_dir = os.path.join(dataset_dir, "Training")
test_dir = os.path.join(dataset_dir, "PublicTest")
test_dir2 = os.path.join(dataset_dir, "PrivateTest")
pkl_folder = 'pkl'
train_pickle_fname = "images_targets_train.pkl"
test_pickle_fname = "images_targets_test_pub.pkl"
test_pickle_fname2 = "images_targets_test_pri.pkl"
labels = [
    "anger", "disgust", "fear", "happy", "neutral", "sad", "surprised"
]
# --------------------------------
xml_processor = DatasetProcessor(train_dir, labels)
xml_processor.parse_dataset()
xml_processor.test_dataset()
pkl_dir = os.path.join(dataset_dir , pkl_folder)
pkl_path = os.path.join(pkl_dir, train_pickle_fname)
os.makedirs(pkl_dir, exist_ok=True)
save_to_pickle((xml_processor.image_path_list, xml_processor.target_list), pkl_path)
# ------------------------------------------------
xml_processor = DatasetProcessor(test_dir, labels)
xml_processor.parse_dataset()
xml_processor.test_dataset()
pkl_path = os.path.join(pkl_dir, test_pickle_fname)
save_to_pickle((xml_processor.image_path_list, xml_processor.target_list), pkl_path)
# ------------------------------------------------
xml_processor = DatasetProcessor(test_dir2, labels)
xml_processor.parse_dataset()
xml_processor.test_dataset()
pkl_path = os.path.join(pkl_dir, test_pickle_fname2)
save_to_pickle((xml_processor.image_path_list, xml_processor.target_list), pkl_path)
# xml_processor.image_path_list, xml_processor.target_list = load_from_pickle(pkl_path)
# xml_processor.show_dataset(random=False)
