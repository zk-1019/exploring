from utils import create_input_files

if __name__ == '__main__':
    create_input_files(dataset='coco',
                       karpathy_json_path='D:\\my-torch - sy\\test.json',
                       image_folder='D:\\my-torch - sy\\JPEGImages',
                       captions_per_image=3,
                       min_word_freq=3,
                       output_folder='D:\\my-torch - sy\\caption data',
                       max_len=50)
