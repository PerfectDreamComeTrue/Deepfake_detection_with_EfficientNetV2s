import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import mixed_precision

def extract_number(filepath):
    # 파일명에서 숫자 부분을 추출합니다.
    number = re.findall(r'\d+', filepath)
    return int(number[0]) if number else float('inf')

# GPU 설정 코드
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 모델 및 데이터 설정
model_name = 'EfficientNetV2S'
model_dir = '/Deepfake_detection_with_EfficientNetV2s/Checkpoint/Models'
sub_dir = '/Deepfake_detection_with_EfficientNetV2s/Sub'
IMG_SIZE = 590
NUM_CLASSES = 2
dropout_rate = 0.5
n_folds = 5

for fold in range(n_folds):
    print(f"Testing fold {fold+1}...")
    # 모델 불러오기
    base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(f"{model_dir}/model_fold_{fold+1}.h5")

    # 테스트 이미지 폴더 경로
    test_images_folder = '/content/drive/MyDrive/dataset/test'  

    # 제출 파일을 불러옵니다.
    submission_df = pd.read_csv('/content/drive/MyDrive/submit.csv')

    # 테스트 이미지 파일 경로를 불러옵니다.
    test_image_files = sorted(os.listdir(test_images_folder), key=extract_number)

    # 이미지를 불러와 모델로 예측하고 결과를 저장합니다.
    for idx in submission_df['idx']:
            image_file = f"{idx}"  
            image_path = os.path.join(test_images_folder, image_file)
            image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
            image_array = img_to_array(image)
            image_array = preprocess_input(image_array)
            image_array = np.expand_dims(image_array, axis=0)  

            # 예측
            predictions = model(image_array, training=False)
            predicted_class = np.argmax(predictions, axis=1)


            # 결과를 저장합니다.
            submission_df.loc[submission_df['idx'] == idx, 'label'] = 'Real' if predicted_class[0] == 1 else 'Fake'

    # 결과를 csv 파일로 저장합니다.
    submission_df.to_csv(f'{sub_dir}/{model_name}_fold_{fold+1}.csv', index=False)

