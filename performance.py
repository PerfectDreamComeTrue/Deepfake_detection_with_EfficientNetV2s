# performance.py
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import mixed_precision

# GPU 설정 코드
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 모델 및 데이터 설정
n_folds = 5
model_name = 'EfficientNetV2S'
model_dir = '/Deepfake_detection_with_EfficientNetV2s/Checkpoint/Models'
history_dir = '/Deepfake_detection_with_EfficientNetV2s/Checkpoint/History'
sub_dir = '/Deepfake_detection_with_EfficientNetV2s/Sub'
IMG_SIZE = 384
NUM_CLASSES = 2
le = LabelEncoder()
dropout_rate = 0.7

print(f"Performance evaluation for {model_name}")

# 각 폴드에 대한 성능 지표를 저장할 리스트
acc_list = []
pre_list = []
rec_list = []
f1_list = []
auc_list = []


for fold in range(n_folds):
    print(f"Evaluating fold {fold+1}...")

    # 폴드별 데이터 로드
    val_images_paths = np.load(f"{history_dir}/val_images_fold_{fold+1}.npy")
    val_labels = np.load(f"{history_dir}/val_labels_fold_{fold+1}.npy")
    
    val_images = []
    for img_path in val_images_paths:
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        val_images.append(img_array)
    val_images = np.array(val_images)

    # 폴드별 모델 로드
    base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(f"{model_dir}/model_fold_{fold+1}.h5")

    # 성능 평가
    val_predictions = model.predict(val_images)
    val_pred_labels = np.argmax(val_predictions, axis=1)

    # val_labels가 numpy 배열인지 확인
    val_labels = np.array(val_labels)
    val_pred_labels = np.array(val_pred_labels)

    # val_labels가 문자열이라면, 이를 정수로 매핑
    val_labels_encoded = le.fit_transform(val_labels)
  
    # 성능 지표 계산
    acc = accuracy_score(val_labels_encoded, val_pred_labels)
    pre = precision_score(val_labels_encoded, val_pred_labels)
    rec = recall_score(val_labels_encoded, val_pred_labels)
    f1 = f1_score(val_labels_encoded, val_pred_labels)
    auc = roc_auc_score(val_labels_encoded, val_predictions[:, 1])

    # 성능 지표를 리스트에 추가
    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)
    auc_list.append(auc)

    # 성능 지표 출력
    print(f"Fold {fold+1} Accuracy: {acc}")
    print(f"Fold {fold+1} Precision: {pre}")
    print(f"Fold {fold+1} Recall: {rec}")
    print(f"Fold {fold+1} F1 Score: {f1}")

    # Confusion Matrix
    cm = confusion_matrix(val_labels_encoded, val_pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Fold {fold+1} Confusion Matrix')
    plt.savefig(f"{sub_dir}/{model_name}_{fold+1}_confusion_matrix.png")
    
    # ROC Curve
    val_predictions = np.argmax(val_predictions, axis=1)
    fpr, tpr, _ = roc_curve(val_labels_encoded, val_predictions)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Fold {fold+1} ROC Curve')
    plt.savefig(f"{sub_dir}/{model_name}_fold_{fold+1}_roc_curve.png")

    # 폴드별 히스토리 데이터 로드
    history = np.load(f"{history_dir}/history_fold_{fold+1}.npy", allow_pickle='TRUE').item()

    # 히스토리 데이터 저장
    pd.DataFrame(history).to_csv(f'{sub_dir}/{model_name}_fold_{fold+1}_history.csv')
    # performance.py

folds = range(1, n_folds + 1)
metrics = [acc_list, pre_list, rec_list, f1_list]
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for metric, label in zip(metrics, metric_labels):
    plt.figure(figsize=(10, 7))
    plt.plot(folds, metric, marker='o')
    plt.title(f'{model_name} {label} by Fold')
    plt.xlabel('Fold')
    plt.ylabel(label)
    plt.xticks(folds)
    plt.grid(True)
    plt.savefig(f"{sub_dir}/{model_name}_{label}_by_fold.png")
    plt.show()

