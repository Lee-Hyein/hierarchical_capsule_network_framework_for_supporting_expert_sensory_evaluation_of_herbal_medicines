import os
import sys
import csv
import tensorflow as tf
from src import MLmodel
import importlib
importlib.reload(MLmodel)


# GPU 사용 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 0 사용

# TensorFlow가 cuDNN 없이도 작동하도록 설정
# 일부 연산은 CPU로 fallback하지만 대부분은 GPU로 실행됨
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# cuDNN 사용 가능 여부 확인 및 경고
try:
    # TensorFlow가 cuDNN을 사용할 수 있는지 확인
    # 실제 연산을 시도하지 않고 설정만 확인
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available:
        print(f"✅ GPU 사용 가능: {len(gpu_available)}개")
        # cuDNN 연산이 필요한 경우를 위해 경고만 출력
        print("⚠️ cuDNN 관련 오류가 발생하면, 스크립트 실행 전에 다음 명령을 실행하세요:")
        print("   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")
except Exception as e:
    print(f"⚠️ GPU/cuDNN 확인 중 오류: {e}")

# GPU 메모리 성장 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 설정 완료: {len(gpus)}개의 GPU 사용 가능")
        print(f"✅ CUDA 사용 가능: {tf.test.is_built_with_cuda()}")
    except RuntimeError as e:
        print(f"⚠️ GPU 설정 오류: {e}")
else:
    print("⚠️ GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")

# Supporting Libraries:
    #Mathplot lib for ploting graphs

import matplotlib
import matplotlib.pyplot as plt
    # numpy and pandas
import numpy as np
import pandas as pd
    #import other libraries
import math
import random
from datetime import datetime
    # ML model, Dataset and evalution metrics
from src import datasets
from src import MLmodel
from src import metrics
from src import sysenv
    # For developind (reloades any python scripts)
import importlib


systeminfo = sysenv.systeminfo()
print(systeminfo)

## For Using Multiple GPUs
gpus = "1" ## Selecting Available gpus
gpugrowth = sysenv.gpugrowth(gpus = gpus) ## Limiting GPUS from OS environment
gpugrowth.memory_growth() #GPU memory growth

train_params = {"n_epochs" : 50,
                "batch_size": 4,  # OOM 방지를 위해 배치 크기 감소 (16 -> 2)
                "lr": 0.001, # Initial learning rate
                "lr_decay": 0.95, # Learning rate decay
                "decay_exe": 9, #learning rate dsecay execution epoch after
               }
model_params = {"P_Cap_Dim" : 8, # Primary Capsule Dimentionsßß
                "S_Cap_Dim" : 16, # Secondary Capsule Dimention
                "Reconstruction_LW" : 0.0005, # Decoder loss weight
                "class_loss" : MLmodel.MarginLoss(), ## Class prediction loss
                "reconstruction_loss" : 'mse'
               }

# Docker 컨테이너와 호스트 모두 동일한 경로 사용
DATA_ROOT = "/mnt/mydisk/hyein/paper_exp_0929/hierarchy_image_dataset_aug"

def scheduler(epoch):
    learning_rate_init = train_params["lr"]
    
    if epoch > train_params["decay_exe"]:
        learning_rate_init = train_params["lr"] * (train_params["lr_decay"] ** (epoch-9))
        
    return learning_rate_init

dataset = datasets.HerbalOSRDataset(DATA_ROOT, image_size=(128, 128), test_split=0.2, unknown_split=0.2)

input_shape = dataset['x_train'].shape[1:]
print('INPUT SHAPE:', input_shape, '\n')

print("TRAIN KNOWN: \r\n")
print(dataset['x_train'].shape)
print(dataset['y_train_fine'].shape)
print(dataset['y_train_medium'].shape)
print(dataset['y_train_coarse'].shape)

print("\nTEST KNOWN: \r\n")
print(dataset['x_test'].shape)
print(dataset['y_test_fine'].shape)
print(dataset['y_test_medium'].shape)
print(dataset['y_test_coarse'].shape)

fine_class = dataset['y_train_fine'].shape[1]
medium_class = dataset['y_train_medium'].shape[1]
coarse_class = dataset['y_train_coarse'].shape[1]

datasets.plot_sample_image(dataset['x_train'],
                           {'coarse':dataset['y_train_coarse'],
                            'medium':dataset['y_train_medium'],
                            'fine':dataset['y_train_fine']})

dataset['tree'].show()

def herbal_batch_generator(
    x_known,
    y_known_coarse,
    y_known_medium,
    y_known_fine,
    batch_size,
    seed=42,
):
    rng = np.random.default_rng(seed)
    num_known = x_known.shape[0]

    while True:
        known_indices = rng.choice(num_known, size=batch_size, replace=batch_size > num_known)
        x_batch = x_known[known_indices]
        y_coarse_batch = y_known_coarse[known_indices]
        y_medium_batch = y_known_medium[known_indices]
        y_fine_batch = y_known_fine[known_indices]

        permutation = rng.permutation(batch_size)
        x_batch = x_batch[permutation]
        y_coarse_batch = y_coarse_batch[permutation]
        y_medium_batch = y_medium_batch[permutation]
        y_fine_batch = y_fine_batch[permutation]

        inputs = [x_batch, y_coarse_batch, y_medium_batch, y_fine_batch]
        outputs = [
            y_coarse_batch,
            y_medium_batch,
            y_fine_batch,
            x_batch,
        ]

        yield inputs, outputs

initial_lw = MLmodel.initial_lw({"coarse": coarse_class,
                                 "medium": medium_class,
                                 "fine": fine_class})

lossweight = {'coarse_lw' : K.variable(value = initial_lw['coarse'], dtype="float32", name="coarse_lw"),
             'medium_lw' : K.variable(value = initial_lw['medium'], dtype="float32", name="medium_lw"),
             'fine_lw' : K.variable(value = initial_lw['fine'], dtype="float32", name="fine_lw"),
              'decoder_lw' : model_params['Reconstruction_LW']
             }

def get_compiled_model():
    model = MLmodel.HCapsNet_3_Level(
        input_shape,
        coarse_class,
        medium_class,
        fine_class,
    )

    losses = [
        model_params["class_loss"],
        model_params["class_loss"],
        model_params["class_loss"],
        model_params["reconstruction_loss"],
    ]

    loss_weights_list = [
        lossweight['coarse_lw'],
        lossweight['medium_lw'],
        lossweight['fine_lw'],
        lossweight['decoder_lw'],
    ]

    model.compile(
        optimizer='adam',
        loss=losses,
        loss_weights=loss_weights_list,
        metrics={
            'Fine_prediction_output_layer': 'accuracy',
            'Medium_prediction_output_layer': 'accuracy',
            'Coarse_prediction_output_layer': 'accuracy'
        }
    )
    return model

tf.keras.backend.clear_session() ## clear session

# GPU 모드로 실행 (단일 GPU 사용 - OOM 방지)
# MirroredStrategy 제거하여 메모리 절약
try:
    # 단일 GPU 사용
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # GPU 메모리 성장 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용: {len(gpus)}개")
        print(f"⚠️ 현재 배치 크기: {train_params['batch_size']}")
        print("⚠️ 여전히 OOM이 발생하면 배치 크기를 더 줄이세요.")
    
    # 모델 생성
    model = get_compiled_model()
    print("✅ GPU 모드로 모델 생성 완료")
except Exception as e:
    print(f"⚠️ GPU 전략 설정 실패: {e}")
    print("⚠️ CPU 모드로 전환합니다...")
    # CPU 모드 fallback
    model = get_compiled_model()
    print("✅ CPU 모드로 모델 생성 완료")

### Log directory
directory = sysenv.log_dir(dataset["name"]+'/'+model.name)

model.summary()
try:
    keras.utils.plot_model(model, to_file = directory+"/H-CapsNet.png", show_shapes=True)
except Exception as plot_err:
    print("Warning: Failed to plot model diagram. Install Graphviz (dot) to enable this feature.")
    print("Plot error:", plot_err)

tb = keras.callbacks.TensorBoard(directory+'./tb_logs')
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(directory, f"log_{timestamp}.csv")
log = keras.callbacks.CSVLogger(log_path, append=False)


checkpoint = keras.callbacks.ModelCheckpoint(
                                                directory+'/epoch_best.weights.h5',
                                                monitor='Fine_prediction_output_layer_accuracy',  # train metric
                                                mode='max',
                                                save_best_only=True, save_weights_only=True, verbose=1
                                            )

change_lw = MLmodel.LossWeightsModifier(lossweight,
                                        initial_lw,
                                        directory = directory)

lr_decay = keras.callbacks.LearningRateScheduler(scheduler)

# TensorFlow Dataset으로 변환 (Keras 3.x 호환)
batch_size = train_params["batch_size"]
def create_tf_generator():
    """Generator를 TensorFlow 호환 형식으로 변환"""
    gen = herbal_batch_generator(
        dataset['x_train'],
        dataset['y_train_coarse'],
        dataset['y_train_medium'],
        dataset['y_train_fine'],
        batch_size=batch_size,
    )
    for inputs, outputs in gen:
        # 리스트를 튜플로 변환
        inputs_tuple = tuple(inputs)
        outputs_tuple = tuple(outputs)
        yield (inputs_tuple, outputs_tuple)

# output_signature 정의 (튜플 구조)
input_signature = (
    tf.TensorSpec(shape=(batch_size, *input_shape), dtype=tf.float32),  # x_batch
    tf.TensorSpec(shape=(batch_size, coarse_class), dtype=tf.float32),  # y_coarse
    tf.TensorSpec(shape=(batch_size, medium_class), dtype=tf.float32),  # y_medium
    tf.TensorSpec(shape=(batch_size, fine_class), dtype=tf.float32),    # y_fine
)
output_signature = (
    tf.TensorSpec(shape=(batch_size, coarse_class), dtype=tf.float32),  # y_coarse
    tf.TensorSpec(shape=(batch_size, medium_class), dtype=tf.float32),  # y_medium
    tf.TensorSpec(shape=(batch_size, fine_class), dtype=tf.float32),    # y_fine
    tf.TensorSpec(shape=(batch_size, *input_shape), dtype=tf.float32),  # x_batch
)

training_dataset = tf.data.Dataset.from_generator(
    create_tf_generator,
    output_signature=(
        input_signature,
        output_signature,
    )
)

steps_per_epoch = math.ceil(dataset['x_train'].shape[0] / train_params["batch_size"])

val_inputs = [
    dataset['x_test'],
    dataset['y_test_coarse'],
    dataset['y_test_medium'],
    dataset['y_test_fine'],
]

val_outputs = [
    dataset['y_test_coarse'],
    dataset['y_test_medium'],
    dataset['y_test_fine'],
    dataset['x_test'],
]

# Validation 데이터를 배치로 처리하기 위해 Generator 사용 (OOM 방지)
# from_tensor_slices는 전체 데이터를 GPU 메모리에 올리려고 시도하므로 generator 사용
val_batch_size = 2  # OOM 방지를 위해 작은 배치 크기 사용

def val_data_generator():
    """Validation 데이터를 배치 단위로 생성 (CPU에서 처리)"""
    num_samples = len(val_inputs[0])
    for i in range(0, num_samples, val_batch_size):
        end_idx = min(i + val_batch_size, num_samples)
        
        # NumPy 배열로 슬라이싱 (CPU에서 처리)
        x_batch = val_inputs[0][i:end_idx]
        coarse_batch = val_inputs[1][i:end_idx]
        medium_batch = val_inputs[2][i:end_idx]
        fine_batch = val_inputs[3][i:end_idx]
        
        inputs = [x_batch, coarse_batch, medium_batch, fine_batch]
        outputs = [
            coarse_batch,
            medium_batch,
            fine_batch,
            x_batch
        ]
        yield tuple(inputs), tuple(outputs)

# Validation dataset 생성 (동적 배치 크기 지원)
val_dataset_final = tf.data.Dataset.from_generator(
    val_data_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),  # x_batch (동적)
            tf.TensorSpec(shape=(None, coarse_class), dtype=tf.float32),  # y_coarse (동적)
            tf.TensorSpec(shape=(None, medium_class), dtype=tf.float32),  # y_medium (동적)
            tf.TensorSpec(shape=(None, fine_class), dtype=tf.float32),    # y_fine (동적)
        ),
        (
            tf.TensorSpec(shape=(None, coarse_class), dtype=tf.float32),  # y_coarse (동적)
            tf.TensorSpec(shape=(None, medium_class), dtype=tf.float32),  # y_medium (동적)
            tf.TensorSpec(shape=(None, fine_class), dtype=tf.float32),    # y_fine (동적)
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),  # x_batch (동적)
        ),
    )
)

# OOM 방지를 위해 validation_freq를 설정하여 매 epoch마다 실행하지 않음
history = model.fit(
    training_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=train_params["n_epochs"],
    validation_data=val_dataset_final,
    validation_steps=min(10, math.ceil(dataset['x_test'].shape[0] / val_batch_size)),  # 최대 10 스텝만 실행
    validation_freq=1,  # 5 epoch마다 validation 실행 (OOM 방지)
    callbacks=[tb, log, checkpoint, lr_decay, change_lw],
    verbose=1,
)
model_save_dir = str(directory+'/trained_model.h5')
try:
    model.save_weights(model_save_dir)
    print('Trained model saved to = ', model_save_dir)
except:
    print('Model Wight is not saved')

# OOM 방지를 위해 evaluate도 작은 배치로 처리
evaluation_results = model.evaluate(
    val_inputs,
    val_outputs,
    batch_size=2,  # OOM 방지를 위해 최소 배치 크기 사용
    verbose=1,
)
print("Evaluation results:")
for name, value in zip(model.metrics_names, evaluation_results):
    print(f"  {name}: {value}")

predictions = model.predict(val_inputs, batch_size=2, verbose=1)  # OOM 방지를 위해 최소 배치 크기 사용

true_label = [dataset['y_test_coarse'], dataset['y_test_medium'], dataset['y_test_fine']]
coarse_pred, medium_pred, fine_pred = predictions[0], predictions[1], predictions[2]
pred_label = [coarse_pred, medium_pred, fine_pred]

metrics.lvl_wise_metric(true_label, pred_label)

h_measurements, consistency, exact_match = metrics.hmeasurements(
    true_label,
    pred_label,
    dataset['tree']
)
print('\nHierarchical Precision =', h_measurements[0],
      '\nHierarchical Recall =', h_measurements[1],
      '\nHierarchical F1-Score =', h_measurements[2],
      '\n Consistency = ', consistency,
      '\n Exact Match = ', exact_match,
     )
metrics.plot_confusion_matrix(true_label,
                                pred_label,
                                dataset['tree'],
                                directory)
