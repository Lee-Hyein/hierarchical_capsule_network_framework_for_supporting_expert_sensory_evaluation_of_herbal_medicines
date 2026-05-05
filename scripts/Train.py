import os
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime
from treelib import Tree


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TODAY = datetime.now().strftime("%Y-%m-%d")

# Add H-CapsNet metrics utilities to the import path so we can reuse hierarchical metrics.
HIERARCHY_UTILS_PATH = PROJECT_ROOT / "H-CapsNet" / "src"
if str(HIERARCHY_UTILS_PATH) not in sys.path:
    sys.path.append(str(HIERARCHY_UTILS_PATH))

import metrics as hierarchy_metrics  # noqa: E402


def parse_args(argv):
    model = argv[1] if len(argv) > 1 else "Recurrent"
    epochs = int(argv[2]) if len(argv) > 2 else 20
    train_batch = int(argv[3]) if len(argv) > 3 else 128
    val_batch = int(argv[4]) if len(argv) > 4 else train_batch
    data_dir = Path(argv[5]).expanduser().resolve() if len(argv) > 5 else None
    return model, epochs, train_batch, val_batch, data_dir


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def build_dataframe(dataset_root: Path) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    for master_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        for sub_dir in sorted([p for p in master_dir.iterdir() if p.is_dir()]):
            for article_dir in sorted([p for p in sub_dir.iterdir() if p.is_dir()]):
                image_paths = sorted([p for p in article_dir.iterdir() if is_image_file(p)])
                for image_path in image_paths:
                    records.append(
                        {
                            "filepath": image_path.relative_to(dataset_root).as_posix(),
                            "masterCategory": master_dir.name,
                            "subCategory": sub_dir.name,
                            "articleType": article_dir.name,
                        }
                    )
    if not records:
        raise ValueError(f"No images found under {dataset_root}")
    return pd.DataFrame.from_records(records)


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    encoders: Dict[str, Dict[str, int]] = {}
    encoded = df.copy()
    for column in ["masterCategory", "subCategory", "articleType"]:
        categories = sorted(encoded[column].unique())
        encoder = {name: idx for idx, name in enumerate(categories)}
        encoders[column] = encoder
        encoded[column] = encoded[column].map(encoder).astype(int)
    return encoded, encoders


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in df.groupby(label_col):
        indices = group.index.to_list()
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        # Adjust if splits would exhaust the bucket
        while n_test + n_val >= n and n_val > 1:
            n_val -= 1
        while n_test + n_val >= n and n_test > 1:
            n_test -= 1
        if n_test + n_val >= n:
            # Fallback to ensure at least one sample ends up in train
            n_val = max(0, min(n_val, n - 2))
            n_test = max(1, min(n_test, n - n_val - 1))

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test : n_test + n_val])
        train_idx.extend(indices[n_test + n_val :])

    def shuffle(df_subset: pd.DataFrame) -> pd.DataFrame:
        return df_subset.sample(frac=1, random_state=seed).reset_index(drop=True)

    return (
        shuffle(df.loc[train_idx]),
        shuffle(df.loc[val_idx]),
        shuffle(df.loc[test_idx]),
    )


def add_one_hot(df: pd.DataFrame, column: str, num_classes: int) -> None:
    one_hot = to_categorical(df[column].values, num_classes=num_classes)
    df[column + "OneHot"] = one_hot.tolist()


def save_class_order(encoders: Dict[str, Dict[str, int]], label: str) -> None:
    """Persist the class ordering (sorted folder names) used for training."""
    class_order_path = PROJECT_ROOT / "weights" / f"{label}_class_order_{TODAY}.json"
    class_order_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "masterCategory": [name for name, idx in sorted(encoders["masterCategory"].items(), key=lambda x: x[1])],
        "subCategory": [name for name, idx in sorted(encoders["subCategory"].items(), key=lambda x: x[1])],
        "articleType": [name for name, idx in sorted(encoders["articleType"].items(), key=lambda x: x[1])],
    }
    class_order_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved class ordering to {class_order_path}")


def _invert_encoder(encoder: Dict[str, int]) -> Dict[int, str]:
    return {idx: label for label, idx in encoder.items()}


def build_hierarchy_tree(
    encoded_df: pd.DataFrame,
    encoders: Optional[Dict[str, Dict[str, int]]] = None,
) -> Optional[Tree]:
    if encoded_df.empty:
        return None

    tree = Tree()
    tree.create_node("root", "root")

    label_maps: Dict[str, Dict[int, str]] = {}
    if encoders:
        for column, mapping in encoders.items():
            label_maps[column] = _invert_encoder(mapping)

    def label_name(column: str, idx: int) -> str:
        column_map = label_maps.get(column)
        if column_map is not None and idx in column_map:
            return str(column_map[idx])
        return str(idx)

    seen_sub_nodes: Dict[int, str] = {}
    seen_article_nodes: Dict[int, str] = {}

    for master_idx, master_group in encoded_df.groupby('masterCategory'):
        master_id = f"L0_{int(master_idx)}"
        if not tree.contains(master_id):
            tree.create_node(label_name('masterCategory', int(master_idx)), master_id, parent='root')

        for sub_idx, sub_group in master_group.groupby('subCategory'):
            sub_id = f"L1_{int(sub_idx)}"
            parent_id = master_id

            if sub_idx in seen_sub_nodes:
                if seen_sub_nodes[sub_idx] != parent_id:
                    raise ValueError(
                        f"Sub-category {sub_idx} assigned to multiple masters: "
                        f"{seen_sub_nodes[sub_idx]} vs {parent_id}"
                    )
            else:
                tree.create_node(label_name('subCategory', int(sub_idx)), sub_id, parent=parent_id)
                seen_sub_nodes[int(sub_idx)] = parent_id

            for article_idx in sub_group['articleType'].unique():
                article_id = f"L2_{int(article_idx)}"
                sub_parent_id = sub_id

                if article_idx in seen_article_nodes:
                    if seen_article_nodes[article_idx] != sub_parent_id:
                        raise ValueError(
                            f"Article class {article_idx} assigned to multiple sub-categories: "
                            f"{seen_article_nodes[article_idx]} vs {sub_parent_id}"
                        )
                else:
                    tree.create_node(
                        label_name('articleType', int(article_idx)),
                        article_id,
                        parent=sub_parent_id,
                    )
                    seen_article_nodes[int(article_idx)] = sub_parent_id

    return tree


def _stack_one_hot(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(df[column].to_numpy())


def evaluate_hierarchical_metrics(
    model,
    dataframe: pd.DataFrame,
    split_name: str,
    label: str,
    hierarchy_tree: Optional[Tree],
    output_dir: Path,
):
    if hierarchy_tree is None or dataframe.empty:
        print(f"Skipping hierarchical metrics for split '{split_name}' (tree or data unavailable).")
        return

    multi_input_model = isinstance(model.inputs, list) and len(model.inputs) > 1

    if multi_input_model:
        base_generator = val_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=direc,
            x_col="filepath",
            y_col=['masterCategoryOneHot', 'subCategoryOneHot', 'articleTypeOneHot'],
            target_size=target_size,
            batch_size=val_batch,
            class_mode='multi_output',
            shuffle=False,
        )
        steps = math.ceil(base_generator.n / base_generator.batch_size)
        sample_count = base_generator.n
        base_generator.reset()
        preds_accum: Optional[List[List[np.ndarray]]] = None
        collected = 0

        for _ in range(steps):
            batch_x, batch_y = next(base_generator)
            batch_inputs = [batch_x, batch_y[0], batch_y[1]]
            batch_preds = model.predict_on_batch(batch_inputs)
            if not isinstance(batch_preds, list):
                batch_preds = [batch_preds]
            if preds_accum is None:
                preds_accum = [[] for _ in range(len(batch_preds))]
            for idx, pred in enumerate(batch_preds):
                preds_accum[idx].append(pred)
            collected += batch_preds[0].shape[0]
            if collected >= sample_count:
                break

        if preds_accum is None:
            return
        predictions = [np.concatenate(chunks, axis=0)[:sample_count] for chunks in preds_accum]
    else:
        generator = val_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=direc,
            x_col="filepath",
            y_col=['masterCategoryOneHot', 'subCategoryOneHot', 'articleTypeOneHot'],
            target_size=target_size,
            batch_size=val_batch,
            class_mode='multi_output',
            shuffle=False,
        )

        steps = math.ceil(generator.n / generator.batch_size)
        generator.reset()
        predictions = model.predict(generator, steps=steps, verbose=1)
        sample_count = generator.n

    predictions = [pred[:sample_count] for pred in predictions]

    y_true_arrays = [
        _stack_one_hot(dataframe, 'masterCategoryOneHot'),
        _stack_one_hot(dataframe, 'subCategoryOneHot'),
        _stack_one_hot(dataframe, 'articleTypeOneHot'),
    ]
    y_true_indices = [np.argmax(arr, axis=1) for arr in y_true_arrays]

    top1 = hierarchy_metrics.get_top_k_taxonomical_accuracy(y_true_indices, predictions, k=1)
    top2 = hierarchy_metrics.get_top_k_taxonomical_accuracy(y_true_indices, predictions, k=2)
    top5 = hierarchy_metrics.get_top_k_taxonomical_accuracy(y_true_indices, predictions, k=5)
    harmonic_k1 = hierarchy_metrics.get_h_accuracy(y_true_indices, predictions, k=1)
    harmonic_k2 = hierarchy_metrics.get_h_accuracy(y_true_indices, predictions, k=2)
    harmonic_k5 = hierarchy_metrics.get_h_accuracy(y_true_indices, predictions, k=5)

    h_measurements, consistency, exact_match = hierarchy_metrics.hmeasurements(
        y_true_arrays.copy(), predictions, hierarchy_tree
    )

    metrics_row = {
        'model': label,
        'split': split_name,
        'epochs': epochs,
        'hierarchical_precision': h_measurements[0],
        'hierarchical_recall': h_measurements[1],
        'hierarchical_f1': h_measurements[2],
        'consistency': consistency,
        'exact_match': exact_match,
        'harmonic_accuracy_k1': harmonic_k1,
        'harmonic_accuracy_k2': harmonic_k2,
        'harmonic_accuracy_k5': harmonic_k5,
    }

    for level, value in enumerate(top1):
        metrics_row[f'level_{level}_top1'] = value
    for level, value in enumerate(top2):
        metrics_row[f'level_{level}_top2'] = value
    for level, value in enumerate(top5):
        metrics_row[f'level_{level}_top5'] = value

    metrics_path = output_dir / f"{label}_{split_name}_{epochs}_epochs_{TODAY}_hierarchical_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)

    print(
        f"Hierarchical metrics ({split_name}) -> "
        f"hP: {h_measurements[0]:.4f}, hR: {h_measurements[1]:.4f}, hF1: {h_measurements[2]:.4f}, "
        f"Consistency: {consistency:.4f}, Exact Match: {exact_match:.4f}"
    )


def load_fashion_csv() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Path]:
    train_df = pd.read_csv("fashion_product_train.csv")
    val_df = pd.read_csv("fashion_product_validation.csv")
    test_df = pd.read_csv("fashion_product_test.csv")

    lblmapsub = {
        'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4,
        'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8,
        'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12,
        'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17,
        'Topwear': 18, 'Wallets': 19, 'Watches': 20
    }
    lblmaparticle = {
        'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4,
        'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8,
        'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12,
        'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16,
        'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21,
        'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24,
        'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27,
        'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32,
        'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35,
        'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39,
        'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44
    }
    lblmapmaster = {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3}

    for df in (train_df, val_df, test_df):
        df['masterCategory'].replace(lblmapmaster, inplace=True)
        df['subCategory'].replace(lblmapsub, inplace=True)
        df['articleType'].replace(lblmaparticle, inplace=True)

    class_counts = {
        'master': len(lblmapmaster),
        'sub': len(lblmapsub),
        'article': len(lblmaparticle),
    }

    for df in (train_df, val_df, test_df):
        add_one_hot(df, 'masterCategory', class_counts['master'])
        add_one_hot(df, 'subCategory', class_counts['sub'])
        add_one_hot(df, 'articleType', class_counts['article'])

    dataset_root = Path('../data/fashion-dataset/images/').resolve()
    return train_df, val_df, test_df, class_counts, dataset_root


model_type, epochs, batch, val_batch, data_dir = parse_args(sys.argv)
hierarchy_tree: Optional[Tree] = None

if data_dir is not None and data_dir.is_dir():
    dataset_root = data_dir
    raw_df = build_dataframe(dataset_root)
    encoded_df, encoders = encode_labels(raw_df)
    class_counts = {
        'master': len(encoders['masterCategory']),
        'sub': len(encoders['subCategory']),
        'article': len(encoders['articleType']),
    }
    # Persist class order to make evaluation reproducible
    save_class_order(encoders, model_type)

    train_df, val_df, test_df = stratified_split(
        encoded_df,
        label_col='articleType',
        val_ratio=DEFAULT_VAL_SPLIT,
        test_ratio=DEFAULT_TEST_SPLIT,
        seed=DEFAULT_SEED,
    )

    try:
        hierarchy_tree = build_hierarchy_tree(
            encoded_df[['masterCategory', 'subCategory', 'articleType']],
            encoders=encoders,
        )
    except ValueError as exc:
        print(f"Warning: could not build hierarchy tree from dataset: {exc}")
        hierarchy_tree = None

    for df in (train_df, val_df, test_df):
        add_one_hot(df, 'masterCategory', class_counts['master'])
        add_one_hot(df, 'subCategory', class_counts['sub'])
        add_one_hot(df, 'articleType', class_counts['article'])

    print(f"Loaded dataset from {dataset_root}")
    print(
        f"Images: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}; "
        f"classes (master/sub/article) = {class_counts['master']}/{class_counts['sub']}/{class_counts['article']}"
    )
else:
    train_df, val_df, test_df, class_counts, dataset_root = load_fashion_csv()
    print("Loaded default fashion product dataset CSVs.")

    combined_encoded = pd.concat(
        [
            train_df[['masterCategory', 'subCategory', 'articleType']],
            val_df[['masterCategory', 'subCategory', 'articleType']],
            test_df[['masterCategory', 'subCategory', 'articleType']],
        ],
        ignore_index=True,
    )

    encoders = {
        'masterCategory': {'Accessories': 0, 'Apparel': 1, 'Footwear': 2, 'Personal Care': 3},
        'subCategory': {
            'Bags': 0, 'Belts': 1, 'Bottomwear': 2, 'Dress': 3, 'Eyewear': 4,
            'Flip Flops': 5, 'Fragrance': 6, 'Headwear': 7, 'Innerwear': 8,
            'Jewellery': 9, 'Lips': 10, 'Loungewear and Nightwear': 11, 'Nails': 12,
            'Sandal': 13, 'Saree': 14, 'Shoes': 15, 'Socks': 16, 'Ties': 17,
            'Topwear': 18, 'Wallets': 19, 'Watches': 20
        },
        'articleType': {
            'Backpacks': 0, 'Belts': 1, 'Bra': 2, 'Briefs': 3, 'Capris': 4,
            'Caps': 5, 'Casual Shoes': 6, 'Clutches': 7, 'Deodorant': 8,
            'Dresses': 9, 'Earrings': 10, 'Flats': 11, 'Flip Flops': 12,
            'Formal Shoes': 13, 'Handbags': 14, 'Heels': 15, 'Innerwear Vests': 16,
            'Jackets': 17, 'Jeans': 18, 'Kurtas': 19, 'Kurtis': 20, 'Leggings': 21,
            'Lipstick': 22, 'Nail Polish': 23, 'Necklace and Chains': 24,
            'Nightdress': 25, 'Pendant': 26, 'Perfume and Body Mist': 27,
            'Sandals': 28, 'Sarees': 29, 'Shirts': 30, 'Shorts': 31, 'Socks': 32,
            'Sports Shoes': 33, 'Sunglasses': 34, 'Sweaters': 35,
            'Sweatshirts': 36, 'Ties': 37, 'Tops': 38, 'Track Pants': 39,
            'Trousers': 40, 'Tshirts': 41, 'Tunics': 42, 'Wallets': 43, 'Watches': 44
        }
    }
    try:
        hierarchy_tree = build_hierarchy_tree(combined_encoded, encoders=encoders)
    except ValueError as exc:
        print(f"Warning: could not build hierarchy tree from dataset: {exc}")
        hierarchy_tree = None

#----------get VGG16 pre-trained weights--------
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
try:
    weights_path = get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        WEIGHTS_PATH,
        cache_dir=str(PROJECT_ROOT),
        cache_subdir='weights'
    )
except Exception as exc:  # pragma: no cover - best effort fallback when offline
    print(f"Warning: could not retrieve pre-trained VGG16 weights: {exc}")
    weights_path = None


#----------globals---------
print(train_df.head())
direc = str(dataset_root)
for relative in ['history', 'plots', 'weights']:
    (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)
# 모든 모델을 128x128 입력으로 통일해 사용한다.
target_size = (128, 128)

#Do additional transformations to support BatchNorm, Featurewise center and scal so each feature roughly N(0,1)
#Try with and without rescale
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)
val_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)

def get_flow_from_dataframe(g, dataframe,image_shape=target_size,batch_size=batch):
    while True:
        x_1 = g.next()

        yield [x_1[0], x_1[1][0], x_1[1][1]], x_1[1]

def train_BCNN(label, model, cbks):
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=val_batch,
        class_mode='multi_output')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("../history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig("../plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')

    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("../weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")

    evaluate_hierarchical_metrics(
        model,
        val_df,
        'val',
        label,
        hierarchy_tree,
        PROJECT_ROOT / 'history',
    )
    evaluate_hierarchical_metrics(
        model,
        test_df,
        'test',
        label,
        hierarchy_tree,
        PROJECT_ROOT / 'history',
    )


def train_recurrent(label, model,cbks):
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True)
    train = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=['masterCategoryOneHot','subCategoryOneHot','articleTypeOneHot'],
        target_size=target_size,
        batch_size=val_batch,
        class_mode='multi_output')

    train_generator = get_flow_from_dataframe(train,dataframe=train_df,image_shape=target_size,batch_size=batch)
    val_generator = get_flow_from_dataframe(val,dataframe=val_df,image_shape=target_size,batch_size=val_batch)
    try:
        STEP_SIZE_TRAIN = train.n // train.batch_size
        STEP_SIZE_VALID = val.n // val.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("../history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)

        # summarize history for loss
        plt.plot(history.history['master_output_loss'])
        plt.plot(history.history['val_master_output_loss'])
        plt.plot(history.history['sub_output_loss'])
        plt.plot(history.history['val_sub_output_loss'])
        plt.plot(history.history['article_output_loss'])
        plt.plot(history.history['val_article_output_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train master', 'val master', 'train sub', 'val sub', 'train article', 'val article'], loc='upper left')
        plt.show()
        plt.savefig("../plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+"_loss.png", bbox_inches='tight')
    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("../weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")     
    evaluate_hierarchical_metrics(
        model,
        val_df,
        'val',
        label,
        hierarchy_tree,
        PROJECT_ROOT / 'history',
    )
    evaluate_hierarchical_metrics(
        model,
        test_df,
        'test',
        label,
        hierarchy_tree,
        PROJECT_ROOT / 'history',
    )
#def BCNN_train():

def train_baseline(label, model,cbks):
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True)
    '''label is masterCategory, subCategory, or, articleType'''
    y = label
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="filepath",
        y_col=y,
        target_size=target_size,
        batch_size=batch,
        class_mode='categorical')
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="filepath",
        y_col=y,
        target_size=target_size,
        batch_size=val_batch,
        class_mode='categorical')
    try:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size
        history = model.fit_generator(train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=epochs,
                            validation_data=val_generator,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("../history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'.csv',index=False)
    
        # plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig("../plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_loss.png', bbox_inches='tight')

    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("../weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+".h5")


if(model_type == 'Recurrent'):
    from RecurrentBranching import RecurrentTrain
    recurrent = RecurrentTrain(model_type)
    model = recurrent.model
    cbks = recurrent.cbks
    train_recurrent(model_type, model, cbks)
elif(model_type=='Condition'):
    from ConditionCNN import ConditionTrain
    condition = ConditionTrain(model_type, class_counts=class_counts, input_shape=(*target_size, 3))
    model = condition.model
    cbks = condition.cbks
    train_recurrent(model_type,model,cbks)
elif(model_type=='ConditionB'):
    from ConditionCNNB import ConditionTrain
    condition = ConditionTrain(model_type, input_shape=(*target_size, 3))
    model = condition.model
    cbks = condition.cbks
    train_recurrent(model_type,model,cbks)
elif(model_type=='ConditionPlus'):
    from ConditionCNNPlus import ConditionPlusTrain
    condition = ConditionPlusTrain(model_type, class_counts=class_counts, input_shape=(*target_size, 3))
    model = condition.model
    cbks = condition.cbks
    train_recurrent(model_type,model,cbks)
elif(model_type=='BCNN'):
    from BCNN import BCNN
    bcnn = BCNN(model_type, class_counts=class_counts, input_shape=(*target_size, 3))
    model = bcnn.model
    cbks = bcnn.cbks
    train_BCNN(model_type, model, cbks)
elif(model_type == 'articleType'):
    from articleType import ArticleType
    articletype = ArticleType(model_type)
    model = articletype.model
    cbks = articletype.cbks
    train_baseline(model_type, model, cbks)
elif(model_type == 'subCategory'):
    from subCategory import SubCategory
    subcategory = SubCategory(model_type)
    model = subcategory.model
    cbks = subcategory.cbks
    train_baseline(model_type,model,cbks)
else:
    #masterCategory
    from masterCategory import MasterCategory
    mastercategory = MasterCategory(model_type)
    model = mastercategory.model
    cbks = mastercategory.cbks
    train_baseline(model_type, model,cbks)
    
