import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Đọc dữ liệu
df = pd.read_csv("price_smp.csv")
print("Dataframe shape:", df.shape)
print(df.head())

feats = ['5', '6']

# Chuyển đổi dữ liệu
transformer = PowerTransformer()
X = transformer.fit_transform(df[feats])
print("Transformed X shape:", X.shape)

# PCA
pca = PCA(n_components=2, random_state=1)
PCA_ds = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])
print("PCA Dataset shape:", PCA_ds.shape)
print(PCA_ds.describe().T)

# Vẽ biểu đồ 2D
plt.figure(figsize=(10, 8))
plt.scatter(PCA_ds["PC1"], PCA_ds["PC2"], c="maroon", marker="o")
plt.title("A 2D Projection Of Data In The Reduced Dimension")
plt.show()

# Elbow Method để xác định số cụm cần tạo
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(random_state=23), k=(4, 12))
Elbow_M.fit(X)
Elbow_M.show()

# Kiểm tra silhouette score với KMeans
optimal_k = 5  # Thay thế bằng số cụm tối ưu từ Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=1).fit(X)
labels = kmeans.labels_
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score for KMeans: {silhouette_avg}')

# Gaussian Mixture Model với nhiều giá trị của n_components
for n_components in range(2, 10):  # Thử nghiệm với các giá trị khác nhau của n_components
    BGM = GaussianMixture(n_components=2, covariance_type='full', random_state=1, n_init=15)
    preds = BGM.fit_predict(X)
    print(f"Preds with n_components={n_components}: {np.unique(preds, return_counts=True)}")
    if len(np.unique(preds)) > 1:  # Nếu có nhiều hơn 1 cụm được tìm thấy
        break

df["Clusters"] = preds

# Kiểm tra phân phối của các cụm
print("Cluster distribution:")
print(df["Clusters"].value_counts())

# Tính toán xác suất dự đoán
pp = BGM.predict_proba(df[feats])
df_new = pd.DataFrame(df[feats])
num_cols = pp.shape[1]
df_new[[f'predict_proba_{i}' for i in range(num_cols)]] = pp
df_new['preds'] = preds
df_new['predict_proba'] = np.max(pp, axis=1)
df_new['predict'] = np.argmax(pp, axis=1)

# Kiểm tra kết quả phân cụm và xác suất dự đoán
print("df_new shape:", df_new.shape)
print(df_new.head())

# Tạo mảng chứa các chỉ mục của mẫu được chọn
train_index = np.array([], dtype=int)
for n in range(optimal_k):  # Sửa 7 thành optimal_k
    n_inx = df_new[(df_new.preds == n) & (df_new.predict_proba > 0.68)].index
    train_index = np.concatenate((train_index, n_inx))


print(preds)
# Kiểm tra train_index
print("Train index length:", len(train_index))
print("Train index:", train_index)

# Kiểm tra y
X_new = df_new.loc[train_index][feats]
y = df_new.loc[train_index]['preds']
print("Y length:", len(y))
print("Y values:", y.unique())

# Đảm bảo rằng không có lỗi do rỗng dữ liệu
if len(train_index) == 0 or len(y) == 0:
    raise ValueError("Train index hoặc y rỗng. Kiểm tra lại logic chọn dữ liệu.")

# Huấn luyện mô hình LightGBM
params_lgb = {
    'learning_rate': 0.06,
    'objective': 'multiclass',
    'boosting': 'gbdt',
    'n_jobs': -1,
    'verbosity': -1,
    'num_classes': optimal_k
}

model_list = []
gkf = StratifiedKFold(n_splits=11)
for fold, (train_idx, valid_idx) in enumerate(gkf.split(X_new, y)):
    tr_dataset = lgb.Dataset(X_new.iloc[train_idx], y.iloc[train_idx], feature_name=feats)
    vl_dataset = lgb.Dataset(X_new.iloc[valid_idx], y.iloc[valid_idx], feature_name=feats)

    model = lgb.train(params=params_lgb,
                      train_set=tr_dataset,
                      valid_sets=vl_dataset,
                      num_boost_round=5000,
                      callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False), lgb.log_evaluation(period=200)])

    model_list.append(model)

# Huấn luyện mô hình Decision Tree
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_new, y)

# Dự đoán với mô hình Decision Tree
y_pred = clf.predict(X_new)
accuracy = accuracy_score(y, y_pred)
print(f"Decision Tree Accuracy: {accuracy}")
print("Classification Report for Decision Tree:")
print(classification_report(y, y_pred))

# Vẽ cây quyết định
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 10))
plot_tree(clf, filled=True, feature_names=feats, class_names=[str(i) for i in range(optimal_k)], rounded=True)
plt.title("Decision Tree")
plt.show()
