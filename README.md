# Comparison of Scalers in Python by Creating Real-World Synthetic Data

In this post, I have reviewed the performances of three widely used scalers in feature engineering, namely `StandardScaler`, `MinMaxScaler`, and `RobustScaler`, available in the `sklearn` library. The scaling process is crucial in preparing real-world datasets for machine learning models as it standardizes the range of features, making them comparable on the same scale.

## Dataset

I created a synthetic dataset with 100,000 samples, each containing 10 features. The feature distributions are shown below.

2. Scalers

### 2.1. StandardScaler

Main features about `StandardScaler`:
- It uses a mean of 0 and a standard deviation of 1 to normalize or scale the features of a dataset.
- It does not affect the distribution's shape, but it brings all features to the same scale.
- However, it is sensitive to outliers since it relies on the mean and standard deviation, both of which can be impacted by outliers.

When compared with Figure-1, it is seen that there is no distortion from the shapes of the histogram graphs, but the mean value of the histograms is zero. Because of the sensitivity to outliers, `StandardScaler` couldn't cope with `feature_9` and `feature_10`. As can be understood from the boxplot, the shapes of boxes in each feature are similar to each other. The shape of the box in `Feature_9` and `Feature_10` is different because they have extreme outliers.

### 2.2. MinMaxScaler

Main features about `MinMaxScaler`:
- It is used to scale a dataset's features to a specific range, typically between 0 and 1.
- It is useful when the features' ranges differ, and you want to bring them all inside the same range.
- It is less sensitive to outliers than other scaling strategies such as `StandardScaler`. However, the presence of severe outliers in the dataset can still impact the scaling process.

As can be seen from Figure-3, the distribution of features has changed. The histogram is between 0-1, and the accretion is not changed significantly in `feature_9` and `feature_10`.

### 2.3. RobustScaler

Main features about `RobustScaler`:
- It uses the median and the interquartile range (IQR) to scale the features.
- It makes the features robust to outliers.
- It does not use mean and standard deviation, making it more resistant to outliers.

The distribution of features is normally distributed, and there are three features that have outliers, which is why `StandardScaler` is the best for this example. However, if we increase the number of features that have outliers, it may affect its performance. Scaling the data to a specific range [0,1] makes `MinMaxScaler` worse because the presence of extreme outliers and highly skewed distributions affect `MinMaxScaler` performance. While `RobustScaler` is designed to be more resistant to the influence of outliers, it doesn't eliminate outliers completely. Its primary goal is to reduce the impact of outliers by adjusting the scaling in a more robust manner.

In conclusion, due to the fact that the dataset generally has a normal distribution, `StandardScaler` performed well. Having outliers makes `RobustScaler`'s performance better. Despite the presence of outliers, it handled the data well and reduced the effect of outliers. On the other hand, `MinMaxScaler` performed poorly when trying to fit a dataset with extreme outliers between 0 and 1.
