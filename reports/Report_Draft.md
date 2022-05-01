# Brain Tomor Detection

**Team Members: Jiawei Liu, Juan Guerrero, Niloufar Nouri**

<br/>

## 1 Introduction


A brain tumor is commonly known as the growth of abnormal cells in the brain, some of which may lead to malignant cancer and eventually death. The usual method to detect the brain tumor is through the use of Magnetic Resonance Imaging (MRI) scans. From these scans, one can find out information about the abnormal tissue growth in the brain. According to the international agency for research on cancer (IARC), the mortality rate due to brain tumors is 76%. To prevent any extremely fatal situation, it is desirable to detect brain tumors as early as possible. In various research papers, the detection of brain tumors is done by applying Machine Learning and Deep Learning algorithms. 

Machine learning and deep learning approaches have become popular among researchers in medical fields, particularly convolutional neural networks (CNN), which can handle and analyze large amounts of complex image data and perform classification. When these algorithms are applied to the MRI images, the prediction of the brain tumor is done very fast and a higher accuracy helps in providing the treatment to the patients. These predictions also help the radiologist in making quick decisions. Therefore, we propose to leverage machine learning and deep learning knowledge to build a classification model using MRI images.  This project is an application project, which has two main contributions listed below:

- We develop classification models based on machine learning, and we will optimize model parameters based on evaluation metrics.
- We also apply deep learning techniques and try multiple neural network architectures to do the classification and compare their performance. The goal is to find the model which not only achieves good accuracy but also it will be able to generalize.

<br/>

## 2 Method 


In this project, we first perform exploratory data analysis (EDA) to understand and clean our dataset. Since our dataset is an image dataset, it is necessary to implement feature selection or dimensionality reduction to reduce the dimensionality of the input data so that we can efficiently apply our data with machine learning algorithms. In addition, image preprocessing is required in this project. We begin by splitting our data into training and testing sets (70% train set, 30% test set or 80% train set, 20% test) and feeding it into a machine learning model chosen by each member. Depending on the performance of each model, we will conduct further experiments which might improve the predictability of the algorithms. We can begin by applying a 10-fold cross-validation in order to introduce a validation set into the experiments with the hope of detecting overfitting thus generalizing the model to future unseen data. We can also apply the method of Ensemble, which usually produces a more accurate solution than a single model would. This has been the case in a number of machine learning competitions where the winning models come from Ensemble methods. Wealso use feature engineering and dimension reduction methods such as principal component analysis to see if we could get acceptable accuracy with fewer features. 

<br/>

## 3 Results  


### 3-1 Exploratory Data Analysis and Feature Extraction
<br/>
### 3-2 Classification with Machine Learning Algorithms
<br/>
### 3-2 Classification with Deep Learning Algorithms
<br/>
### 3-3 Image Segmentation

<br/>

## 4 Disscussion and Coclusions
