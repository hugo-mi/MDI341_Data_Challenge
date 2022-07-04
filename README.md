# Data Challenge 2 : Challenge Large Scale Machine Learning (Performance and Fairness Optimization)

##### Author : Hugo Michel, June 2022

### Data Challenge Description : Face Recognition

In recent years, face recognition systems have achieved extremely high levels of performance, opening the door to a wider range of applications where reliability levels were previously prohibitive to consider automation. This is mainly due to the adoption of deep learning techniques in computer vision. The most widely adopted paradigm is to train a $f: \mathcal{X} \rightarrow \mathbb{R}^d$ which, from a given image $im \in \mathcal{X}$, extracts a feature vector $z \in \mathbb{R}^d$ which synthesizes the relevant features of $im$. 

The recognition phase then consists, from two images $im_1, im_2$, in predicting whether or not they correspond to the same identity. This is done from the extracted features $z_1, z_2$.

### Goal

In this data challenge, the goal is to train a machine learning model that, given a vector $[z_1, z_2]$ consisting of the concatenation of two patterns $z_1$ and $z_2$, predicts whether or not these two images match the same identity.

In addition, special attention will be paid to the fairness of the prediction model with respect to the gender of the individuals. This means that the model must be as efficient as possible, regardless of the gender of the person. 

### Training Data

The train set consists of two files ``train_data.npy`` and ```train_labels.txt```


The ```train_data.npy``` file contains one observation per line, which consists of the concatenation of two templates, each of **dimension 48**
    
The file ```train_labels.npy``` contains two classes labeled per line that indicate whether the image pair matches the same identity: 
    
- ```1``` => image pairs belonging to the same identity
- ```0``` => image pairs not belonging to the same identity

In total, there are 267508 observations.

### Performance

For the evaluation of the performance of the models, the idea is to minimize the sum of the rate of **false positives rate** ```FPR``` and the rate of **false negatives rate** ```FNR```. The performance score of the model is calculated using the following equation.

$score = 1 - (FPR + FNR)$

### Fairness Peformance criterion

Moreover, we want the prediction to be as fair as possible with respect to the gender attribute. In our case, we want to make the ratios 


$$ BFPR := \frac{\max(FPR(male),FPR(female))}{\mathrm{GeomMean}(FPR(male),FPR(female))} \geq 1 $$
and


$$ BFNR := \frac{\max(FNR(male),FNR(female))}{\mathrm{GeomMean}(FNR(male),FNR(female))} \geq 1 $$


as close to 1 as possible, which corresponds to its minimum value. </br>

## Overall Strategy

As previously explained, since this new data challenge uses the same data as the previous one, there is no need to perform a data analysis as this has already been done extensively for the first data challenge. It is possible to review the exploratory analysis I made for the first data challenge by consulting directly my github, the notebook is available 
[here](https://github.com/hugo-mi/MDI343_Data_Challenge_Face_Recognition/blob/main/MDI_343_Data_Challenge_Hugo_Michel.ipynb "here") 

Moreover, it will be interesting to perform a deeper exploratory analysis of the columns ``X[8]`` and ``X[56]`` which are crucial columns for the fairness metric.

Moreover, through the experience of this first data challenge, we know that the generation of new features can improve the performance of the model. In addition, dimension reduction, in order to simplify the classification model by applying different feature selection models, does not seem to be a good option for this data challenge at least as far as only the performance metric is concerned.

This is why the strategy I will opt for is to reuse the model with the best performance of the first data challenge as a baseline model. 

The real challenge of this data challenge is to find the proper trade-off between the two metrics. To achieve this, one of possible good approach is to train the model on a **custom metric**.

## The strategic approach

The ultimate challenge is to find the best trade-off between performance and equity. 

This is a very active research topic. When we try to maximize the performance of a model, this is done at the expense of the fairness of the model.

The control of the fairness of the model can be done :
- At the pre-processing level
- At the training level of the model solving a constrained optimization problem 
- At the post-processing level by adjusting the prediction threshold

### Pre-processing level
Hide from the model the genders relative features and gender relative correlated features with the gender of the person

### Training level
Solving a constrained optimization problem
The last approach is more difficult to implement because it requires to modify the objective function of the model. Indeed, the ideal would be to inject in the loss function of the model the performance and fairness metrics and thus train the model to find natively the best trade-off between performance and fairness.
But the problem is that the Fairness and Performance Score metrics are not differentiable and therefore it is not possible to include these metrics in the loss function of the model because it will be difficult to minimize this loss function since optimization algorithms such as gradient descent (1st optimization order) cannot be applied in this case.

The only approach that can be applied in our case is **free gradient optimization**. The goal would be to define a custom metric combined with the use of optimization tools such as ``optuna`` to find the best combination of model hyperparameters.

### Post-processing level
Adjusting the prediction threshold
