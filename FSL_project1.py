import numpy as np
import scipy.io
from scipy.stats import multivariate_normal
#print("hello")
#print("-------------------")

#Training data

train_data = scipy.io.loadmat("train_data.mat")
#print(train_data['label'][0][0])
train_data_label = train_data['label'].flatten()
#print("Trained data label: " + str(train_data_label))
#print("Training data length: " + str())
transformed_train_data = []
for i in train_data['data']:
    flattened_trained_data = i.flatten()
    list = []
    list.append(np.mean(flattened_trained_data))
    list.append(np.std(flattened_trained_data))
    transformed_train_data.append(list) #in the form [mi,si]
#print(transformed_data)
Mean_train = np.mean(transformed_train_data,axis=0) #[M1,M2]
SD_train = np.std(transformed_train_data,axis=0)    #[S1,S2]
print("[M1 M2]: " + str(Mean_train))
#print(Mean_train)
print("[S1 S2]: " + str(SD_train))

Y_transform_train = [] #Images in normalized form
for i in transformed_train_data:
    list = []
    list.append((i[0]-Mean_train[0])/SD_train[0])
    list.append((i[1]-Mean_train[1])/SD_train[1])
    Y_transform_train.append(list)

label_3=[] #all the normalized trained data labelled as 3
label_7=[] ##all the normalized trained data labelled as 7
training_data_3=[]
training_data_7=[]
for i in range(0,len(transformed_train_data)):

    if train_data_label[i] == 3:
        label_3.append(Y_transform_train[i])
        #training_data_3.append()
    elif train_data_label[i] == 7 :
        label_7.append(Y_transform_train[i])

Mean3 = np.mean(label_3,axis=0)
Mean7 = np.mean(label_7,axis=0)
print("Mu3(Mean3) " +str(Mean3)) #Mean3 - Mean vector for all label3 data (theta1 for label 3 data)
print("Mu7(Mean7) " +str(Mean7)) #Mean7 - Mean vector for all label7 data (theta1 for label 7 data)
#print(Mean3)
#print(Mean7)

sigma3 = np.zeros((2,2))
for data in label_3:
    sub = np.subtract(data,Mean3)
    sigma3 = np.add(sigma3, np.outer(sub,sub))
sigma3 = sigma3/len(label_3)

print("Sigma3 "+ str(sigma3)) #Sigma3 - Covariance vector for all label3 data (theta2 for label 3 data)
#print(sigma3)
sigma7 = np.zeros((2,2))
for data in label_7:
    sub = np.subtract(data,Mean7)
    sigma7 = np.add(sigma7, np.outer(sub,sub))
sigma7 = sigma7/len(label_7)

print("Sigma7 " +str(sigma7)) #Sigma3 - Covariance vector for all label7 data (theta2 for label 7 data)
#print(sigma7)
#Multivariate pdf calculation
# y_dummy = multivariate_normal.pdf(Y_transform_train[0], mean=Mean3, cov=sigma3);
# print("Single Multivariate pdf calculation " + str(y_dummy))
#print(y_dummy)
#

prior_prob_3 = 0.5
prior_prob_7 = 0.5



# method  - for training data calculate pdf(3)*P(3) and pdf(7)*P(7) whichever is minimum is the error

total_error_2_case1 = 0
for i in range(0,len(Y_transform_train)):
    # if train_data_label[i] == 3:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7);
    # elif train_data_label[i] == 7:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3);
    value3 = multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3
    value7 = multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7
    total_error_2_case1 += min(value3,value7)

total_error_2_case1 = total_error_2_case1/len(Y_transform_train)
print("Total error for training set for case1 is: " + str(total_error_2_case1))

prior_prob_7 = 0.7
prior_prob_3 = 0.3
total_error_2_case2 = 0
for i in range(0,len(Y_transform_train)):
    # if train_data_label[i] == 3:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7);
    # elif train_data_label[i] == 7:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3);
    value3 = multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3
    value7 = multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7
    total_error_2_case2 += min(value3,value7)

total_error_2_case2 = total_error_2_case2/len(Y_transform_train)
print("Total error for training set for case2 is: " + str(total_error_2_case2))



#--------------------------------------------------------------------------------------------

#Test data

test_data = scipy.io.loadmat("test_data.mat")
test_data_label = test_data['label'].flatten()
#print("Trained data label: " + str(train_data_label))
#print("Training data length: " + str())
transformed_test_data = []
for i in test_data['data']:
    flattened_test_data = i.flatten()
    list = []
    list.append(np.mean(flattened_test_data))
    list.append(np.std(flattened_test_data))
    transformed_test_data.append(list) #in the form [mi,si]

Y_transform_test = []
for i in transformed_test_data:
    list = []
    list.append((i[0]-Mean_train[0])/SD_train[0])
    list.append((i[1]-Mean_train[1])/SD_train[1])
    Y_transform_test.append(list)

prior_prob_7 = 0.5
prior_prob_3 = 0.5
total_error_2_test_data_case1 = 0
for i in range(0,len(Y_transform_test)):
    # if train_data_label[i] == 3:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7);
    # elif train_data_label[i] == 7:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3);
    value3 = multivariate_normal.pdf(Y_transform_test[i], mean=Mean3, cov=sigma3)*prior_prob_3
    value7 = multivariate_normal.pdf(Y_transform_test[i], mean=Mean7, cov=sigma7)*prior_prob_7
    total_error_2_test_data_case1 += min(value3,value7)

total_error_2_test_data_case1 = total_error_2_test_data_case1/len(Y_transform_test)
print("Total error for test set for case1 is: " + str(total_error_2_test_data_case1))


prior_prob_7 = 0.7
prior_prob_3 = 0.3
total_error_2_test_data_case2 = 0
for i in range(0,len(Y_transform_test)):
    # if train_data_label[i] == 3:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7);
    # elif train_data_label[i] == 7:
    #     total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3);
    value3 = multivariate_normal.pdf(Y_transform_test[i], mean=Mean3, cov=sigma3)*prior_prob_3
    value7 = multivariate_normal.pdf(Y_transform_test[i], mean=Mean7, cov=sigma7)*prior_prob_7
    total_error_2_test_data_case2 += min(value3,value7)

total_error_2_test_data_case2 = total_error_2_test_data_case2/len(Y_transform_test)
print("Total error for test set for case2 is: " + str(total_error_2_test_data_case2))
