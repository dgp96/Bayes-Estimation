print("Task 2")
#method 0-- wrong


# pdf3 = 0
# for i in range(0,len(label_3)):
#     pdf3 += multivariate_normal.pdf(label_3[i], mean=Mean3, cov=sigma3);
#
# error3 = (pdf3*prior_prob_3)/len(label_3)
#
# pdf7 = 0
# for i in range(0,len(label_7)):
#     pdf7 += multivariate_normal.pdf(label_7[i], mean=Mean7, cov=sigma7);
#
# error7 = (pdf7*prior_prob_7)/len(label_7)
#
# total_error_0 = 1 - (error3 + error7)
#
# print("Probability error1: " +str(total_error_0))
#
# # method 1 - for training data if label is 3 error is pdf(7)*P(7)
# total_error_1 = 0
# for i in range(0,len(Y_transform_train)):
#     if train_data_label[i] == 3:
#         total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean7, cov=sigma7)*prior_prob_7);
#     elif train_data_label[i] == 7:
#         total_error_1 += (multivariate_normal.pdf(Y_transform_train[i], mean=Mean3, cov=sigma3)*prior_prob_3);
#
# total_error_1 = total_error_1/len(Y_transform_train)
#
# print("Total error by method 1 is: " + str(total_error_1))
