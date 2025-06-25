# Core Data Manipulation and Cleaning
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("e1071")
install.packages("rpart")
install.packages("dplyr")
install.packaegs("olsrr")

# Loading the required libraries
library(tidyverse)
library(ggplot2)
library(corrplot)
library(e1071)
library(rpart)
library(dplyr)
library(olsrr)
library(class)

# Importing the dataset as fashionDataSet
library(readr)

fashionDataSet <- read_csv(file = "fashion_data_2018_2022.csv",
                           col_names = TRUE,
                           col_types = cols(
                             product_id = col_integer(),
                             product_name = col_character(),
                             gender = col_character(),
                             category = col_character(),
                             pattern = col_character(),
                             color = col_character(),
                             age_group = col_character(),
                             season = col_character(),
                             price_USD = col_double(),
                             material = col_character(),
                             sales_count = col_integer(),
                             reviews_count = col_integer(),
                             average_rating = col_double(),
                             out_of_stock_times = col_integer(),
                             brand = col_character(),
                             discount = col_character(),
                             last_stock_date = col_date(format = "%Y-%m-%d"),
                             wish_list_count = col_integer(),
                             month_of_sale = col_integer(),
                             year_of_sale = col_integer(),
                             High_Demand _Dependent Variable = col_integer()
                           ))

# Generate histogram for average_rating
ggplot(fashionDataSet, aes(x = average_rating)) +
  geom_histogram(binwidth = 0.4, fill = "skyblue", color = "black") +
  ggtitle("Histogram of Average Rating") +
  xlab("Average Rating") +
  ylab("Frequency") +
  scale_x_continuous(limits = c(0, 5), breaks = seq(0, 5, by = 0.5)) +
  theme_minimal()

# Histogram for price_USD
ggplot(fashionDataSet, aes(x = price_USD)) +
  geom_histogram(binwidth = 10, fill = "lightblue", color = "black") +
  ggtitle("Histogram of Price (USD)") +
  xlab("Price (USD)") +
  ylab("Frequency") +
  theme_minimal()

# Histogram for sales_count
ggplot(fashionDataSet, aes(x = sales_count)) +
  geom_histogram(binwidth = 20, fill = "lightgreen", color = "black") +
  ggtitle("Histogram of Sales Count") +
  xlab("Sales Count") +
  ylab("Frequency") +
  theme_minimal()

# Histogram for reviews_count
ggplot(fashionDataSet, aes(x = reviews_count)) +
  geom_histogram(binwidth = 10, fill = "coral", color = "black") +
  ggtitle("Histogram of Reviews Count") +
  xlab("Reviews Count") +
  ylab("Frequency") +
  theme_minimal()

# Histogram for out_of_stock_times
ggplot(fashionDataSet, aes(x = out_of_stock_times)) +
  geom_histogram(binwidth = 1, fill = "lightpink", color = "black") +
  ggtitle("Histogram of Out of Stock Times") +
  xlab("Out of Stock Times") +
  ylab("Frequency") +
  scale_x_continuous(limits = c(1, 10), breaks = seq(1, 10, by = 1)) +
  theme_minimal()

# Histogram for wish_list_count
ggplot(fashionDataSet, aes(x = wish_list_count)) +
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") +
  ggtitle("Histogram of Wish List Count") +
  xlab("Wish List Count") +
  ylab("Frequency") +
  theme_minimal()

# Normalize continuous variables
fashionDataSet <- fashionDataSet %>%
  mutate(
    # Normalizing using Min-Max Scaling
    average_rating_norm = (average_rating - min(average_rating)) / 
      (max(average_rating) - min(average_rating)),
    price_USD_norm = (price_USD - min(price_USD)) / 
      (max(price_USD) - min(price_USD)),
    out_of_stock_times_norm = (out_of_stock_times - min(out_of_stock_times)) / 
      (max(out_of_stock_times) - min(out_of_stock_times)),
    
    # Normalizing using Z-score
    sales_count_zscore = (sales_count - mean(sales_count)) / sd(sales_count),
    reviews_count_zscore = (reviews_count - mean(reviews_count)) / 
      sd(reviews_count),
    wish_list_count_zscore = (wish_list_count - mean(wish_list_count)) /
      sd(wish_list_count)
  )

# Summary of normalized variables
summary(fashionDataSet %>% select(average_rating_norm, price_USD_norm, 
                                  out_of_stock_times_norm,sales_count_zscore, 
                                  reviews_count_zscore, wish_list_count_zscore))

# List of normalized features
normalized_features <- c("average_rating_norm", "price_USD_norm",
                         "out_of_stock_times_norm", "sales_count_zscore", 
                         "reviews_count_zscore", "wish_list_count_zscore")

# Generate and display histograms for each normalized feature
for (feature in normalized_features) {
  plot <- ggplot(fashionDataSet, aes_string(x = feature)) +
    geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black") +
    ggtitle(paste("Histogram of", feature)) +
    xlab(feature) +
    ylab("Frequency") +
    theme_minimal()
  print(plot)  
}

set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(fashionDataSet), 0.7 * nrow(fashionDataSet))
train_data <- fashionDataSet[train_indices, ]
test_data <- fashionDataSet[-train_indices, ]

# Check column names in train_data and test_data
colnames(train_data)
colnames(test_data)

train_data$`High_Demand _Dependent Variable` <- 
  fashionDataSet$`High_Demand _Dependent Variable`[train_indices]
test_data$`High_Demand _Dependent Variable` <- 
  fashionDataSet$`High_Demand _Dependent Variable`[-train_indices]

# Check the structure of train_data and test_data
str(train_data)
str(test_data)

# Confirm presence of dependent variable
colnames(train_data)
colnames(test_data)

# Fit logistic regression model
logistic_model <- glm(`High_Demand _Dependent Variable` ~ ., 
                      family = "binomial", 
                      data = train_data)

# Summary of the logistic regression model
summary(logistic_model)

# Predict probabilities on test data
logistic_preds <- predict(logistic_model, newdata = test_data, 
                          type = "response")

# Convert probabilities to class labels (0 or 1)
logistic_classes <- ifelse(logistic_preds > 0.5, 1, 0)

# View predictions
head(logistic_classes)

# Confusion matrix
confusion_matrix <- table(test_data$`High_Demand _Dependent Variable`, 
                          logistic_classes)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))

# Check class distribution in the original dataset
table(fashionDataSet$`High_Demand _Dependent Variable`)

# Separate the minority and majority classes
minority_class <- train_data %>%
  filter(`High_Demand _Dependent Variable` == 0)

majority_class <- train_data %>%
  filter(`High_Demand _Dependent Variable` == 1)

# Oversample the minority class by duplicating rows
oversampled_minority <- minority_class %>%
  slice_sample(n = nrow(majority_class), replace = TRUE)

# Combine oversampled minority class with majority class
balanced_train_data <- bind_rows(majority_class, oversampled_minority)

# Shuffle the balanced dataset
balanced_train_data <- balanced_train_data %>%
  slice_sample(n = nrow(balanced_train_data))

# Check the new class distribution
table(balanced_train_data$`High_Demand _Dependent Variable`)

# Fit logistic regression model on balanced data
balanced_logistic_model <- glm(`High_Demand _Dependent Variable` ~ ., 
                               family = "binomial", 
                               data = balanced_train_data)

# Summary of the retrained logistic regression model
summary(balanced_logistic_model)

# Predict probabilities on test data
balanced_logistic_preds <- predict(balanced_logistic_model, newdata = 
                                     test_data, type = "response")

# Convert probabilities to class labels (0 or 1)
balanced_logistic_classes <- ifelse(balanced_logistic_preds > 0.5, 1, 0)

# Confusion matrix
confusion_matrix <- table(test_data$`High_Demand _Dependent Variable`, 
                          balanced_logistic_classes)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))

# Extract confusion matrix values
TN <- confusion_matrix[1, 1]  # True Negatives
FP <- confusion_matrix[1, 2]  # False Positives
FN <- confusion_matrix[2, 1]  # False Negatives
TP <- confusion_matrix[2, 2]  # True Positives

# Precision
precision <- TP / (TP + FP)

# Recall
recall <- TP / (TP + FN)

# F1-Score
f1_score <- 2 * ((precision * recall) / (precision + recall))

# Print the metrics
print(paste("Precision:", round(precision, 2)))
print(paste("Recall:", round(recall, 2)))
print(paste("F1-Score:", round(f1_score, 2)))

# KNN Model
# Select normalized features for KNN
knn_train <- train_data[, c("genderFemale", "genderMale", 
                            "seasonAutumn", "seasonSpring", 
                            "seasonSummer", "seasonWinter")]

knn_test <- test_data[, c("genderFemale", "genderMale", 
                          "seasonAutumn", "seasonSpring", 
                          "seasonSummer", "seasonWinter")]

# Ensure the dependent variable is separate
knn_train_labels <- train_data$`High_Demand _Dependent Variable`
knn_test_labels <- test_data$`High_Demand _Dependent Variable`

# Apply KNN with k = 3
knn_preds <- knn(knn_train, knn_test, cl = knn_train_labels, k = 3)

# View predictions
head(knn_preds)

# Confusion matrix
knn_confusion_matrix <- table(knn_test_labels, knn_preds)
print(knn_confusion_matrix)

# Calculate accuracy
knn_accuracy <- sum(diag(knn_confusion_matrix)) / sum(knn_confusion_matrix)
print(paste("Accuracy:", round(knn_accuracy, 2)))

k_values <- seq(1, 20, by = 2)  # Test odd values of k to avoid ties
accuracy_values <- sapply(k_values, function(k) {
  knn_preds <- knn(knn_train, knn_test, cl = knn_train_labels, k = k)
  confusion_matrix <- table(knn_test_labels, knn_preds)
  sum(diag(confusion_matrix)) / sum(confusion_matrix)  # Accuracy
})

optimal_k <- k_values[which.max(accuracy_values)]
print(paste("Optimal k:", optimal_k))

# Naive Bayes Model
# Train Naive Bayes model
naive_model <- naiveBayes(`High_Demand _Dependent Variable` ~ ., 
                          data = train_data)

# View model summary
naive_model

# Predict class labels on test data
naive_preds <- predict(naive_model, newdata = test_data)

# View predictions
head(naive_preds)

# Confusion matrix
naive_confusion_matrix <- table(test_data$`High_Demand _Dependent Variable`,
                                naive_preds)
print(naive_confusion_matrix)

# Calculate accuracy
naive_accuracy <- sum(diag(naive_confusion_matrix)) / 
  sum(naive_confusion_matrix)
print(paste("Accuracy:", round(naive_accuracy, 2)))

# Optional: Calculate precision, recall, and F1-Score
# Extract confusion matrix values
TN <- naive_confusion_matrix[1, 1]  # True Negatives
FP <- naive_confusion_matrix[1, 2]  # False Positives
FN <- naive_confusion_matrix[2, 1]  # False Negatives
TP <- naive_confusion_matrix[2, 2]  # True Positives

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * ((precision * recall) / (precision + recall))

print(paste("Precision:", round(precision, 2)))
print(paste("Recall:", round(recall, 2)))
print(paste("F1-Score:", round(f1_score, 2)))

# Decision Tree Model