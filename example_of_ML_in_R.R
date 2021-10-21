### A ML project in R: compute a learner for flowers using the IRIS dataset

library(caret)
# attach the iris dataset to the environment
data(iris)
# rename the dataset
dataset <- iris
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]
# dimensions of dataset
dim(dataset)
# [1] 120   5
# list the levels for the class
levels(dataset$Species)
# [1] "setosa"  "versicolor" "virginica" , we have three kind of iris flower in the dataset
# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
# kNN
set.seed(1)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# summarize Best Model
print(fit.knn)
# estimate skill of LDA on the validation dataset
predictions <- predict(fit.knn, validation)
confusionMatrix(predictions, validation$Species)
