setwd("/Users/utkusizanli/Desktop/UC3M/StatisticalLearningGitHub")

# Libraries
library(caret)
library(MASS)
library(klaR)
library(dplyr)
library(e1071)

# Load dataset
mental = read.csv("mental_health.csv")

# Check size and structure of data
dim(mental)     # number of rows and columns
str(mental)     # variable types

# Convert all character columns to factors automatically
char_cols <- names(mental)[sapply(mental, is.character)]
mental[char_cols] <- lapply(mental[char_cols], as.factor)

# Convert target to factor, 1 = Yes (positive), 0 = No
mental$Has_Mental_Health_Issue <- factor(ifelse(mental$Has_Mental_Health_Issue == 1, "Yes", "No"), levels = c("Yes", "No"))


# Check for missing values
colSums(is.na(mental))   # No missing values 

# Check data again after conversion
str(mental)
summary(mental)

# Check target distribution 
table(mental$Has_Mental_Health_Issue)
prop.table(table(mental$Has_Mental_Health_Issue))

# Notes: The target variable is highly imbalanced, with 7.84% of observations in class 0 and 92.16% in class 1. 
# This imbalance may affect classification performance, since models could favor the majority class.

# Select categorical variables (exclude target)
cat_vars = setdiff(names(mental)[sapply(mental, is.factor)], "Has_Mental_Health_Issue")

# Measure strength of association using chi-square statistic
chi_strength = sapply(cat_vars, function(v) {
  as.numeric(chisq.test(table(mental[[v]], mental$Has_Mental_Health_Issue))$statistic)
})

# Select top 6 categorical variables
top_vars_cat = names(sort(chi_strength, decreasing = TRUE))[1:6]

# Plot barplots for selected categorical variables
par(mfrow = c(2,3))

lapply(top_vars_cat, function(v) {
  tab = prop.table(table(mental[[v]], mental$Has_Mental_Health_Issue), 1)
  
  barplot(t(tab),
          beside = TRUE,
          main = v,
          legend.text = TRUE,
          xlab = "Category",
          ylab = "Proportion")
})

# Screen numeric predictors using t-tests
pvals = sapply(mental[sapply(mental, is.numeric)],
               function(x) t.test(x ~ mental$Has_Mental_Health_Issue)$p.value)

sort(pvals)   

# Compute mean difference between classes 
mean_diff = sapply(mental[sapply(mental, is.numeric)], function(x) {
  abs(mean(x[mental$Has_Mental_Health_Issue=="Yes"]) -
        mean(x[mental$Has_Mental_Health_Issue=="No"]))
})

sort(mean_diff, decreasing=TRUE)   

# Select top numeric predictors
top_vars = names(sort(mean_diff, decreasing=TRUE)[1:6])

# Plot boxplots for selected numeric predictors
par(mfrow=c(2,3))

lapply(top_vars, function(v) {
  boxplot(mental[[v]] ~ mental$Has_Mental_Health_Issue,
          main=v)
})

#EDA

# Demographic categorical variables (like gender, country, marital status) do not seem to strongly separate people with and without mental health issues.

# Psychological and stress-related numeric variables show clear differences between the two groups.

# Variables such as Work_Stress_Level, Feeling_Sad_Down, Financial_Stress, and Anxious_Nervous look like strong predictors of mental health issues.



# 60% Train - 20% Validation - 20% Test split via stratified sampling
set.seed(42)

# 60% TRAIN, 20% VAL, 20% TEST (all stratified by the target)
idx_train <- createDataPartition(mental$Has_Mental_Health_Issue, p = 0.60, list = FALSE)
mental_train <- mental[idx_train, ]
mental_tmp   <- mental[-idx_train, ]

# half of remaining
idx_val <- createDataPartition(mental_tmp$Has_Mental_Health_Issue, p = 0.50, list = FALSE)
mental_val  <- mental_tmp[idx_val, ]
mental_test <- mental_tmp[-idx_val, ]

# check proportions to confirm stratifications
prop.table(table(mental$Has_Mental_Health_Issue))
prop.table(table(mental_train$Has_Mental_Health_Issue))
prop.table(table(mental_val$Has_Mental_Health_Issue))
prop.table(table(mental_test$Has_Mental_Health_Issue))


target_col <- "Has_Mental_Health_Issue"
num_cols <- names(mental_train)[sapply(mental_train, is.numeric)]
num_cols <- setdiff(num_cols, target_col)

mu <- sapply(mental_train[, num_cols, drop = FALSE], mean)
sd <- sapply(mental_train[, num_cols, drop = FALSE], sd)
sd[sd == 0] <- 1

scale_apply <- function(df, num_cols, mu, sd) {
  out <- df
  out[, num_cols] <- sweep(out[, num_cols, drop = FALSE], 2, mu, "-")
  out[, num_cols] <- sweep(out[, num_cols, drop = FALSE], 2, sd, "/")
  out
}

train_sc <- scale_apply(mental_train, num_cols, mu, sd)
val_sc   <- scale_apply(mental_val,   num_cols, mu, sd)
test_sc  <- scale_apply(mental_test,  num_cols, mu, sd)



# setup 5-fold cross-validation with 3 repeats
folds <- createMultiFolds(train_sc$Has_Mental_Health_Issue, k = 5, times = 3)

# Train initial logreg, LDA, QDA and NB models
# Since the dataset is imbalanced we avoid using accuracy and optimized based on ROC
set.seed(42)
fit_logreg <- train(Has_Mental_Health_Issue ~ ., data = mental_train, method = "glm", family = binomial(), metric = "ROC", trControl = ctrl)
fit_lda <- train(Has_Mental_Health_Issue ~ ., data = mental_train, method = "lda", metric = "ROC", trControl = ctrl)
fit_qda <- train(Has_Mental_Health_Issue ~ ., data = mental_train, method = "qda", metric = "ROC", trControl = ctrl)
fit_nb <- train(Has_Mental_Health_Issue ~ ., data = mental_train, method = "nb", metric = "ROC", trControl = ctrl, tuneLength = 10)


# same columns across splits?
stopifnot(identical(names(mental_train), names(mental_val)))
stopifnot(identical(names(mental_train), names(mental_test)))

# any weird non-atomic columns?
stopifnot(all(sapply(mental_train, is.atomic)))


models <- list(LogReg = fit_logreg, LDA = fit_lda, QDA = fit_qda, NB = fit_nb)

# (optional) CV ROC summary
sapply(models, function(m) max(m$results$ROC))

