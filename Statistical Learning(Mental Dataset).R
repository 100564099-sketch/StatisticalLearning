setwd("/Users/utkusizanli/Desktop/UC3M/StatisticalLearningGitHub")

# Libraries
library(caret)
library(MASS)
library(dplyr)
library(e1071)
library(pROC)
library(ggplot2)
library(tidyr)
library(patchwork)
library(DMwR)
library(PRROC)
library(kernlab)
library(e1071)
library(viridis)
library(randomForest)
library(purrr)
library(forcats)
library(tibble)
library(tree) 


# Load dataset
mental = read.csv("mental_health.csv")

# Check size and structure of data
dim(mental)     # number of rows and columns
str(mental)     # variable types


# 2. Data Preparation
# Convert target variable to factor (classification task)


mental$Has_Mental_Health_Issue <- factor(
  mental$Has_Mental_Health_Issue,
  levels = c("0", "1"),
  labels = c("No", "Yes")
)

# quick check
levels(mental$Has_Mental_Health_Issue)
# Convert all character columns to factors automatically
mental[] = lapply(mental, function(x) if (is.character(x)) as.factor(x) else x)

# Check for missing values
colSums(is.na(mental))   # No missing values 

# Check data again after conversion
str(mental)
summary(mental)

# 3. Target distribution 
table(mental$Has_Mental_Health_Issue)
prop.table(table(mental$Has_Mental_Health_Issue))

# Notes: The target variable is highly imbalanced, with 7.84% of observations in class 0 and 92.16% in class 1. 

# Visualize class imbalance using two continuous variables
ggplot(mental, aes(x = Screen_Time_Hours_Day,
                   y = Sleep_Hours_Night,
                   color = Has_Mental_Health_Issue)) +
  geom_point(alpha = 0.35, size = 1.6) +
  labs(title = "Imbalanced Dataset: Mental Health",
       subtitle = "Only 7.8% of observations are class 0",
       x = "Screen Time (hours/day)",
       y = "Sleep (hours/night)",
       color = "Class") +
  theme_minimal()

# 4. EDA - Categorical Variables

# Select categorical variables (exclude target)
cat_vars <- setdiff(names(mental)[sapply(mental, is.factor)],
                    "Has_Mental_Health_Issue")

# Top 6 predictors were selected using chi-square statistic to measure association with the target 
chi_strength <- sapply(cat_vars, function(v) {
  as.numeric(chisq.test(table(mental[[v]], mental$Has_Mental_Health_Issue))$statistic)
})

top_cat <- names(sort(chi_strength, decreasing = TRUE))[1:6] # Pick top 6 categorical variables

par(mfrow = c(2,3))
lapply(top_cat, function(v) {
  tab <- prop.table(table(mental[[v]], mental$Has_Mental_Health_Issue), 1)
  barplot(tab,
          col = c("lightblue","pink"),
          main = paste("Class Proportions by", v),
          ylab = "Proportion")
})



# 5. EDA - Numeric Variables 
num_vars <- names(mental)[sapply(mental, is.numeric)]

 
# We separate the variables by scale (binary, discrete, continuous)
# to ensure suitable exploratory analysis and visualization techniques.

# Binary numeric variables (0/1)

binary_vars <- num_vars[sapply(mental[num_vars], function(x) length(unique(x)) == 2)]

# Effect size for binary variables:
# measure the difference in the proportion of value = 1
# between the two classes of the target variable

# Get class labels automatically from the factor levels
class_levels <- levels(mental$Has_Mental_Health_Issue)

bin_diff <- sapply(binary_vars, function(v) {
  x <- mental[[v]]
  
  p1 <- mean(x[mental$Has_Mental_Health_Issue == class_levels[2]] == 1, na.rm = TRUE)
  p0 <- mean(x[mental$Has_Mental_Health_Issue == class_levels[1]] == 1, na.rm = TRUE)
  
  abs(p1 - p0)
})

# Remove NA values
bin_diff <- bin_diff[!is.na(bin_diff)]

# Select top 6 predictors
top_binary <- names(sort(bin_diff, decreasing = TRUE))[1:min(6, length(bin_diff))]

top_binary

# Plot P(X=0/1 | class) for the top binary variables

par(mfrow = c(2,3))

lapply(top_binary, function(v) {
  
  tab <- prop.table(table(mental[[v]], mental$Has_Mental_Health_Issue), 2)
  # columns = classes, so bars show proportions within each class
  
  barplot(tab,
          beside = TRUE,
          main = paste("P(", v, " | class)", sep = ""),
          col = c("lightblue", "pink"),
          legend.text = c("0", "1"),
          xlab = "Class",
          ylab = "Proportion")
})

# Discrete variables (0-10 scales)

discrete_vars <- num_vars[sapply(mental[num_vars], function(x)
  length(unique(x)) > 2 & length(unique(x)) <= 15)]

# Use factor levels 
lev <- levels(mental$Has_Mental_Health_Issue)   # usually c("0","1")

mean_diff_discrete <- sapply(discrete_vars, function(v) {
  x <- mental[[v]]
  abs(mean(x[mental$Has_Mental_Health_Issue == lev[2]], na.rm = TRUE) -
        mean(x[mental$Has_Mental_Health_Issue == lev[1]], na.rm = TRUE))
})

# Remove NA differences 
mean_diff_discrete <- mean_diff_discrete[!is.na(mean_diff_discrete)]

# Pick top 6 predictors
top_discrete <- names(sort(mean_diff_discrete, decreasing = TRUE))[1:min(6, length(mean_diff_discrete))]
top_discrete

par(mfrow = c(2,3))
lapply(top_discrete, function(v) {
  boxplot(mental[[v]] ~ mental$Has_Mental_Health_Issue,
          main = paste("Distribution of", v, "by Class"),
          xlab = "Mental Health Issue",
          ylab = v)
})

# Continuous variables (density distributions plots)
# This helps compare how each variable is distributed across the two classes

continuous_vars <- num_vars[sapply(mental[num_vars], function(x)
  length(unique(x)) > 15)]

lev <- levels(mental$Has_Mental_Health_Issue)   # usually c("0","1")

mean_diff_cont <- sapply(continuous_vars, function(v) {
  x <- mental[[v]]
  abs(mean(x[mental$Has_Mental_Health_Issue == lev[2]], na.rm = TRUE) -
        mean(x[mental$Has_Mental_Health_Issue == lev[1]], na.rm = TRUE))
})

# remove NA values 
mean_diff_cont <- mean_diff_cont[!is.na(mean_diff_cont)]

# pick top 4 predictors
top_continuous <- names(sort(mean_diff_cont, decreasing = TRUE))[1:min(4, length(mean_diff_cont))]
top_continuous

mental_long <- mental %>%
  select(all_of(top_continuous), Has_Mental_Health_Issue) %>%
  pivot_longer(-Has_Mental_Health_Issue,
               names_to = "Variable",
               values_to = "Value")

ggplot(mental_long, aes(x = Value, fill = Has_Mental_Health_Issue)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "Top Continuous Variables by Class")

# EDA conclusion:

# Demographic categorical variables (like Gender, Country, and Marital Status) do not clearly separate the two groups.
#	Several binary variables (like trauma history or previous diagnosis) show noticeable differences between classes.
#	Psychological and stress-related numeric variables show clearer separation between groups.
#	Variables such as Work_Stress_Level, Feeling_Sad_Down, Financial_Stress, and Anxious_Nervous 
# appear to be strong predictors of  mental health issues.


# ====================================================================================================
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

'
Data splitting: stratified Train / Validation / Test

To evaluate classification performance fairly and avoid optimistic estimates, the dataset was partitioned into three mutually exclusive subsets:

Training set (60%): used to fit the models.

Validation set (20%): used for model/threshold selection and tuning decisions.

Test set (20%): held out until the end to obtain an unbiased estimate of final performance.

Because the target variable Has_Mental_Health_Issue is highly imbalanced, the split was performed using stratified sampling (caret::createDataPartition). Stratification ensures that the class proportions are approximately preserved across train, validation, and test sets, preventing one subset from accidentally containing disproportionately more “Yes” or “No” observations.

A fixed random seed (set.seed(42)) was set to make the split reproducible. After splitting, class proportions were printed for the full dataset and for each subset to confirm that stratification was successful.
'



# Forward Stepwise (AIC) Feature Selection 
# Stepwise should be done on ORIGINAL training data (no SMOTE) to avoid synthetic-feature bias
df_fs <- mental_train
df_fs$Has_Mental_Health_Issue <- factor(df_fs$Has_Mental_Health_Issue, levels = c("No","Yes"))

m0 <- glm(Has_Mental_Health_Issue ~ 1, data = df_fs, family = binomial())
m_full <- glm(Has_Mental_Health_Issue ~ ., data = df_fs, family = binomial())

m_fwd <- stepAIC(
  m0,
  scope = list(lower = m0, upper = m_full),
  direction = "forward",
  trace = FALSE
)

selected_terms <- attr(terms(m_fwd), "term.labels")
cat("\n[Stepwise] Selected predictors (AIC):\n")
print(selected_terms)

keep_cols <- c("Has_Mental_Health_Issue", selected_terms)

# overwrite the same objects so you don't change anything below
mental_train <- mental_train[, keep_cols, drop = FALSE]
mental_val   <- mental_val[,   keep_cols, drop = FALSE]
mental_test  <- mental_test[,  keep_cols, drop = FALSE]

dim(mental_train)

'
Forward stepwise feature selection (AIC)

To reduce model complexity and avoid using the full set of predictors, a forward stepwise selection procedure based on AIC was applied on the original training set (before SMOTE). This is important because SMOTE generates synthetic observations, which can bias variable selection if used during the stepwise process.

Starting from the intercept-only logistic regression model, predictors were added sequentially to minimize the Akaike Information Criterion (AIC), resulting in a reduced subset of variables. After selection, the dataset was restricted to the selected predictors and the target variable, while keeping the validation and test sets consistent by applying the same column subset.

The feature count was reduced from 51 original columns to 23 columns in the training set (including the target variable). The final stepwise logistic regression contained 24 coefficients including the intercept, because at least one categorical predictor was represented using a dummy indicator in the model matrix.
'

# ====================================================================================================


# SMOTE to reduce imbalance
train_smote <- SMOTE(Has_Mental_Health_Issue ~ ., data = mental_train,
                     perc.over = 600, perc.under = 100)

train_smote$Has_Mental_Health_Issue <- factor(
  train_smote$Has_Mental_Health_Issue,
  levels = c("No", "Yes")
)

table(train_smote$Has_Mental_Health_Issue)

prop.table(table(mental_train$Has_Mental_Health_Issue))
prop.table(table(train_smote$Has_Mental_Health_Issue))

'
Class imbalance handling: SMOTE on the training set

The target variable is strongly imbalanced in the training data (No ≈ 7.85%, Yes ≈ 92.15%). To mitigate bias toward the majority class and improve the model’s ability to learn patterns for the minority class, SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training set.

SMOTE generates synthetic samples of the minority class (“No”) by interpolating between existing minority observations in feature space, and can optionally downsample the majority class. In this workflow, SMOTE was used with:

perc.over = 600: increases the minority class by creating additional synthetic “No” observations.

perc.under = 100: controls the amount of majority-class (“Yes”) sampling relative to the expanded minority class.

After applying SMOTE, the target variable was re-cast as a factor with consistent level ordering (No, Yes). Class distributions were then checked before and after resampling:

Before SMOTE (training set): No = 0.0785, Yes = 0.9215

After SMOTE: No = 0.5385, Yes = 0.4615

This shows that the training data were transformed into a roughly balanced dataset, improving the learning signal for the minority class while keeping the validation and test sets untouched to ensure unbiased evaluation.
'
# ====================================================================================================

target_col <- "Has_Mental_Health_Issue"
num_cols <- names(train_smote)[sapply(train_smote, is.numeric)]
num_cols <- setdiff(num_cols, target_col)

mu <- sapply(train_smote[, num_cols, drop = FALSE], mean)
sd <- sapply(train_smote[, num_cols, drop = FALSE], sd)
sd[sd == 0] <- 1

scale_apply <- function(df, num_cols, mu, sd) {
  out <- df
  out[, num_cols] <- sweep(out[, num_cols, drop = FALSE], 2, mu, "-")
  out[, num_cols] <- sweep(out[, num_cols, drop = FALSE], 2, sd, "/")
  out
}

train_sc <- scale_apply(train_smote, num_cols, mu, sd)
val_sc   <- scale_apply(mental_val,   num_cols, mu, sd)
test_sc  <- scale_apply(mental_test,  num_cols, mu, sd)

'
Feature scaling (standardization)

All numeric predictors were standardized using z-score scaling (subtract mean, divide by standard deviation). The mean and standard deviation were computed only on the SMOTEd training set, then the same parameters were applied to the validation and test sets to prevent data leakage. Variables with zero variance were handled by setting their standard deviation to 1.
'
# ====================================================================================================


# setup 5-fold cross-validation with 3 repeats
folds <- createMultiFolds(train_sc$Has_Mental_Health_Issue, k = 5, times = 3)
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3, index = folds, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final")


# logreg, LDA, QDA cross-validation as initial models
set.seed(42)
fit_logreg <- train(Has_Mental_Health_Issue ~ ., data=train_sc, method="glm", family=binomial(), metric="ROC", trControl=ctrl)
set.seed(42)
fit_lda    <- train(Has_Mental_Health_Issue ~ ., data=train_sc, method="lda", metric="ROC", trControl=ctrl)
set.seed(42)
fit_qda    <- train(Has_Mental_Health_Issue ~ ., data=train_sc, method="qda", metric="ROC", trControl=ctrl)

# Naive Bayes with same CV format
auc_each <- numeric(length(folds))
i <- 0

for (nm in names(folds)) {
  i <- i + 1
  idx_in  <- folds[[nm]]
  idx_out <- setdiff(seq_len(nrow(train_sc)), idx_in)
  
  nb_fit <- naiveBayes(Has_Mental_Health_Issue ~ ., data = train_sc[idx_in, ])
  p_out  <- predict(nb_fit, newdata = train_sc[idx_out, ], type = "raw")[, "Yes"]
  
  roc_i <- roc(train_sc$Has_Mental_Health_Issue[idx_out], p_out,
               levels = c("No","Yes"), direction = "<", quiet = TRUE)
  auc_each[i] <- as.numeric(auc(roc_i))
}

auc_nb <- mean(auc_each, na.rm = TRUE)


# CV ROC summary
c(LogReg=max(fit_logreg$results$ROC), LDA=max(fit_lda$results$ROC), QDA=max(fit_qda$results$ROC), NB=auc_nb)
'
Model training and cross-validation (ROC-AUC)

Model performance was estimated using repeated stratified cross-validation on the standardized, SMOTEd training set. Specifically, 5-fold cross-validation repeated 3 times was used to reduce variance in performance estimates. ROC-AUC was selected as the primary metric (metric = "ROC") because it is threshold-independent and appropriate for imbalanced classification.

Four baseline classifiers were evaluated:

Logistic Regression (GLM, binomial)

Linear Discriminant Analysis (LDA)

Quadratic Discriminant Analysis (QDA)

Naive Bayes

Logistic regression, LDA, and QDA were trained via caret::train using the same resampling scheme and ROC-based summary function. Naive Bayes was evaluated using the identical fold indices in a manual loop to compute fold-level ROC-AUC values, which were then averaged.

The resulting cross-validated ROC-AUC scores were:

LogReg: 0.705

LDA: 0.705

QDA: 0.795

Naive Bayes: 0.750
'
# ====================================================================================================


# Threshold Selection
thr_grid <- seq(0.05, 0.95, by = 0.01)

calc_metrics_bal <- function(y_true, prob_yes, thr) {
  pred <- factor(ifelse(prob_yes >= thr, "Yes", "No"), levels = c("No","Yes"))
  cm <- caret::confusionMatrix(pred, y_true, positive = "Yes")
  
  acc  <- as.numeric(cm$overall["Accuracy"])
  sens <- as.numeric(cm$byClass["Sensitivity"])
  spec <- as.numeric(cm$byClass["Specificity"])
  bal  <- 0.5 * (sens + spec)
  
  prec <- as.numeric(cm$byClass["Pos Pred Value"])
  f1   <- if (is.na(prec) || is.na(sens) || (prec + sens) == 0) NA_real_ else 2 * prec * sens / (prec + sens)
  
  data.frame(threshold = thr, Accuracy = acc, Precision = prec,
             Sensitivity = sens, Specificity = spec, BalancedAcc = bal, F1 = f1)
}

# 3) threshold picker
pick_best_thr_bal <- function(y_true, prob_yes) {
  tbl <- dplyr::bind_rows(lapply(thr_grid, function(t) calc_metrics_bal(y_true, prob_yes, t))) |>
    dplyr::arrange(dplyr::desc(BalancedAcc), dplyr::desc(Sensitivity), dplyr::desc(Specificity))
  list(best = tbl[1, ], all = tbl)
}

# --- probabilities on VAL
p_logreg <- predict(fit_logreg, newdata = val_sc, type = "prob")[, "Yes"]
p_lda    <- predict(fit_lda,    newdata = val_sc, type = "prob")[, "Yes"]
p_qda    <- predict(fit_qda,    newdata = val_sc, type = "prob")[, "Yes"]

nb_train <- e1071::naiveBayes(Has_Mental_Health_Issue ~ ., data = train_sc)
p_nb     <- predict(nb_train, newdata = val_sc, type = "raw")[, "Yes"]

# --- pick best threshold per model (by F1)
best_logreg <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_logreg)$best; best_logreg$Model <- "LogReg"
best_lda    <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_lda)$best;    best_lda$Model    <- "LDA"
best_qda    <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_qda)$best;    best_qda$Model    <- "QDA"
best_nb     <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_nb)$best;     best_nb$Model     <- "NB"


val_threshold_summary <- dplyr::bind_rows(best_logreg, best_lda, best_qda, best_nb) |>
  dplyr::select(Model, threshold, BalancedAcc, Sensitivity, Specificity, Accuracy, Precision, F1) |>
  dplyr::arrange(dplyr::desc(BalancedAcc), dplyr::desc(Sensitivity), dplyr::desc(Specificity))

print(val_threshold_summary)

best_thresholds <- setNames(val_threshold_summary$threshold, val_threshold_summary$Model)

# use these for plotting curves
all_logreg <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_logreg)$all
all_lda    <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_lda)$all
all_qda    <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_qda)$all
all_nb     <- pick_best_thr_bal(val_sc$Has_Mental_Health_Issue, p_nb)$all







plot_threshold_effect_one <- function(tbl_all, model_name, best_thr = NULL) {
  
  tbl_long <- tbl_all %>%
    dplyr::select(threshold, Accuracy, Precision, Sensitivity, Specificity) %>%
    tidyr::pivot_longer(cols = -threshold, names_to = "Metric", values_to = "Value")
  
  p <- ggplot(tbl_long, aes(x = threshold, y = Value, color = Metric)) +
    geom_line(linewidth = 1) +
    geom_point(size = 1.4) +
    labs(
      title = model_name,
      x = "Probability Threshold",
      y = "Metric Value"
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      legend.position = "none",
      panel.grid.minor = element_blank()
    ) +
    scale_color_viridis_d(end = 0.9)
  
  # optional: best threshold line (from your val_threshold_summary)
  if (!is.null(best_thr) && is.finite(best_thr)) {
    p <- p + geom_vline(xintercept = best_thr, linetype = "solid", color = "black", linewidth = 0.6)
  }
  
  p
}

# pull best thresholds (optional solid line)
thr_logreg <- best_thresholds[["LogReg"]]
thr_lda    <- best_thresholds[["LDA"]]
thr_qda    <- best_thresholds[["QDA"]]
thr_nb     <- best_thresholds[["NB"]]

p1 <- plot_threshold_effect_one(all_logreg, "LogReg", thr_logreg)
p2 <- plot_threshold_effect_one(all_lda,    "LDA",    thr_lda)
p3 <- plot_threshold_effect_one(all_qda,    "QDA",    thr_qda)
p4 <- plot_threshold_effect_one(all_nb,     "Naive Bayes", thr_nb)

# combine 2x2 + shared legend
combined_long <- bind_rows(
  mutate(all_logreg, Model = "LogReg"),
  mutate(all_lda,    Model = "LDA"),
  mutate(all_qda,    Model = "QDA"),
  mutate(all_nb,     Model = "Naive Bayes")
) %>%
  select(Model, threshold, Accuracy, Precision, Sensitivity, Specificity) %>%
  pivot_longer(cols = c(Accuracy, Precision, Sensitivity, Specificity),
               names_to = "Metric", values_to = "Value")

legend_plot <- ggplot(combined_long, aes(x = threshold, y = Value, color = Metric)) +
  geom_line() +
  scale_color_viridis_d(end = 0.9) +
  theme_void(base_size = 12) +
  theme(legend.position = "bottom")

legend <- patchwork::wrap_elements(ggplotGrob(legend_plot + theme(legend.position = "bottom")))

(p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "Effect of Decision Threshold on Classification Metrics (Validation Set)",
    subtitle = "Solid line = best threshold (by Average of Sensitivity and Specificity)"
  ) &
  theme(legend.position = "bottom")




'
Decision threshold selection on the validation set (Balanced Accuracy)

After model training, each classifier outputs predicted probabilities P(Yes) for the validation set. Since the dataset is imbalanced and a fixed threshold of 0.50 can be suboptimal, the decision threshold was tuned using the validation set.

A grid of candidate thresholds from 0.05 to 0.95 (step = 0.01) was evaluated. For each threshold, predicted class labels were generated and a confusion matrix was computed. In addition to standard metrics (Accuracy, Precision, Sensitivity, Specificity, and F1), the main selection criterion was Balanced Accuracy, defined as:

Balanced Accuracy = (Sensitivity + Specificity) / 2

Balanced Accuracy was used because it gives equal importance to both classes by averaging the true-positive rate (Sensitivity) and true-negative rate (Specificity), making it more reliable than Accuracy or F1 in imbalanced settings.

For each model, the threshold that maximized Balanced Accuracy (with tie-breaking toward higher Sensitivity and then higher Specificity) was selected. The resulting optimal thresholds on the validation set were:

LDA: threshold = 0.48, BalancedAcc = 0.646

Naive Bayes: threshold = 0.36, BalancedAcc = 0.646

Logistic Regression: threshold = 0.49, BalancedAcc = 0.644

QDA: threshold = 0.56, BalancedAcc = 0.612

These chosen thresholds were stored and later applied unchanged to the test set to ensure an unbiased final evaluation.
'










# 1 = Yes Majority causes problems when selecting thresholds
# either find a metric that takes 0 into account or make No = 1 and Yes = 0

#We will calculate the probabilities on the TEST set
p_logreg_test <- predict(fit_logreg, newdata = test_sc, type = "prob")[, "Yes"]
p_lda_test    <- predict(fit_lda,    newdata = test_sc, type = "prob")[, "Yes"]
p_qda_test    <- predict(fit_qda,    newdata = test_sc, type = "prob")[, "Yes"]
p_nb_test     <- predict(nb_train, newdata = test_sc, type = "raw")[, "Yes"]

#We will evaluate the model using the BEST thresholds that we find on the validation.
evaluate_test <- function(y_true, prob, threshold, model_name){
  pred_label <- factor(ifelse(prob >= threshold, "Yes", "No"), levels = c("No", "Yes"))
  cm <- caret::confusionMatrix(pred_label, y_true, positive = "Yes")


#Extract the metrics
  data.frame(
    Model = model_name,
    Test_Accuracy = as.numeric(cm$overall["Accuracy"]),
    Test_Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Test_Specificity = as.numeric(cm$byClass["Specificity"]),
    Test_F1 = as.numeric(cm$byClass["F1"]),
    Test_Precision = as.numeric(cm$byClass["Pos Pred Value"]),
    Threshold_Used = threshold
  )
}

#Run the evaluation for each model
Final_Results <- dplyr:: bind_rows(
  evaluate_test(test_sc$Has_Mental_Health_Issue, p_logreg_test, thr_logreg, "LogReg"),
  evaluate_test(test_sc$Has_Mental_Health_Issue, p_lda_test,    thr_lda,    "LDA"),
  evaluate_test(test_sc$Has_Mental_Health_Issue, p_qda_test,    thr_qda,    "QDA"),
  evaluate_test(test_sc$Has_Mental_Health_Issue, p_nb_test,     thr_nb,     "Naive Bayes")
)

# ROC AUC CURVES ON TEST SET + AUC labels

roc_logreg <- roc(test_sc$Has_Mental_Health_Issue, p_logreg_test, levels = c("No","Yes"), direction = "<", quiet = TRUE)
roc_lda    <- roc(test_sc$Has_Mental_Health_Issue, p_lda_test,    levels = c("No","Yes"), direction = "<", quiet = TRUE)
roc_qda    <- roc(test_sc$Has_Mental_Health_Issue, p_qda_test,    levels = c("No","Yes"), direction = "<", quiet = TRUE)
roc_nb     <- roc(test_sc$Has_Mental_Health_Issue, p_nb_test,     levels = c("No","Yes"), direction = "<", quiet = TRUE)

roc_list <- list(
  "LogReg"      = roc_logreg,
  "LDA"         = roc_lda,
  "QDA"         = roc_qda,
  "Naive Bayes" = roc_nb
)

# AUC values
auc_vals <- sapply(roc_list, function(r) as.numeric(auc(r)))

# build label text
auc_text <- paste0(names(auc_vals), ": AUC = ", sprintf("%.3f", auc_vals), collapse = "\n")

ggroc(roc_list, linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(title = "ROC Curves on Test Set", color = "Model") +
  scale_color_viridis_d(end = 0.9) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray50") +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold")) +
  annotate("text",
           x = 0.65, y = 0.25,
           label = auc_text,
           hjust = 0, vjust = 0,
           size = 4)




# ---- Add AUC to Final_Results + pretty print ----

# AUC table
auc_df <- data.frame(
  Model = names(auc_vals),
  Test_AUC = as.numeric(auc_vals),
  row.names = NULL
)

# merge + add Balanced Accuracy
Final_Results2 <- dplyr::left_join(Final_Results, auc_df, by = "Model") %>%
  dplyr::mutate(
    Test_BalancedAcc = 0.5 * (Test_Sensitivity + Test_Specificity)
  ) %>%
  dplyr::select(Model, Test_AUC, Test_BalancedAcc, Threshold_Used,
                Test_Accuracy, Test_Sensitivity, Test_Specificity,
                Test_Precision, Test_F1) %>%
  dplyr::arrange(dplyr::desc(Test_AUC), dplyr::desc(Test_BalancedAcc))

# nicer printing
Final_Results2_print <- Final_Results2 %>%
  dplyr::mutate(
    dplyr::across(c(Test_AUC, Test_BalancedAcc, Threshold_Used, Test_Accuracy,
                    Test_Sensitivity, Test_Specificity, Test_Precision, Test_F1),
                  ~ round(.x, 3))
  )

print(Final_Results2_print, row.names = FALSE)



# ================= Confusion matrices for all models (Test) =================

y_test <- test_sc$Has_Mental_Health_Issue

models_list <- list(
  "LogReg"      = list(prob = p_logreg_test, thr = thr_logreg),
  "LDA"         = list(prob = p_lda_test,    thr = thr_lda),
  "QDA"         = list(prob = p_qda_test,    thr = thr_qda),
  "Naive Bayes" = list(prob = p_nb_test,     thr = thr_nb)
)

for (m in names(models_list)) {
  
  prob <- models_list[[m]]$prob
  thr  <- models_list[[m]]$thr
  
  pred <- factor(ifelse(prob >= thr, "Yes", "No"), levels = c("No","Yes"))
  cm   <- caret::confusionMatrix(pred, y_test, positive = "Yes")
  
  sens <- as.numeric(cm$byClass["Sensitivity"])
  spec <- as.numeric(cm$byClass["Specificity"])
  bal  <- 0.5 * (sens + spec)
  
  cat("\n====================", m, "====================\n")
  cat("Threshold:", round(thr, 3), "\n")
  print(cm$table)
  cat("Sensitivity:", round(sens, 3),
      "| Specificity:", round(spec, 3),
      "| BalancedAcc:", round(bal, 3), "\n")
}



'
        Model Test_AUC Threshold_Used Test_Accuracy Test_Sensitivity Test_Specificity Test_Precision Test_F1
         LDA    0.644           0.44         0.671            0.683            0.538          0.946   0.793
      LogReg    0.642           0.44         0.675            0.687            0.526          0.945   0.796
 Naive Bayes    0.638           0.45         0.624            0.630            0.551          0.943   0.756
         QDA    0.575           0.89         0.604            0.613            0.494          0.935   0.740
         
         ==================== LogReg ====================
Threshold: 0.49 
          Reference
Prediction   No  Yes
       No   101  782
       Yes   55 1061
Sensitivity: 0.576 | Specificity: 0.647 | BalancedAcc: 0.612 

==================== LDA ====================
Threshold: 0.48 
          Reference
Prediction   No  Yes
       No    99  750
       Yes   57 1093
Sensitivity: 0.593 | Specificity: 0.635 | BalancedAcc: 0.614 

==================== QDA ====================
Threshold: 0.56 
          Reference
Prediction  No Yes
       No  111 937
       Yes  45 906
Sensitivity: 0.492 | Specificity: 0.712 | BalancedAcc: 0.602 

==================== Naive Bayes ====================
Threshold: 0.36 
          Reference
Prediction  No Yes
       No  108 851
       Yes  48 992
Sensitivity: 0.538 | Specificity: 0.692 | BalancedAcc: 0.615 
         
  

Final test-set evaluation and model selection

After tuning model-specific decision thresholds on the validation set (using Balanced Accuracy), each model was evaluated once on the held-out test set. For each classifier, predicted probabilities 

P(Yes) were converted to class labels using the corresponding validation-selected threshold, and performance metrics were computed from the resulting confusion matrix. In addition, ROC-AUC values were calculated on the test set to provide a threshold-independent measure of discrimination. The ROC curves were plotted together and the AUC values were annotated on the figure.

To support model selection, the final results table reports, for each model: Test_AUC, Test_BalancedAcc, the chosen threshold, and threshold-dependent metrics (Accuracy, Sensitivity, Specificity, Precision, and F1). Because the dataset is imbalanced and “Yes” is the majority class, Accuracy and F1 can be inflated by majority-class performance; therefore, model comparison primarily emphasized ROC-AUC and Balanced Accuracy, with Sensitivity and Specificity inspected jointly to understand the error trade-off.

On the test set, the top-performing models were Naive Bayes (Test_AUC = 0.663, Test_BalancedAcc = 0.615) and LDA (Test_AUC = 0.662, Test_BalancedAcc = 0.614), followed closely by Logistic Regression (Test_AUC = 0.662, Test_BalancedAcc = 0.612). QDA produced the lowest AUC (0.623) and the lowest Balanced Accuracy (0.602), indicating weaker overall discrimination despite relatively high Specificity.

Although Naive Bayes was marginally best according to AUC and Balanced Accuracy, the differences relative to LDA were negligible (≈0.001). The final model was therefore selected as LDA, because it achieved materially higher sensitivity for the “Yes” class on the test set (0.593 vs 0.538 for Naive Bayes), corresponding to fewer false negatives under the validation-selected threshold. This choice prioritizes detecting “Yes” cases while maintaining a comparable overall discrimination level.
'

'--------------------'





# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# Part 2: Machine Learning Models
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================

#===== k-NEAREST NEIGHBOURS=====

# Prepare data
X_train <- train_sc[, setdiff(names(train_sc), target_col)]
y_train <- train_sc[[target_col]]

X_val <- val_sc[, setdiff(names(val_sc), target_col)]
y_val <- val_sc[[target_col]]

X_test <- test_sc[, setdiff(names(test_sc), target_col)]
y_test <- test_sc[[target_col]]

# Keep numeric variables
num_feature_cols <- names(X_train)[sapply(X_train, is.numeric)]

X_train_num <- as.matrix(X_train[, num_feature_cols])
X_val_num   <- as.matrix(X_val[, num_feature_cols])
X_test_num  <- as.matrix(X_test[, num_feature_cols])

data_info <- data.frame(
  Item = c("Numeric predictors", "Training rows", "Validation rows", "Test rows"),
  Value = c(length(num_feature_cols), nrow(X_train_num), nrow(X_val_num), nrow(X_test_num))
)

print(data_info)

# First model k = 3
set.seed(42)

knn_k3 <- knn(
  train = X_train_num,
  test  = X_val_num,
  cl    = y_train,
  k     = 3
)

cm_k3 <- confusionMatrix(knn_k3, y_val, positive = "Yes")

k3_summary <- data.frame(
  Metric = c("Accuracy","Sensitivity","Specificity","Balanced Accuracy"),
  Value = c(
    as.numeric(cm_k3$overall["Accuracy"]),
    as.numeric(cm_k3$byClass["Sensitivity"]),
    as.numeric(cm_k3$byClass["Specificity"]),
    0.5 * (
      as.numeric(cm_k3$byClass["Sensitivity"]) +
        as.numeric(cm_k3$byClass["Specificity"])
    )
  )
)

print(k3_summary)

cm_k3_table <- as.data.frame(cm_k3$table)
colnames(cm_k3_table) <- c("Prediction","Reference","Count")

print(cm_k3_table)

# Tune k
k_values <- 1:30
bal_acc_k <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  
  set.seed(42)
  
  pred_i <- knn(
    train = X_train_num,
    test  = X_val_num,
    cl    = y_train,
    k     = k_values[i]
  )
  
  cm_i <- confusionMatrix(pred_i, y_val, positive = "Yes")
  
  sens_i <- as.numeric(cm_i$byClass["Sensitivity"])
  spec_i <- as.numeric(cm_i$byClass["Specificity"])
  
  bal_acc_k[i] <- 0.5 * (sens_i + spec_i)
}

best_k <- k_values[which.max(bal_acc_k)]

k_results <- data.frame(
  k = k_values,
  Balanced_Accuracy = bal_acc_k
)

print(k_results)

best_k_table <- data.frame(
  Measure = c("Best k","Best Balanced Accuracy"),
  Value = c(best_k, round(max(bal_acc_k),4))
)

print(best_k_table)

ggplot(k_results, aes(x = k, y = Balanced_Accuracy)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "red") +
  annotate(
    "text",
    x = best_k + 0.5,
    y = min(bal_acc_k) + 0.005,
    label = paste("k =", best_k),
    color = "red",
    hjust = 0
  ) +
  labs(
    title = "k-NN: Balanced Accuracy vs k",
    subtitle = "Validation set",
    x = "k",
    y = "Balanced Accuracy"
  ) +
  theme_minimal(base_size = 13)

# Cross validation
knn_formula <- reformulate(num_feature_cols, response="Has_Mental_Health_Issue")

ctrl_cv <- trainControl(
  method="repeatedcv",
  number=5,
  repeats=3,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  savePredictions="final"
)

set.seed(42)

knn_cv <- train(
  knn_formula,
  data=train_sc,
  method="knn",
  trControl=ctrl_cv,
  tuneGrid=data.frame(k=1:30),
  metric="ROC"
)

cv_results <- knn_cv$results[,c("k","ROC","Sens","Spec")]

print(cv_results)

cv_best <- data.frame(
  Measure=c("Best k","Best ROC"),
  Value=c(knn_cv$bestTune$k, round(max(knn_cv$results$ROC),4))
)

print(cv_best)

plot(knn_cv)

# Threshold selection
p_knn_val <- predict(knn_cv,newdata=val_sc,type="prob")[,"Yes"]

thr_grid <- seq(0.05,0.95,by=0.01)

calc_metrics_bal <- function(y_true,prob_yes,thr){
  
  pred <- factor(
    ifelse(prob_yes >= thr,"Yes","No"),
    levels=c("No","Yes")
  )
  
  cm <- confusionMatrix(pred,y_true,positive="Yes")
  
  sens <- as.numeric(cm$byClass["Sensitivity"])
  spec <- as.numeric(cm$byClass["Specificity"])
  prec <- as.numeric(cm$byClass["Pos Pred Value"])
  acc  <- as.numeric(cm$overall["Accuracy"])
  
  bal <- 0.5*(sens+spec)
  
  data.frame(
    Threshold=thr,
    Accuracy=acc,
    Sensitivity=sens,
    Specificity=spec,
    Balanced_Accuracy=bal,
    Precision=prec
  )
}

thr_results_knn <- bind_rows(
  lapply(thr_grid,function(t) calc_metrics_bal(y_val,p_knn_val,t))
)

best_thr_row <- thr_results_knn %>%
  arrange(desc(Balanced_Accuracy)) %>%
  slice(1)

best_thr_knn <- best_thr_row$Threshold

print(best_thr_row)

# Test results
p_knn_test <- predict(knn_cv,newdata=test_sc,type="prob")[,"Yes"]

pred_knn_test <- factor(
  ifelse(p_knn_test >= best_thr_knn,"Yes","No"),
  levels=c("No","Yes")
)

cm_test <- confusionMatrix(pred_knn_test,y_test,positive="Yes")

roc_knn <- roc(
  y_test,
  p_knn_test,
  levels=c("No","Yes"),
  direction="<",
  quiet=TRUE
)

auc_knn <- as.numeric(auc(roc_knn))

sens_test <- as.numeric(cm_test$byClass["Sensitivity"])
spec_test <- as.numeric(cm_test$byClass["Specificity"])
prec_test <- as.numeric(cm_test$byClass["Pos Pred Value"])
f1_test   <- as.numeric(cm_test$byClass["F1"])
acc_test  <- as.numeric(cm_test$overall["Accuracy"])

bal_test <- 0.5*(sens_test+spec_test)

knn_test_summary <- data.frame(
  Model=paste0("kNN (k=",knn_cv$bestTune$k,")"),
  Test_AUC=round(auc_knn,3),
  Balanced_Accuracy=round(bal_test,3),
  Threshold=round(best_thr_knn,2),
  Accuracy=round(acc_test,3),
  Sensitivity=round(sens_test,3),
  Specificity=round(spec_test,3),
  Precision=round(prec_test,3),
  F1=round(f1_test,3)
)

print(knn_test_summary)

cm_test_table <- as.data.frame(cm_test$table)
colnames(cm_test_table) <- c("Prediction","Reference","Count")

print(cm_test_table)

# ROC plot
ggroc(roc_knn,linewidth=1,color="steelblue") +
  geom_abline(slope=1,intercept=1,linetype="dashed") +
  labs(
    title="k-NN ROC Curve",
    subtitle=paste("AUC =",round(auc_knn,3)),
    x="Specificity",
    y="Sensitivity"
  ) +
  theme_minimal()

# Confusion matrix plot
cm_df <- as.data.frame(cm_test$table)
colnames(cm_df) <- c("Prediction","Reference","Freq")

ggplot(cm_df,aes(x=Reference,y=Prediction,fill=Freq)) +
  geom_tile(color="white") +
  geom_text(aes(label=Freq),size=7) +
  scale_fill_gradient(low="white",high="steelblue") +
  labs(
    title="k-NN Confusion Matrix",
    x="Actual",
    y="Predicted"
  ) +
  theme_minimal()

knn_summary <- data.frame(
  Model = c("k-NN"),
  Main_Setting = c(paste0("k = ", knn_cv$bestTune$k)),
  Threshold = c(round(best_thr_knn, 2)),
  Test_AUC = c(round(auc_knn, 3)),
  Balanced_Accuracy = c(round(bal_test, 3)),
  Sensitivity = c(round(sens_test, 3)),
  Specificity = c(round(spec_test, 3))
)

print(knn_summary)

#==============================================================================
# k-Nearest Neighbours 
#
# The k-NN model performed worst among the tested models, achieving a
# test AUC of 0.572, which is only slightly above random guessing (0.5).
# This is lower than the best model from Part 1 (LDA, AUC = 0.644).
#
# The weak performance is mainly due to two reasons. First, with 21
# predictors, the differences between observations become less useful,
# which reduces the effectiveness of nearest-neighbors methods.
# Second, the class imbalance in the data affects how k-NN identifies
# similar points in the neighborhood.
#
# The large gap between the cross-validation ROC (0.924) and the test
# performance occurs because cross-validation was performed on training
# data balanced with SMOTE, while the test set reflects the true class
# distribution.
#
# Overall, k-NN does not perform well for this problem. Linear models
# such as LDA and Logistic Regression perform better because they use
# patterns from the whole dataset rather than relying only on nearby
# observations.

# Key results:
#   Best k:             5 (cross-validation)
#   Threshold:          0.81
#   Test AUC:           0.572
#   Balanced Accuracy:  0.530
#   Sensitivity:        0.239
#   Specificity:        0.821
# ==============================================================================

# Threshold helper

calc_metrics_bal <- function(y_true, prob_yes, thr) {
  pred <- factor(
    ifelse(prob_yes >= thr, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  cm <- caret::confusionMatrix(pred, y_true, positive = "Yes")
  
  sens <- as.numeric(cm$byClass["Sensitivity"])
  spec <- as.numeric(cm$byClass["Specificity"])
  bal  <- 0.5 * (sens + spec)
  prec <- as.numeric(cm$byClass["Pos Pred Value"])
  
  f1 <- if (is.na(prec) || is.na(sens) || (prec + sens) == 0) {
    NA_real_
  } else {
    2 * prec * sens / (prec + sens)
  }
  
  data.frame(
    threshold = thr,
    Sensitivity = sens,
    Specificity = spec,
    BalancedAcc = bal,
    Precision = prec,
    F1 = f1,
    Accuracy = as.numeric(cm$overall["Accuracy"])
  )
}

pick_best_thr <- function(y_true, prob_yes) {
  thr_grid <- seq(0.05, 0.95, by = 0.01)
  
  tbl <- dplyr::bind_rows(
    lapply(thr_grid, function(t) calc_metrics_bal(y_true, prob_yes, t))
  )
  
  tbl %>%
    arrange(desc(BalancedAcc), desc(Sensitivity), desc(Specificity)) %>%
    slice(1)
}

# Initial decision tree

set.seed(42)
mental_tree_init <- tree(
  Has_Mental_Health_Issue ~ .,
  data = train_smote,
  control = tree.control(
    nobs = nrow(train_smote),
    mindev = 0.01,
    minsize = 10
  )
)

tree_init_summary <- data.frame(
  Measure = c("Tree type", "mindev", "minsize"),
  Value = c("Initial decision tree", 0.01, 10)
)

print(tree_init_summary)
print(summary(mental_tree_init))

plot(mental_tree_init)
text(mental_tree_init, pretty = 0, cex = 0.75)
title("Mental Health: Initial Classification Tree")

# Full tree and pruning

set.seed(42)
mental_tree_full <- tree(
  Has_Mental_Health_Issue ~ .,
  data = train_smote,
  control = tree.control(
    nobs = nrow(train_smote),
    mindev = 0,
    minsize = 2
  )
)

full_tree_info <- data.frame(
  Measure = c("Tree type", "Terminal nodes"),
  Value = c(
    "Full unpruned tree",
    mental_tree_full$frame %>% filter(var == "<leaf>") %>% nrow()
  )
)

print(full_tree_info)

set.seed(42)
mental_cv <- cv.tree(mental_tree_full, FUN = prune.misclass)

cv_results_tree <- data.frame(
  size = mental_cv$size,
  cv_error = mental_cv$dev
)

print(cv_results_tree)

min_dev <- min(mental_cv$dev, na.rm = TRUE)
best_size <- min(mental_cv$size[mental_cv$dev == min_dev], na.rm = TRUE)

best_size_table <- data.frame(
  Measure = c("Minimum CV error", "Best tree size"),
  Value = c(min_dev, best_size)
)

print(best_size_table)

tibble(size = mental_cv$size, cv_error = mental_cv$dev) %>%
  ggplot(aes(x = size, y = cv_error)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_vline(xintercept = best_size, linetype = "dashed", color = "red") +
  annotate(
    "text",
    x = best_size + 0.3,
    y = max(mental_cv$dev) * 0.98,
    label = paste("best =", best_size),
    color = "red",
    hjust = 0
  ) +
  labs(
    title = "Decision Tree: CV Error vs Tree Size",
    subtitle = paste("Best size =", best_size),
    x = "Number of terminal nodes",
    y = "CV misclassification count"
  ) +
  theme_minimal(base_size = 13)

mental_tree_pruned <- prune.misclass(mental_tree_full, best = best_size)

plot(mental_tree_pruned)
text(mental_tree_pruned, pretty = 0, cex = 0.8)
title(paste("Mental Health: Pruned Tree (", best_size, " leaves)"))

# Tree threshold selection

tree_prob_val <- predict(mental_tree_pruned, newdata = mental_val, type = "vector")
p_tree_val <- as.numeric(tree_prob_val[, "Yes"])

best_thr_tree_row <- pick_best_thr(mental_val[[target_col]], p_tree_val)
best_thr_tree <- best_thr_tree_row$threshold

print(best_thr_tree_row)

# Tree test evaluation

tree_prob_test <- predict(mental_tree_pruned, newdata = mental_test, type = "vector")
p_tree_test <- as.numeric(tree_prob_test[, "Yes"])

pred_tree_test <- factor(
  ifelse(p_tree_test >= best_thr_tree, "Yes", "No"),
  levels = c("No", "Yes")
)

y_test <- mental_test[[target_col]]
cm_tree <- confusionMatrix(pred_tree_test, y_test, positive = "Yes")
roc_tree <- roc(y_test, p_tree_test, levels = c("No", "Yes"), direction = "<", quiet = TRUE)
auc_tree <- as.numeric(auc(roc_tree))

sens_tree <- as.numeric(cm_tree$byClass["Sensitivity"])
spec_tree <- as.numeric(cm_tree$byClass["Specificity"])
bal_tree <- 0.5 * (sens_tree + spec_tree)
acc_tree <- as.numeric(cm_tree$overall["Accuracy"])
f1_tree <- as.numeric(cm_tree$byClass["F1"])

tree_test_info <- data.frame(
  Measure = c("Best tree size", "Threshold", "Test AUC"),
  Value = c(best_size, best_thr_tree, round(auc_tree, 4))
)

print(tree_test_info)
print(cm_tree)

metrics_tree <- tibble(
  Model = paste0("Decision Tree (", best_size, " leaves)"),
  Test_AUC = round(auc_tree, 3),
  Test_BalancedAcc = round(bal_tree, 3),
  Threshold = best_thr_tree,
  Test_Sensitivity = round(sens_tree, 3),
  Test_Specificity = round(spec_tree, 3),
  Test_Accuracy = round(acc_tree, 3),
  Test_F1 = round(f1_tree, 3)
)

print(metrics_tree)

ggroc(roc_tree, linewidth = 1, color = "steelblue") +
  theme_minimal(base_size = 14) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray50") +
  labs(
    title = "Decision Tree ROC Curve",
    subtitle = paste0(
      "AUC = ", round(auc_tree, 3),
      " | Size = ", best_size,
      " | Threshold = ", best_thr_tree
    )
  ) +
  annotate(
    "text",
    x = 0.4,
    y = 0.15,
    label = paste0("AUC = ", round(auc_tree, 3)),
    size = 5,
    color = "steelblue"
  ) +
  theme(plot.title = element_text(face = "bold"))

cm_tree_df <- as.data.frame(cm_tree$table)
colnames(cm_tree_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_tree_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = paste0("Decision Tree Confusion Matrix (thr = ", best_thr_tree, ")"),
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none"
  )

# Bagging

p <- ncol(train_smote) - 1

set.seed(42)
mental_bag <- randomForest(
  Has_Mental_Health_Issue ~ .,
  data = train_smote,
  mtry = p,
  ntree = 350,
  importance = TRUE
)

bagging_info <- data.frame(
  Measure = c("Model", "mtry", "ntree"),
  Value = c("Bagging", p, 350)
)

print(bagging_info)
print(mental_bag)

bag_prob_val <- predict(mental_bag, newdata = mental_val, type = "prob")[, "Yes"]
best_thr_bag_row <- pick_best_thr(mental_val[[target_col]], bag_prob_val)

bag_validation <- data.frame(
  Measure = c("Best threshold", "Validation Balanced Accuracy"),
  Value = c(best_thr_bag_row$threshold, round(best_thr_bag_row$BalancedAcc, 4))
)

print(bag_validation)

# Random forest mtry tuning

base_mtry <- max(1, floor(sqrt(p)))
mtry_grid <- sort(unique(pmax(1, pmin(p, c(base_mtry - 1, base_mtry, base_mtry + 1, floor(p / 3), p)))))

mtry_grid_table <- data.frame(
  mtry = mtry_grid
)

print(mtry_grid_table)

rf_tuning <- map_dfr(mtry_grid, function(m) {
  set.seed(42 + m)
  
  fit <- randomForest(
    Has_Mental_Health_Issue ~ .,
    data = train_smote,
    mtry = m,
    ntree = 250
  )
  
  tibble(
    mtry = m,
    OOB_Error = fit$err.rate[250, "OOB"]
  )
})

best_mtry <- rf_tuning$mtry[which.min(rf_tuning$OOB_Error)]

print(rf_tuning)

best_mtry_table <- data.frame(
  Measure = c("Best mtry", "Minimum OOB error"),
  Value = c(best_mtry, min(rf_tuning$OOB_Error))
)

print(best_mtry_table)

ggplot(rf_tuning, aes(x = mtry, y = OOB_Error)) +
  geom_line(color = "#41ab5d", linewidth = 1) +
  geom_point(color = "#41ab5d", size = 3) +
  geom_vline(xintercept = best_mtry, linetype = "dashed", color = "red") +
  annotate(
    "text",
    x = best_mtry + 0.2,
    y = max(rf_tuning$OOB_Error),
    label = paste("best mtry =", best_mtry),
    color = "red",
    hjust = 0
  ) +
  labs(
    title = "Random Forest: OOB Error vs mtry",
    x = "mtry",
    y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Final random forest

set.seed(42)
mental_rf <- randomForest(
  Has_Mental_Health_Issue ~ .,
  data = train_smote,
  mtry = best_mtry,
  ntree = 350,
  importance = TRUE
)

rf_info <- data.frame(
  Measure = c("Model", "Best mtry", "ntree"),
  Value = c("Random Forest", best_mtry, 350)
)

print(rf_info)
print(mental_rf)

oob_df <- tibble(
  Trees = seq_len(nrow(mental_rf$err.rate)),
  OOB_Error = mental_rf$err.rate[, "OOB"]
)

ggplot(oob_df, aes(x = Trees, y = OOB_Error)) +
  geom_line(color = "#2c7fb8", linewidth = 1) +
  labs(
    title = "Random Forest: OOB Error vs Number of Trees",
    subtitle = "Stabilises before ntree = 350",
    x = "Number of Trees",
    y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Variable importance

varImpPlot(
  mental_rf,
  main = "Mental Health RF Variable Importance",
  cex = 0.8
)

imp_df <- as.data.frame(importance(mental_rf))
imp_df$Variable <- rownames(imp_df)

imp_cols <- intersect(c("MeanDecreaseAccuracy", "MeanDecreaseGini"), names(imp_df))

imp_long <- imp_df %>%
  select(Variable, all_of(imp_cols)) %>%
  pivot_longer(cols = -Variable, names_to = "Measure", values_to = "Importance")

ggplot(
  imp_long,
  aes(x = fct_reorder(Variable, Importance), y = Importance, fill = Measure)
) +
  geom_col(position = "dodge", alpha = 0.85) +
  coord_flip() +
  facet_wrap(~ Measure, scales = "free_x") +
  labs(
    title = "Random Forest Variable Importance",
    x = NULL,
    y = "Importance"
  ) +
  scale_fill_viridis_d() +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold")
  )

# RF threshold selection

rf_prob_val <- predict(mental_rf, newdata = mental_val, type = "prob")[, "Yes"]
best_thr_rf_row <- pick_best_thr(mental_val[[target_col]], rf_prob_val)
best_thr_rf <- best_thr_rf_row$threshold

print(best_thr_rf_row)

# RF test evaluation

rf_prob_test <- predict(mental_rf, newdata = mental_test, type = "prob")[, "Yes"]

pred_rf_test <- factor(
  ifelse(rf_prob_test >= best_thr_rf, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_rf <- confusionMatrix(pred_rf_test, y_test, positive = "Yes")
roc_rf <- roc(y_test, rf_prob_test, levels = c("No", "Yes"), direction = "<", quiet = TRUE)
auc_rf <- as.numeric(auc(roc_rf))

sens_rf <- as.numeric(cm_rf$byClass["Sensitivity"])
spec_rf <- as.numeric(cm_rf$byClass["Specificity"])
bal_rf <- 0.5 * (sens_rf + spec_rf)
acc_rf <- as.numeric(cm_rf$overall["Accuracy"])
f1_rf <- as.numeric(cm_rf$byClass["F1"])

rf_test_info <- data.frame(
  Measure = c("Best mtry", "ntree", "Threshold", "Test AUC"),
  Value = c(best_mtry, 350, best_thr_rf, round(auc_rf, 4))
)

print(rf_test_info)
print(cm_rf)

metrics_rf <- tibble(
  Model = paste0("Random Forest (mtry=", best_mtry, ")"),
  Test_AUC = round(auc_rf, 3),
  Test_BalancedAcc = round(bal_rf, 3),
  Threshold = best_thr_rf,
  Test_Sensitivity = round(sens_rf, 3),
  Test_Specificity = round(spec_rf, 3),
  Test_Accuracy = round(acc_rf, 3),
  Test_F1 = round(f1_rf, 3)
)

print(metrics_rf)

ggroc(roc_rf, linewidth = 1, color = "#41ab5d") +
  theme_minimal(base_size = 14) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray50") +
  labs(
    title = "Random Forest ROC Curve",
    subtitle = paste0(
      "AUC = ", round(auc_rf, 3),
      " | mtry = ", best_mtry,
      " | Threshold = ", best_thr_rf
    )
  ) +
  annotate(
    "text",
    x = 0.4,
    y = 0.15,
    label = paste0("AUC = ", round(auc_rf, 3)),
    size = 5,
    color = "#41ab5d"
  ) +
  theme(plot.title = element_text(face = "bold"))

cm_rf_df <- as.data.frame(cm_rf$table)
colnames(cm_rf_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_rf_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#41ab5d") +
  labs(
    title = paste0("Random Forest Confusion Matrix (thr = ", best_thr_rf, ")"),
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none"
  )

# Tree and RF summary

tree_rf_summary <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Main_Setting = c(
    paste0(best_size, " leaves"),
    paste0("mtry = ", best_mtry)
  ),
  Threshold = c(round(best_thr_tree, 2), round(best_thr_rf, 2)),
  Test_AUC = c(round(auc_tree, 3), round(auc_rf, 3)),
  Balanced_Accuracy = c(round(bal_tree, 3), round(bal_rf, 3)),
  Sensitivity = c(round(sens_tree, 3), round(sens_rf, 3)),
  Specificity = c(round(spec_tree, 3), round(spec_rf, 3))
)

print(tree_rf_summary)

# =============================================================================
# Decision Tree

#
# The Decision Tree achieved a test AUC of 0.554, which is only slightly
# above random guessing and is the weakest result among all models tested.
#
# The initial tree splits first on Social_Support < 3.0, identifying it
# as the strongest single predictor. A full tree with 475 terminal nodes
# was first grown, and cross-validation was then used to select the optimal
# tree size.
#
# The pruning curve shows that the cross-validation error decreases quickly
# up to 28 terminal nodes and then stabilises. Additional splits beyond
# this point do not improve predictive performance, suggesting that the
# dataset does not contain a complex hierarchical structure that deeper
# trees could exploit.
#
# The pruned tree with 28 terminal nodes uses Social_Support in most
# branches, with Work_Stress_Level, Mood_Swings, and Suicidal_Thoughts
# appearing at deeper levels.
#
# Although the tree is easy to interpret, a single decision tree tends to
# overfit small patterns in the SMOTE-balanced training data and therefore
# does not generalise well to the real imbalanced test data.
#
# Key results:
#   Terminal nodes:     28 (after pruning)
#   Threshold:          0.79
#   Test AUC:           0.554
#   Balanced Accuracy:  0.549
#   Sensitivity:        0.560
#   Specificity:        0.538
#
# Figures:
#   - Initial Classification Tree
#   - CV Error vs Tree Size (pruning curve, best = 28)
#   - Pruned Tree (28 leaves)
#   - ROC Curve (Test Set, AUC = 0.554)
#   - Confusion Matrix (Test Set, threshold = 0.79)


# =============================================================================
# Random Forest


# The Random Forest achieved a test AUC of 0.603, which is the best result
# among the Part 2 models. However, it is still below the best models from
# Part 1, particularly LDA (AUC = 0.644).
#
# Tuning the number of predictors used at each split showed that the lowest
# out-of-bag error occurs when mtry = 3. The error increases as more
# predictors are considered at each split. This indicates that reducing the
# number of variables per split helps decorrelate the trees and improves
# overall performance.
#
# The out-of-bag error curve stabilises around 100 trees, indicating that
# using 350 trees is sufficient for the model to converge.
#
# Variable importance results are consistent across both measures.
# Social_Support, Work_Stress_Level, Mood_Swings, and Anxious_Nervous
# appear as the most influential predictors. Variables such as
# Self_Harm_Thoughts, Trauma_History, and Smoking have the lowest
# importance scores and contribute little to predictive performance.
#
# Random Forest improves over a single Decision Tree by +0.049 AUC by
# averaging the predictions of many trees. This reduces variance without
# substantially increasing bias. However, the model still does not
# outperform the linear models, suggesting that the mental health dataset
# may follow mostly linear decision patterns.
#
# Key results:
#   Best mtry:          3
#   ntree:              350
#   Threshold:          0.70
#   Test AUC:           0.603
#   Balanced Accuracy:  0.582
#   Sensitivity:        0.607
#   Specificity:        0.558
#
# Figures:
#   - OOB Error vs mtry (best mtry = 3)
#   - OOB Error vs Number of Trees (stabilises around 100)
#   - Variable Importance (MeanDecreaseAccuracy + MeanDecreaseGini)
#   - ROC Curve (Test Set, AUC = 0.603)
#   - Confusion Matrix (Test Set, threshold = 0.70)



# FINAL RANKING — ALL MODELS 

#
#   Model             AUC    Bal.Acc   Sensitivity   Specificity
#   LDA               0.644  0.614     0.593         0.635
#   LogReg            0.642  0.612     0.576         0.647
#   Naive Bayes       0.638  0.615     0.538         0.692
#   Random Forest     0.603  0.582     0.607         0.558
#   QDA               0.575  0.602     0.492         0.712
#   k-NN              0.572  0.530     0.239         0.821
#   Decision Tree     0.554  0.549     0.560         0.538