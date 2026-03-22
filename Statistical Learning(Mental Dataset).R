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
library (tree)



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
library(randomForest)
library(xgboost)
library(kknn)
library(tree)
library(class)
library(keras3)
library(tensorflow)


# HELPER FUNCTIONS
# Calculate metrics for a specific threshold
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
# Threshold picker
pick_best_thr_bal <- function(y_true, prob_yes) {
  tbl <- dplyr::bind_rows(lapply(thr_grid, function(t) calc_metrics_bal(y_true, prob_yes, t))) |>
    dplyr::arrange(dplyr::desc(BalancedAcc), dplyr::desc(Sensitivity), dplyr::desc(Specificity))
  list(best = tbl[1, ], all = tbl)
}
# Threshold selection plot
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
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    ) +
    scale_color_viridis_d(end = 0.9)
  
  # optional: best threshold line (from your val_threshold_summary)
  if (!is.null(best_thr) && is.finite(best_thr)) {
    p <- p + geom_vline(xintercept = best_thr, linetype = "solid", color = "black", linewidth = 0.6)
  }
  p
}
# apply SMOTE + scaling for a given feature set
prepare_data <- function(train, val, test, feature_cols, scale = TRUE) {
  
  keep <- c("Has_Mental_Health_Issue", feature_cols)
  
  tr <- train[, keep]
  va <- val[,   keep]
  te <- test[,  keep]
  
  # SMOTE on training set
  tr_smote <- SMOTE(Has_Mental_Health_Issue ~ ., data = tr,
                    perc.over = 600, perc.under = 100)
  tr_smote$Has_Mental_Health_Issue <- factor(
    tr_smote$Has_Mental_Health_Issue, levels = c("No", "Yes")
  )
  
  if (scale) {
    # Scaling — fit on SMOTE'd train, apply to all
    num_cols <- setdiff(names(tr_smote)[sapply(tr_smote, is.numeric)],
                        "Has_Mental_Health_Issue")
    mu <- colMeans(tr_smote[, num_cols])
    sd <- apply(tr_smote[, num_cols], 2, sd)
    sd[sd == 0] <- 1
    
    list(
      train = scale_apply(tr_smote, num_cols, mu, sd),
      val   = scale_apply(va,       num_cols, mu, sd),
      test  = scale_apply(te,       num_cols, mu, sd)
    )
  } else {
    # No scaling — return SMOTE'd data as is
    list(train = tr_smote, val = va, test = te)
  }
}


# Load dataset
mental = read.csv("mental_health.csv")

# Check size and structure of data
dim(mental)     # number of rows and columns
str(mental)     # variable types


# Data Preparation
# Convert target variable to factor (classification task)


mental$Has_Mental_Health_Issue <- factor(
  mental$Has_Mental_Health_Issue,
  levels = c("0", "1"),
  labels = c("No", "Yes")
)

#Convert all the character variables to factors
mental[] = lapply(mental, function(x) if(is.character(x)) as.factor(x) else x)

# We will start the data splitting (60% train, 20% validation, 20% test) 
set.seed(42)  # for reproducibility
idx_train = createDataPartition(mental$Has_Mental_Health_Issue, p = 0.60, list = FALSE)
mental_train = mental[idx_train, ]
mental_tmp   = mental[-idx_train, ]

idx_val = createDataPartition(mental_tmp$Has_Mental_Health_Issue, p = 0.50, list = FALSE)
mental_val  = mental_tmp[idx_val, ]
mental_test = mental_tmp[-idx_val, ]

#===========================================================
#================== k-NEAREST NEIGHBOURS ===================
#===========================================================

# Step 1: Feature selection using RF on original training data (before SMOTE)
set.seed(42)
rf_fs <- randomForest(
  Has_Mental_Health_Issue ~ .,
  data       = mental_train,  # original data, not SMOTE'd
  ntree      = 500,
  importance = TRUE
)

# Get importance scores
imp <- importance(rf_fs, type = 1)  # Mean Decrease Accuracy
imp_df <- data.frame(
  Feature    = rownames(imp),
  Importance = imp[, 1]
) %>% arrange(desc(Importance))

print(imp_df)

# Plot feature importance
ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title    = "Random Forest Feature Importance",
    subtitle = "Feature selection for k-NN (trained on original data)",
    x = "", y = "Mean Decrease Accuracy"
  ) +
  theme_minimal(base_size = 13)

# Select features above mean importance
# Only keep features with positive importance above mean of positive values
positive_imp <- imp_df$Importance[imp_df$Importance > 0]

selected_knn_features <- imp_df$Feature[imp_df$Importance > mean(positive_imp)]
cat("Selected features for kNN:", length(selected_knn_features), "\n")
print(selected_knn_features)

# Remove binary (0/1) and categorical variables
unique_counts <- sapply(mental_train[, selected_knn_features], function(x) length(unique(x)))
print(sort(unique_counts))

binary_or_cat <- sapply(mental_train[, selected_knn_features], function(x) {
  length(unique(x)) <= 2 | is.factor(x)
})

knn_features_final <- selected_knn_features[!binary_or_cat]

cat("Final kNN features (excluding binary & categorical):\n")
cat("Count:", length(knn_features_final), "\n")
print(knn_features_final)

# Step 2: SMOTE + Scaling using prepare_data function
knn_data <- prepare_data(mental_train, mental_val, mental_test, knn_features_final, scale = TRUE)

train_sc_knn <- knn_data$train
val_sc_knn   <- knn_data$val
test_sc_knn  <- knn_data$test

# Step 3: Prepare matrices for class::knn
target_col       <- "Has_Mental_Health_Issue"
num_feature_cols <- knn_features_final

X_train_num <- as.matrix(train_sc_knn[, num_feature_cols])
X_val_num   <- as.matrix(val_sc_knn[,   num_feature_cols])
X_test_num  <- as.matrix(test_sc_knn[,  num_feature_cols])

y_train <- train_sc_knn[[target_col]]
y_val   <- val_sc_knn[[target_col]]
y_test  <- test_sc_knn[[target_col]]

data_info <- data.frame(
  Item  = c("Numeric predictors", "Training rows", "Validation rows", "Test rows"),
  Value = c(length(num_feature_cols), nrow(X_train_num), nrow(X_val_num), nrow(X_test_num))
)
print(data_info)

# Step 4: Baseline model k = 3
set.seed(42)

knn_k3 <- class::knn(
  train = X_train_num,
  test  = X_val_num,
  cl    = y_train,
  k     = 3
)

cm_k3 <- confusionMatrix(knn_k3, y_val, positive = "Yes")

k3_summary <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Balanced Accuracy"),
  Value  = c(
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
colnames(cm_k3_table) <- c("Prediction", "Reference", "Count")
print(cm_k3_table)

# Step 5: Cross-validation to select best k
knn_formula <- reformulate(num_feature_cols, response = "Has_Mental_Health_Issue")

ctrl_cv <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

set.seed(42)

knn_cv <- train(
  knn_formula,
  data      = train_sc_knn,
  method    = "knn",
  trControl = ctrl_cv,
  tuneGrid  = data.frame(k = seq(1, 30, by = 2)),
  metric    = "ROC"
)

# CV results table
cv_results <- knn_cv$results[, c("k", "ROC", "Sens", "Spec")]
print(cv_results)

cv_best <- data.frame(
  Measure = c("Best k", "Best ROC"),
  Value   = c(knn_cv$bestTune$k, round(max(knn_cv$results$ROC), 4))
)
print(cv_best)

# CV ROC-AUC vs k plot
ggplot(knn_cv$results, aes(x = k, y = ROC)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  geom_vline(xintercept = knn_cv$bestTune$k, linetype = "dashed", color = "red") +
  annotate(
    "text",
    x     = knn_cv$bestTune$k + 0.5,
    y     = min(knn_cv$results$ROC) + 0.002,
    label = paste("k =", knn_cv$bestTune$k),
    color = "red",
    hjust = 0
  ) +
  labs(
    title    = "k-NN: ROC-AUC vs k",
    subtitle = "5-fold repeated CV (3 repeats)",
    x = "k",
    y = "ROC-AUC"
  ) +
  theme_minimal(base_size = 13)

# Step 6: Threshold — default 0.50 (kNN produces coarse probabilities)
# kNN outputs only k+1 unique probability values, making threshold
# optimisation unreliable — default 0.50 is used instead.
best_thr_knn <- 0.50

# Step 7: Final evaluation on test set
p_knn_test <- predict(knn_cv, newdata = test_sc_knn, type = "prob")[, "Yes"]

pred_knn_test <- factor(
  ifelse(p_knn_test >= best_thr_knn, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_test <- confusionMatrix(pred_knn_test, y_test, positive = "Yes")

roc_knn <- roc(
  y_test, p_knn_test,
  levels    = c("No", "Yes"),
  direction = "<",
  quiet     = TRUE
)

auc_knn   <- as.numeric(auc(roc_knn))
sens_test <- as.numeric(cm_test$byClass["Sensitivity"])
spec_test <- as.numeric(cm_test$byClass["Specificity"])
prec_test <- as.numeric(cm_test$byClass["Pos Pred Value"])
f1_test   <- as.numeric(cm_test$byClass["F1"])
acc_test  <- as.numeric(cm_test$overall["Accuracy"])
bal_test  <- 0.5 * (sens_test + spec_test)

knn_test_summary <- data.frame(
  Model             = paste0("kNN (k=", knn_cv$bestTune$k, ")"),
  Test_AUC          = round(auc_knn,  3),
  Balanced_Accuracy = round(bal_test, 3),
  Threshold         = round(best_thr_knn, 2),
  Accuracy          = round(acc_test, 3),
  Sensitivity       = round(sens_test, 3),
  Specificity       = round(spec_test, 3),
  Precision         = round(prec_test, 3),
  F1                = round(f1_test,  3)
)
print(knn_test_summary)

cm_test_table <- as.data.frame(cm_test$table)
colnames(cm_test_table) <- c("Prediction", "Reference", "Count")
print(cm_test_table)

# ROC curve plot
ggroc(roc_knn, linewidth = 1, color = "steelblue") +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title    = "k-NN ROC Curve",
    subtitle = paste("AUC =", round(auc_knn, 3)),
    x = "Specificity",
    y = "Sensitivity"
  ) +
  theme_minimal()

# Confusion matrix plot
cm_df <- as.data.frame(cm_test$table)
colnames(cm_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = "k-NN Confusion Matrix",
    x = "Actual",
    y = "Predicted"
  ) +
  theme_minimal()

# Final summary
knn_summary <- data.frame(
  Model             = "k-NN",
  Main_Setting      = paste0("k = ", knn_cv$bestTune$k),
  Threshold         = round(best_thr_knn, 2),
  Test_AUC          = round(auc_knn,  3),
  Balanced_Accuracy = round(bal_test, 3),
  Sensitivity       = round(sens_test, 3),
  Specificity       = round(spec_test, 3)
)
print(knn_summary)

# Decision boundary plot using top 2 RF features
top2 <- knn_features_final[1:2]  # Anxious_Nervous, Feel_Understood

# Create grid over 2D feature space
x_range <- seq(min(train_sc_knn[[top2[1]]]) - 0.5,
               max(train_sc_knn[[top2[1]]]) + 0.5,
               length.out = 150)
y_range <- seq(min(train_sc_knn[[top2[2]]]) - 0.5,
               max(train_sc_knn[[top2[2]]]) + 0.5,
               length.out = 150)

grid_df <- expand.grid(setNames(list(x_range, y_range), top2))

# Fill remaining features with their mean from training set
for (col in setdiff(num_feature_cols, top2)) {
  grid_df[[col]] <- mean(train_sc_knn[[col]])
}

# Predict on grid
grid_pred <- predict(knn_cv, newdata = grid_df, type = "raw")
grid_df$Predicted <- grid_pred

# Train plot
p_train <- ggplot() +
  geom_tile(
    data  = grid_df,
    aes(x = .data[[top2[1]]], y = .data[[top2[2]]], fill = Predicted),
    alpha = 0.2
  ) +
  geom_jitter(
    data   = train_sc_knn,
    aes(x  = .data[[top2[1]]], y = .data[[top2[2]]], color = Has_Mental_Health_Issue),
    size   = 1.5, alpha = 0.6, shape = 16,
    width  = 0.05, height = 0.05
  ) +
  scale_fill_manual(values  = c("No" = "#4393c3", "Yes" = "#d6604d")) +
  scale_color_manual(values = c("No" = "#1a6099", "Yes" = "#b2182b")) +
  labs(
    title = "Training Set",
    x     = top2[1], y = top2[2],
    fill  = "Predicted", color = "Actual"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# Test plot
p_test <- ggplot() +
  geom_tile(
    data  = grid_df,
    aes(x = .data[[top2[1]]], y = .data[[top2[2]]], fill = Predicted),
    alpha = 0.2
  ) +
  geom_jitter(
    data   = test_sc_knn,
    aes(x  = .data[[top2[1]]], y = .data[[top2[2]]], color = Has_Mental_Health_Issue),
    size   = 2, alpha = 0.7, shape = 16,
    width  = 0.05, height = 0.05
  ) +
  scale_fill_manual(values  = c("No" = "#4393c3", "Yes" = "#d6604d")) +
  scale_color_manual(values = c("No" = "#1a6099", "Yes" = "#b2182b")) +
  labs(
    title = "Test Set",
    x     = top2[1], y = top2[2],
    fill  = "Predicted", color = "Actual"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# Combine
p_train + p_test +
  plot_annotation(
    title    = "k-NN Decision Boundary",
    subtitle = paste0("Top 2 features: ", top2[1], " vs ", top2[2],
                      " | k = ", knn_cv$bestTune$k,
                      " | Other features fixed at training mean",
                      " | Jitter applied for discrete Likert values")
  )
#==============================================================================
# k-Nearest Neighbours
#
# FEATURE SELECTION
# k-NN requires a separate feature selection step because it relies on
# Euclidean distance rather than model coefficients. Random Forest importance
# (Mean Decrease Accuracy) was used, as it captures non-linear relationships
# and ranks features by their contribution to predictive accuracy — more
# appropriate for a distance-based method. Features with negative or
# near-zero importance were excluded, and binary/categorical variables were
# removed since Euclidean distance is not meaningful for 0/1 values.
# This left 8 Likert-scale predictors.
#
# CLASS IMBALANCE
# SMOTE was applied to balance the training set before scaling. Note that
# SMOTE interpolates between existing observations to generate synthetic
# minority class samples. For Likert-scale variables (e.g. 1-10), this
# produces non-integer values (e.g. 3.7) that do not exist in the original
# data. This distorts the distance space that k-NN relies on, and is a
# known limitation of applying SMOTE to ordinal data.
#
# THRESHOLD SELECTION
# k-NN probability estimates are inherently coarse: with k neighbours,
# only k+1 distinct probability values are possible (0/k, 1/k, ..., k/k).
# With k=5, this means only 6 unique probability values exist on the
# validation set, making threshold optimisation unreliable. The default
# threshold of 0.50 was retained.
#
# MODEL SELECTION OF k
# k was selected via 5-fold repeated cross-validation (3 repeats) using
# ROC-AUC as the tuning metric, consistent with the imbalanced class setting.
# The optimal k=5 achieved a CV ROC of 0.926. However, this inflated
# CV performance reflects the balanced SMOTE training set, not the true
# class distribution in the test set.
#
# TEST PERFORMANCE
# The large gap between CV ROC (0.926) and test AUC (0.545) confirms that
# the model does not generalise well. The test set reflects the true class
# distribution (majority = Yes), while CV was evaluated on SMOTE-balanced
# folds. This gap is a direct consequence of SMOTE inflating CV performance.
# The ROC curve on the test set is nearly diagonal, indicating the model
# performs only marginally better than random guessing.
#
# WHY k-NN IS UNSUITABLE FOR THIS DATASET
# 1. Ordinal/Likert predictors: All 8 features are Likert scales (1-10).
#    Euclidean distance treats these as continuous, but the intervals are
#    not guaranteed to be equal. This undermines the distance metric that
#    k-NN relies on.
# 2. Curse of dimensionality: With 8 features, distance-based methods
#    begin to lose discriminative power as observations become equidistant.
# 3. No parametric structure: k-NN cannot exploit global patterns in the
#    data. It relies only on local neighbourhoods, which are less reliable
#    in high-dimensional ordinal spaces.
#
# Key results:
#   Feature selection:  RF importance (8 Likert-scale predictors)
#   Best k:             5 (5-fold repeated CV, ROC metric)
#   Threshold:          0.50 (default — coarse probability output)
#   CV ROC:             0.926 (on SMOTE-balanced training folds)
#   Test AUC:           0.545
#   Balanced Accuracy:  0.537
#   Sensitivity:        0.594
#   Specificity:        0.481
#==============================================================================


#===========================================================
#========================= DECISION TREE ===================
#===========================================================

# Prepare data — all features, SMOTE + scaling via prepare_data
all_features <- setdiff(names(mental_train), "Has_Mental_Health_Issue")
tree_data <- prepare_data(mental_train, mental_val, mental_test, all_features, scale = FALSE)

train_sc_tree <- tree_data$train
val_sc_tree   <- tree_data$val
test_sc_tree  <- tree_data$test

y_val_tree  <- val_sc_tree$Has_Mental_Health_Issue
y_test_tree <- test_sc_tree$Has_Mental_Health_Issue

# Step 1: Hyperparameter grid search (mindev + minsize) via CV
mindev_grid  <- c(0.001, 0.005, 0.01, 0.02)
minsize_grid <- c(5, 10, 20, 30)

hp_grid <- expand.grid(mindev = mindev_grid, minsize = minsize_grid)

cv_hp_results <- map_dfr(seq_len(nrow(hp_grid)), function(i) {
  
  mindev_i  <- hp_grid$mindev[i]
  minsize_i <- hp_grid$minsize[i]
  
  set.seed(42)
  tree_i <- tree(
    Has_Mental_Health_Issue ~ .,
    data    = train_sc_tree,
    control = tree.control(
      nobs    = nrow(train_sc_tree),
      mindev  = mindev_i,
      minsize = minsize_i
    )
  )
  
  # CV to find best pruned size for this combination
  cv_i <- cv.tree(tree_i, FUN = prune.misclass, K = 5)
  
  best_size_i <- min(cv_i$size[cv_i$dev == min(cv_i$dev)])
  
  # Prune and evaluate on validation set
  pruned_i <- prune.misclass(tree_i, best = best_size_i)
  
  prob_val_i <- predict(pruned_i, newdata = val_sc_tree, type = "vector")[, "Yes"]
  
  roc_i  <- roc(y_val_tree, prob_val_i, levels = c("No", "Yes"),
                direction = "<", quiet = TRUE)
  auc_i  <- as.numeric(auc(roc_i))
  
  thr_row <- pick_best_thr_bal(y_val_tree, prob_val_i)$best
  bal_i   <- thr_row$BalancedAcc
  
  tibble(
    mindev    = mindev_i,
    minsize   = minsize_i,
    best_size = best_size_i,
    Val_AUC   = round(auc_i, 4),
    Val_BalancedAcc = round(bal_i, 4)
  )
})

print(cv_hp_results)

# Best hyperparameter combination (by Val AUC)
best_hp <- cv_hp_results %>% arrange(desc(Val_AUC)) %>% slice(1)
print(best_hp)

best_mindev  <- best_hp$mindev
best_minsize <- best_hp$minsize

# Since minsize has no effect, simplify plot to show only mindev
ggplot(cv_hp_results %>% distinct(mindev, .keep_all = TRUE),
       aes(x = factor(mindev), y = Val_AUC, group = 1)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2.5) +
  labs(
    title    = "Decision Tree: Hyperparameter Grid Search",
    subtitle = "effect of mindev on validation AUC",
    x = "mindev",
    y = "Validation AUC"
  ) +
  theme_minimal(base_size = 13)


# Step 2: Grow full tree with best hyperparameters
set.seed(42)
mental_tree_full <- tree(
  Has_Mental_Health_Issue ~ .,
  data    = train_sc_tree,
  control = tree.control(
    nobs    = nrow(train_sc_tree),
    mindev  = best_mindev,
    minsize = best_minsize
  )
)

cat("Full tree terminal nodes:", sum(mental_tree_full$frame$var == "<leaf>"), "\n")

# Step 3: CV pruning to select best size
set.seed(42)
mental_cv <- cv.tree(mental_tree_full, FUN = prune.misclass, K = 5)

cv_results_tree <- data.frame(
  size     = mental_cv$size,
  cv_error = mental_cv$dev
)
print(cv_results_tree)

best_size <- min(mental_cv$size[mental_cv$dev == min(mental_cv$dev)])

best_size_table <- data.frame(
  Measure = c("Minimum CV error", "Best tree size"),
  Value   = c(min(mental_cv$dev), best_size)
)
print(best_size_table)

# CV error vs tree size plot
tibble(size = mental_cv$size, cv_error = mental_cv$dev) %>%
  ggplot(aes(x = size, y = cv_error)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_vline(xintercept = best_size, linetype = "dashed", color = "red") +
  annotate(
    "text",
    x     = best_size + 0.3,
    y     = max(mental_cv$dev) * 0.98,
    label = paste("best =", best_size),
    color = "red", hjust = 0
  ) +
  labs(
    title    = "Decision Tree: CV Error vs Tree Size",
    subtitle = paste("Best size =", best_size),
    x = "Number of terminal nodes",
    y = "CV misclassification count"
  ) +
  theme_minimal(base_size = 13)

# Step 4: Pruned tree
mental_tree_pruned <- prune.misclass(mental_tree_full, best = best_size)

plot(mental_tree_pruned)
text(mental_tree_pruned, pretty = 0, cex = 0.8)
title(paste("Mental Health: Pruned Tree (", best_size, "leaves)"))

# Step 5: Threshold selection on validation set
p_tree_val <- predict(mental_tree_pruned, newdata = val_sc_tree,
                      type = "vector")[, "Yes"]

best_thr_tree_row <- pick_best_thr_bal(y_val_tree, p_tree_val)$best
best_thr_tree     <- best_thr_tree_row$threshold

print(best_thr_tree_row)

# Threshold effect plot
all_thr_tree <- pick_best_thr_bal(y_val_tree, p_tree_val)$all
plot_threshold_effect_one(all_thr_tree, "Decision Tree", best_thr_tree)

# Step 6: Final evaluation on test set
p_tree_test <- predict(mental_tree_pruned, newdata = test_sc_tree,
                       type = "vector")[, "Yes"]

pred_tree_test <- factor(
  ifelse(p_tree_test >= best_thr_tree, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_tree  <- confusionMatrix(pred_tree_test, y_test_tree, positive = "Yes")
roc_tree <- roc(y_test_tree, p_tree_test, levels = c("No", "Yes"),
                direction = "<", quiet = TRUE)
auc_tree <- as.numeric(auc(roc_tree))

sens_tree <- as.numeric(cm_tree$byClass["Sensitivity"])
spec_tree <- as.numeric(cm_tree$byClass["Specificity"])
bal_tree  <- 0.5 * (sens_tree + spec_tree)
acc_tree  <- as.numeric(cm_tree$overall["Accuracy"])
f1_tree   <- as.numeric(cm_tree$byClass["F1"])
prec_tree <- as.numeric(cm_tree$byClass["Pos Pred Value"])

tree_test_summary <- data.frame(
  Model             = paste0("Decision Tree (", best_size, " leaves)"),
  Test_AUC          = round(auc_tree,  3),
  Balanced_Accuracy = round(bal_tree,  3),
  Threshold         = round(best_thr_tree, 2),
  Accuracy          = round(acc_tree,  3),
  Sensitivity       = round(sens_tree, 3),
  Specificity       = round(spec_tree, 3),
  Precision         = round(prec_tree, 3),
  F1                = round(f1_tree,   3)
)
print(tree_test_summary)

cm_tree_table <- as.data.frame(cm_tree$table)
colnames(cm_tree_table) <- c("Prediction", "Reference", "Count")
print(cm_tree_table)

# ROC curve
ggroc(roc_tree, linewidth = 1, color = "steelblue") +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title    = "Decision Tree ROC Curve",
    subtitle = paste("AUC =", round(auc_tree, 3)),
    x = "Specificity", y = "Sensitivity"
  ) +
  theme_minimal()

# Confusion matrix plot
cm_tree_df <- as.data.frame(cm_tree$table)
colnames(cm_tree_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_tree_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = paste0("Decision Tree Confusion Matrix (thr = ", best_thr_tree, ")"),
    x = "Actual", y = "Predicted"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Final summary
tree_summary <- data.frame(
  Model             = "Decision Tree",
  Main_Setting      = paste0(best_size, " leaves | mindev=", best_mindev,
                             " | minsize=", best_minsize),
  Threshold         = round(best_thr_tree, 2),
  Test_AUC          = round(auc_tree,  3),
  Balanced_Accuracy = round(bal_tree,  3),
  Sensitivity       = round(sens_tree, 3),
  Specificity       = round(spec_tree, 3)
)
print(tree_summary)





#===========================================================
#========================= BAGGING =========================
#===========================================================

# Same unscaled data as decision tree
train_bag_data <- train_sc_tree
val_bag_data   <- val_sc_tree
test_bag_data  <- test_sc_tree

y_val_bag  <- val_bag_data$Has_Mental_Health_Issue
y_test_bag <- test_bag_data$Has_Mental_Health_Issue

p <- ncol(train_bag_data) - 1  # number of predictors

# Step 1: Find stable ntree via OOB error plot
set.seed(42)
mental_bag_ntree <- randomForest(
  Has_Mental_Health_Issue ~ .,
  data      = train_bag_data,
  mtry      = p,  # bagging = all predictors at each split
  ntree     = 500,
  importance = TRUE
)

oob_df_bag <- tibble(
  Trees     = seq_len(nrow(mental_bag_ntree$err.rate)),
  OOB_Error = mental_bag_ntree$err.rate[, "OOB"]
)

ggplot(oob_df_bag, aes(x = Trees, y = OOB_Error)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(
    title    = "Bagging: OOB Error vs Number of Trees",
    subtitle = "Used to determine stable ntree",
    x = "Number of Trees", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Identify stable ntree — first tree where OOB stabilises
oob_min <- min(oob_df_bag$OOB_Error)
ntree_stable <- which(oob_df_bag$OOB_Error <= oob_min * 1.01)[1]
cat("Stable ntree:", ntree_stable, "\n")

# Step 2: nodesize grid search via OOB error
nodesize_grid <- c(1, 5, 10, 20, 30)

nodesize_results <- map_dfr(nodesize_grid, function(ns) {
  set.seed(42)
  fit <- randomForest(
    Has_Mental_Health_Issue ~ .,
    data      = train_bag_data,
    mtry      = p,
    ntree     = ntree_stable,
    nodesize  = ns
  )
  tibble(
    nodesize  = ns,
    OOB_Error = fit$err.rate[ntree_stable, "OOB"]
  )
})

print(nodesize_results)

best_nodesize_bag <- nodesize_results$nodesize[which.min(nodesize_results$OOB_Error)]

cat("Best nodesize:", best_nodesize_bag, "\n")

ggplot(nodesize_results, aes(x = nodesize, y = OOB_Error)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_vline(xintercept = best_nodesize_bag, linetype = "dashed", color = "red") +
  annotate("text", x = best_nodesize_bag + 0.5, y = max(nodesize_results$OOB_Error),
           label = paste("best =", best_nodesize_bag), color = "red", hjust = 0) +
  labs(
    title    = "Bagging: OOB Error vs nodesize",
    subtitle = "Regularisation via minimum terminal node size",
    x = "nodesize", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Step 3: maxnodes grid search via OOB error
maxnodes_grid <- list(10, 20, 50, 100, 200, NULL)

maxnodes_results <- map_dfr(seq_along(maxnodes_grid), function(i) {
  mn <- maxnodes_grid[[i]]
  set.seed(42)
  fit <- if (is.null(mn)) {
    randomForest(
      Has_Mental_Health_Issue ~ .,
      data     = train_bag_data,
      mtry     = p,
      ntree    = ntree_stable,
      nodesize = best_nodesize_bag
    )
  } else {
    randomForest(
      Has_Mental_Health_Issue ~ .,
      data     = train_bag_data,
      mtry     = p,
      ntree    = ntree_stable,
      nodesize = best_nodesize_bag,
      maxnodes = mn
    )
  }
  tibble(
    maxnodes  = ifelse(is.null(mn), Inf, mn),
    OOB_Error = fit$err.rate[ntree_stable, "OOB"]
  )
})

print(maxnodes_results)

best_maxnodes_bag <- maxnodes_results$maxnodes[which.min(maxnodes_results$OOB_Error)]
cat("Best maxnodes:", best_maxnodes_bag, "\n")

ggplot(maxnodes_results, aes(x = maxnodes, y = OOB_Error)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_vline(xintercept = best_maxnodes_bag, linetype = "dashed", color = "red") +
  labs(
    title    = "Bagging: OOB Error vs maxnodes",
    subtitle = "Regularisation via maximum number of terminal nodes",
    x = "maxnodes", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Step 4: Final bagging model
set.seed(42)
mental_bag <- if (is.infinite(best_maxnodes_bag)) {
  randomForest(
    Has_Mental_Health_Issue ~ .,
    data      = train_bag_data,
    mtry      = p,
    ntree     = ntree_stable,
    nodesize  = best_nodesize_bag,
    importance = TRUE
  )
} else {
  randomForest(
    Has_Mental_Health_Issue ~ .,
    data      = train_bag_data,
    mtry      = p,
    ntree     = ntree_stable,
    nodesize  = best_nodesize_bag,
    maxnodes  = best_maxnodes_bag,
    importance = TRUE
  )
}

print(mental_bag)

bag_info <- data.frame(
  Measure = c("Model", "mtry (= p)", "ntree", "nodesize", "maxnodes"),
  Value   = c("Bagging", p, ntree_stable, best_nodesize_bag,
              ifelse(is.infinite(best_maxnodes_bag), "unrestricted", best_maxnodes_bag))
)
print(bag_info)

# Variable importance
varImpPlot(mental_bag, main = "Bagging: Variable Importance", cex = 0.8)

# Step 5: Threshold selection on validation set
p_bag_val <- predict(mental_bag, newdata = val_bag_data, type = "prob")[, "Yes"]

best_thr_bag_row <- pick_best_thr_bal(y_val_bag, p_bag_val)$best
best_thr_bag     <- best_thr_bag_row$threshold

print(best_thr_bag_row)

all_thr_bag <- pick_best_thr_bal(y_val_bag, p_bag_val)$all
plot_threshold_effect_one(all_thr_bag, "Bagging", best_thr_bag)

# Step 6: Final evaluation on test set
p_bag_test <- predict(mental_bag, newdata = test_bag_data, type = "prob")[, "Yes"]

pred_bag_test <- factor(
  ifelse(p_bag_test >= best_thr_bag, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_bag  <- confusionMatrix(pred_bag_test, y_test_bag, positive = "Yes")
roc_bag <- roc(y_test_bag, p_bag_test, levels = c("No", "Yes"),
               direction = "<", quiet = TRUE)
auc_bag <- as.numeric(auc(roc_bag))

sens_bag <- as.numeric(cm_bag$byClass["Sensitivity"])
spec_bag <- as.numeric(cm_bag$byClass["Specificity"])
bal_bag  <- 0.5 * (sens_bag + spec_bag)
acc_bag  <- as.numeric(cm_bag$overall["Accuracy"])
f1_bag   <- as.numeric(cm_bag$byClass["F1"])
prec_bag <- as.numeric(cm_bag$byClass["Pos Pred Value"])

bag_test_summary <- data.frame(
  Model             = paste0("Bagging (ntree=", ntree_stable, ")"),
  Test_AUC          = round(auc_bag,  3),
  Balanced_Accuracy = round(bal_bag,  3),
  Threshold         = round(best_thr_bag, 2),
  Accuracy          = round(acc_bag,  3),
  Sensitivity       = round(sens_bag, 3),
  Specificity       = round(spec_bag, 3),
  Precision         = round(prec_bag, 3),
  F1                = round(f1_bag,   3)
)
print(bag_test_summary)

cm_bag_table <- as.data.frame(cm_bag$table)
colnames(cm_bag_table) <- c("Prediction", "Reference", "Count")
print(cm_bag_table)

# ROC curve
ggroc(roc_bag, linewidth = 1, color = "steelblue") +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title    = "Bagging ROC Curve",
    subtitle = paste("AUC =", round(auc_bag, 3)),
    x = "Specificity", y = "Sensitivity"
  ) +
  theme_minimal()

# Confusion matrix
cm_bag_df <- as.data.frame(cm_bag$table)
colnames(cm_bag_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_bag_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(
    title = paste0("Bagging Confusion Matrix (thr = ", best_thr_bag, ")"),
    x = "Actual", y = "Predicted"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Final summary
bag_summary <- data.frame(
  Model             = "Bagging",
  Main_Setting      = paste0("mtry=p | ntree=", ntree_stable,
                             " | nodesize=", best_nodesize_bag,
                             " | maxnodes=", ifelse(is.infinite(best_maxnodes_bag),
                                                    "unrestricted", best_maxnodes_bag)),
  Threshold         = round(best_thr_bag, 2),
  Test_AUC          = round(auc_bag,  3),
  Balanced_Accuracy = round(bal_bag,  3),
  Sensitivity       = round(sens_bag, 3),
  Specificity       = round(spec_bag, 3)
)
print(bag_summary)




#===========================================================
#========================= RANDOM FOREST ===================
#===========================================================

# Same unscaled data as decision tree and bagging
train_rf_data <- train_sc_tree
val_rf_data   <- val_sc_tree
test_rf_data  <- test_sc_tree

y_val_rf  <- val_rf_data$Has_Mental_Health_Issue
y_test_rf <- test_rf_data$Has_Mental_Health_Issue

p <- ncol(train_rf_data) - 1  # number of predictors

# Step 1: Find stable ntree via OOB error plot
set.seed(42)
mental_rf_ntree <- randomForest(
  Has_Mental_Health_Issue ~ .,
  data      = train_rf_data,
  mtry      = floor(sqrt(p)),  # default RF mtry
  ntree     = 500,
  importance = TRUE
)

oob_df_rf <- tibble(
  Trees     = seq_len(nrow(mental_rf_ntree$err.rate)),
  OOB_Error = mental_rf_ntree$err.rate[, "OOB"]
)

ggplot(oob_df_rf, aes(x = Trees, y = OOB_Error)) +
  geom_line(color = "#41ab5d", linewidth = 1) +
  labs(
    title    = "Random Forest: OOB Error vs Number of Trees",
    subtitle = "Used to determine stable ntree",
    x = "Number of Trees", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

oob_min_rf    <- min(oob_df_rf$OOB_Error)
ntree_stable_rf <- which(oob_df_rf$OOB_Error <= oob_min_rf * 1.01)[1]
cat("Stable ntree:", ntree_stable_rf, "\n")

# Step 2: mtry grid search via OOB error
base_mtry  <- floor(sqrt(p))
mtry_grid  <- sort(unique(pmax(1, pmin(p, c(
  base_mtry - 2, base_mtry - 1, base_mtry,
  base_mtry + 1, base_mtry + 2,
  floor(p / 3), floor(p / 2)
)))))

mtry_results <- map_dfr(mtry_grid, function(m) {
  set.seed(42)
  fit <- randomForest(
    Has_Mental_Health_Issue ~ .,
    data  = train_rf_data,
    mtry  = m,
    ntree = ntree_stable_rf
  )
  tibble(mtry = m, OOB_Error = fit$err.rate[ntree_stable_rf, "OOB"])
})

print(mtry_results)
best_mtry_rf <- mtry_results$mtry[which.min(mtry_results$OOB_Error)]
cat("Best mtry:", best_mtry_rf, "\n")

ggplot(mtry_results, aes(x = mtry, y = OOB_Error)) +
  geom_line(color = "#41ab5d", linewidth = 1) +
  geom_point(color = "#41ab5d", size = 2.5) +
  geom_vline(xintercept = best_mtry_rf, linetype = "dashed", color = "red") +
  annotate("text", x = best_mtry_rf + 0.3, y = max(mtry_results$OOB_Error),
           label = paste("best =", best_mtry_rf), color = "red", hjust = 0) +
  labs(
    title    = "Random Forest: OOB Error vs mtry",
    subtitle = "Number of predictors sampled at each split",
    x = "mtry", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Step 3: nodesize grid search via OOB error
nodesize_grid_rf <- c(1, 5, 10, 20, 30)

nodesize_results_rf <- map_dfr(nodesize_grid_rf, function(ns) {
  set.seed(42)
  fit <- randomForest(
    Has_Mental_Health_Issue ~ .,
    data     = train_rf_data,
    mtry     = best_mtry_rf,
    ntree    = ntree_stable_rf,
    nodesize = ns
  )
  tibble(nodesize = ns, OOB_Error = fit$err.rate[ntree_stable_rf, "OOB"])
})

print(nodesize_results_rf)
best_nodesize_rf <- nodesize_results_rf$nodesize[which.min(nodesize_results_rf$OOB_Error)]
cat("Best nodesize:", best_nodesize_rf, "\n")

ggplot(nodesize_results_rf, aes(x = nodesize, y = OOB_Error)) +
  geom_line(color = "#41ab5d", linewidth = 1) +
  geom_point(color = "#41ab5d", size = 2.5) +
  geom_vline(xintercept = best_nodesize_rf, linetype = "dashed", color = "red") +
  annotate("text", x = best_nodesize_rf + 0.5, y = max(nodesize_results_rf$OOB_Error),
           label = paste("best =", best_nodesize_rf), color = "red", hjust = 0) +
  labs(
    title    = "Random Forest: OOB Error vs nodesize",
    subtitle = "Regularisation via minimum terminal node size",
    x = "nodesize", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Step 4: maxnodes grid search via OOB error
maxnodes_grid_rf <- list(10, 20, 50, 100, 200, NULL)

maxnodes_results_rf <- map_dfr(seq_along(maxnodes_grid_rf), function(i) {
  mn <- maxnodes_grid_rf[[i]]
  set.seed(42)
  fit <- if (is.null(mn)) {
    randomForest(
      Has_Mental_Health_Issue ~ .,
      data     = train_rf_data,
      mtry     = best_mtry_rf,
      ntree    = ntree_stable_rf,
      nodesize = best_nodesize_rf
    )
  } else {
    randomForest(
      Has_Mental_Health_Issue ~ .,
      data     = train_rf_data,
      mtry     = best_mtry_rf,
      ntree    = ntree_stable_rf,
      nodesize = best_nodesize_rf,
      maxnodes = mn
    )
  }
  tibble(
    maxnodes  = ifelse(is.null(mn), Inf, mn),
    OOB_Error = fit$err.rate[ntree_stable_rf, "OOB"]
  )
})

print(maxnodes_results_rf)
best_maxnodes_rf <- maxnodes_results_rf$maxnodes[which.min(maxnodes_results_rf$OOB_Error)]
cat("Best maxnodes:", best_maxnodes_rf, "\n")

ggplot(maxnodes_results_rf, aes(x = maxnodes, y = OOB_Error)) +
  geom_line(color = "#41ab5d", linewidth = 1) +
  geom_point(color = "#41ab5d", size = 2.5) +
  geom_vline(xintercept = best_maxnodes_rf, linetype = "dashed", color = "red") +
  labs(
    title    = "Random Forest: OOB Error vs maxnodes",
    subtitle = "Regularisation via maximum number of terminal nodes",
    x = "maxnodes", y = "OOB Error"
  ) +
  theme_minimal(base_size = 13)

# Step 5: Final RF model
set.seed(42)
mental_rf <- if (is.infinite(best_maxnodes_rf)) {
  randomForest(
    Has_Mental_Health_Issue ~ .,
    data      = train_rf_data,
    mtry      = best_mtry_rf,
    ntree     = ntree_stable_rf,
    nodesize  = best_nodesize_rf,
    importance = TRUE
  )
} else {
  randomForest(
    Has_Mental_Health_Issue ~ .,
    data      = train_rf_data,
    mtry      = best_mtry_rf,
    ntree     = ntree_stable_rf,
    nodesize  = best_nodesize_rf,
    maxnodes  = best_maxnodes_rf,
    importance = TRUE
  )
}

print(mental_rf)

rf_info <- data.frame(
  Measure = c("Model", "mtry", "ntree", "nodesize", "maxnodes"),
  Value   = c("Random Forest", best_mtry_rf, ntree_stable_rf, best_nodesize_rf,
              ifelse(is.infinite(best_maxnodes_rf), "unrestricted", best_maxnodes_rf))
)
print(rf_info)

# Variable importance
varImpPlot(mental_rf, main = "Random Forest: Variable Importance", cex = 0.8)

imp_df_rf <- as.data.frame(importance(mental_rf))
imp_df_rf$Variable <- rownames(imp_df_rf)

imp_long_rf <- imp_df_rf %>%
  select(Variable, MeanDecreaseAccuracy, MeanDecreaseGini) %>%
  pivot_longer(cols = -Variable, names_to = "Measure", values_to = "Importance")

ggplot(imp_long_rf,
       aes(x = fct_reorder(Variable, Importance), y = Importance, fill = Measure)) +
  geom_col(position = "dodge", alpha = 0.85) +
  coord_flip() +
  facet_wrap(~ Measure, scales = "free_x") +
  labs(
    title = "Random Forest: Variable Importance",
    x = NULL, y = "Importance"
  ) +
  scale_fill_viridis_d() +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

# Step 6: Threshold selection on validation set
p_rf_val <- predict(mental_rf, newdata = val_rf_data, type = "prob")[, "Yes"]

best_thr_rf_row <- pick_best_thr_bal(y_val_rf, p_rf_val)$best
best_thr_rf     <- best_thr_rf_row$threshold

print(best_thr_rf_row)

all_thr_rf <- pick_best_thr_bal(y_val_rf, p_rf_val)$all
plot_threshold_effect_one(all_thr_rf, "Random Forest", best_thr_rf)

# Step 7: Final evaluation on test set
p_rf_test <- predict(mental_rf, newdata = test_rf_data, type = "prob")[, "Yes"]

pred_rf_test <- factor(
  ifelse(p_rf_test >= best_thr_rf, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_rf  <- confusionMatrix(pred_rf_test, y_test_rf, positive = "Yes")
roc_rf <- roc(y_test_rf, p_rf_test, levels = c("No", "Yes"),
              direction = "<", quiet = TRUE)
auc_rf <- as.numeric(auc(roc_rf))

sens_rf <- as.numeric(cm_rf$byClass["Sensitivity"])
spec_rf <- as.numeric(cm_rf$byClass["Specificity"])
bal_rf  <- 0.5 * (sens_rf + spec_rf)
acc_rf  <- as.numeric(cm_rf$overall["Accuracy"])
f1_rf   <- as.numeric(cm_rf$byClass["F1"])
prec_rf <- as.numeric(cm_rf$byClass["Pos Pred Value"])

rf_test_summary <- data.frame(
  Model             = paste0("Random Forest (mtry=", best_mtry_rf, ")"),
  Test_AUC          = round(auc_rf,  3),
  Balanced_Accuracy = round(bal_rf,  3),
  Threshold         = round(best_thr_rf, 2),
  Accuracy          = round(acc_rf,  3),
  Sensitivity       = round(sens_rf, 3),
  Specificity       = round(spec_rf, 3),
  Precision         = round(prec_rf, 3),
  F1                = round(f1_rf,   3)
)
print(rf_test_summary)

cm_rf_table <- as.data.frame(cm_rf$table)
colnames(cm_rf_table) <- c("Prediction", "Reference", "Count")
print(cm_rf_table)

# ROC curve
ggroc(roc_rf, linewidth = 1, color = "#41ab5d") +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title    = "Random Forest ROC Curve",
    subtitle = paste("AUC =", round(auc_rf, 3)),
    x = "Specificity", y = "Sensitivity"
  ) +
  theme_minimal()

# Confusion matrix
cm_rf_df <- as.data.frame(cm_rf$table)
colnames(cm_rf_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_rf_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#41ab5d") +
  labs(
    title = paste0("Random Forest Confusion Matrix (thr = ", best_thr_rf, ")"),
    x = "Actual", y = "Predicted"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Final summary
rf_summary <- data.frame(
  Model             = "Random Forest",
  Main_Setting      = paste0("mtry=", best_mtry_rf,
                             " | ntree=", ntree_stable_rf,
                             " | nodesize=", best_nodesize_rf,
                             " | maxnodes=", ifelse(is.infinite(best_maxnodes_rf),
                                                    "unrestricted", best_maxnodes_rf)),
  Threshold         = round(best_thr_rf, 2),
  Test_AUC          = round(auc_rf,  3),
  Balanced_Accuracy = round(bal_rf,  3),
  Sensitivity       = round(sens_rf, 3),
  Specificity       = round(spec_rf, 3)
)
print(rf_summary)





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






#===========================================================
#========================= SVM =============================
#===========================================================

# SVM requires scaling — use prepare_data with scale = TRUE
all_features <- setdiff(names(mental_train), "Has_Mental_Health_Issue")
svm_data <- prepare_data(mental_train, mental_val, mental_test, all_features, scale = TRUE)

train_sc_svm <- svm_data$train
val_sc_svm   <- svm_data$val
test_sc_svm  <- svm_data$test

y_val_svm  <- val_sc_svm$Has_Mental_Health_Issue
y_test_svm <- test_sc_svm$Has_Mental_Health_Issue

# CV control — 5-fold, ROC metric
ctrl_svm <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

#--- 1. Linear Kernel (baseline) ---
# Only C to tune
grid_linear <- expand.grid(C = c(0.01, 0.1, 1, 10))

set.seed(42)
svm_linear <- train(
  Has_Mental_Health_Issue ~ .,
  data      = train_sc_svm,
  method    = "svmLinear",
  trControl = ctrl_svm,
  tuneGrid  = grid_linear,
  metric    = "ROC"
)

linear_results <- svm_linear$results[, c("C", "ROC", "Sens", "Spec")]
print(linear_results)

best_linear <- data.frame(
  Measure = c("Best C", "Best CV ROC"),
  Value   = c(svm_linear$bestTune$C, round(max(svm_linear$results$ROC), 4))
)
print(best_linear)

ggplot(svm_linear$results, aes(x = C, y = ROC)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2.5) +
  geom_vline(xintercept = svm_linear$bestTune$C,
             linetype = "dashed", color = "red") +
  scale_x_log10() +
  labs(
    title    = "SVM Linear Kernel: CV ROC vs C",
    subtitle = "5-fold cross-validation",
    x = "C (log scale)", y = "CV ROC-AUC"
  ) +
  theme_minimal(base_size = 13)

#--- 2. Radial Kernel ---
# C and sigma (gamma) to tune
grid_radial <- expand.grid(
  C     = c(0.1, 1, 10),
  sigma = c(0.01, 0.1, 1)
)

set.seed(42)
svm_radial <- train(
  Has_Mental_Health_Issue ~ .,
  data      = train_sc_svm,
  method    = "svmRadial",
  trControl = ctrl_svm,
  tuneGrid  = grid_radial,
  metric    = "ROC"
)

radial_results <- svm_radial$results[, c("C", "sigma", "ROC", "Sens", "Spec")]
print(radial_results)

best_radial <- data.frame(
  Measure = c("Best C", "Best sigma", "Best CV ROC"),
  Value   = c(svm_radial$bestTune$C,
              svm_radial$bestTune$sigma,
              round(max(svm_radial$results$ROC), 4))
)
print(best_radial)

ggplot(svm_radial$results,
       aes(x = C, y = ROC, color = factor(sigma), group = factor(sigma))) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_log10() +
  labs(
    title    = "SVM Radial Kernel: CV ROC vs C",
    subtitle = "5-fold cross-validation",
    x = "C (log scale)", y = "CV ROC-AUC",
    color = "sigma"
  ) +
  theme_minimal(base_size = 13)

#--- 3. Polynomial Kernel ---
# C and degree to tune
grid_poly <- expand.grid(
  C      = c(0.1, 1, 10),
  degree = c(2, 3, 4),
  scale  = 1  # fixed
)

set.seed(42)
svm_poly <- train(
  Has_Mental_Health_Issue ~ .,
  data      = train_sc_svm,
  method    = "svmPoly",
  trControl = ctrl_svm,
  tuneGrid  = grid_poly,
  metric    = "ROC"
)

poly_results <- svm_poly$results[, c("C", "degree", "ROC", "Sens", "Spec")]
print(poly_results)

best_poly <- data.frame(
  Measure = c("Best C", "Best degree", "Best CV ROC"),
  Value   = c(svm_poly$bestTune$C,
              svm_poly$bestTune$degree,
              round(max(svm_poly$results$ROC), 4))
)
print(best_poly)

ggplot(svm_poly$results,
       aes(x = C, y = ROC, color = factor(degree), group = factor(degree))) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_log10() +
  labs(
    title    = "SVM Polynomial Kernel: CV ROC vs C",
    subtitle = "5-fold cross-validation",
    x = "C (log scale)", y = "CV ROC-AUC",
    color = "degree"
  ) +
  theme_minimal(base_size = 13)

#--- 4. Kernel comparison ---
kernel_comparison <- data.frame(
  Kernel   = c("Linear", "Radial", "Polynomial"),
  Best_C   = c(svm_linear$bestTune$C,
               svm_radial$bestTune$C,
               svm_poly$bestTune$C),
  Best_ROC = c(round(max(svm_linear$results$ROC), 4),
               round(max(svm_radial$results$ROC), 4),
               round(max(svm_poly$results$ROC), 4))
)
print(kernel_comparison)

# Select best kernel
best_kernel_row <- kernel_comparison %>% arrange(desc(Best_ROC)) %>% slice(1)
best_kernel_name <- best_kernel_row$Kernel
cat("Best kernel:", best_kernel_name, "\n")

best_svm <- switch(best_kernel_name,
                   "Linear"     = svm_linear,
                   "Radial"     = svm_radial,
                   "Polynomial" = svm_poly
)

# Step 5: Threshold selection on validation set
p_svm_val <- predict(best_svm, newdata = val_sc_svm, type = "prob")[, "Yes"]

best_thr_svm_row <- pick_best_thr_bal(y_val_svm, p_svm_val)$best
best_thr_svm     <- best_thr_svm_row$threshold

print(best_thr_svm_row)

all_thr_svm <- pick_best_thr_bal(y_val_svm, p_svm_val)$all
plot_threshold_effect_one(all_thr_svm, paste("SVM -", best_kernel_name, "Kernel"), best_thr_svm)

# Step 6: Final evaluation on test set
p_svm_test <- predict(best_svm, newdata = test_sc_svm, type = "prob")[, "Yes"]

pred_svm_test <- factor(
  ifelse(p_svm_test >= best_thr_svm, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_svm  <- confusionMatrix(pred_svm_test, y_test_svm, positive = "Yes")
roc_svm <- roc(y_test_svm, p_svm_test, levels = c("No", "Yes"),
               direction = "<", quiet = TRUE)
auc_svm <- as.numeric(auc(roc_svm))

sens_svm <- as.numeric(cm_svm$byClass["Sensitivity"])
spec_svm <- as.numeric(cm_svm$byClass["Specificity"])
bal_svm  <- 0.5 * (sens_svm + spec_svm)
acc_svm  <- as.numeric(cm_svm$overall["Accuracy"])
f1_svm   <- as.numeric(cm_svm$byClass["F1"])
prec_svm <- as.numeric(cm_svm$byClass["Pos Pred Value"])

svm_test_summary <- data.frame(
  Model             = paste0("SVM (", best_kernel_name, " kernel)"),
  Test_AUC          = round(auc_svm,  3),
  Balanced_Accuracy = round(bal_svm,  3),
  Threshold         = round(best_thr_svm, 2),
  Accuracy          = round(acc_svm,  3),
  Sensitivity       = round(sens_svm, 3),
  Specificity       = round(spec_svm, 3),
  Precision         = round(prec_svm, 3),
  F1                = round(f1_svm,   3)
)
print(svm_test_summary)

cm_svm_table <- as.data.frame(cm_svm$table)
colnames(cm_svm_table) <- c("Prediction", "Reference", "Count")
print(cm_svm_table)

# ROC curve
ggroc(roc_svm, linewidth = 1, color = "purple") +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title    = paste("SVM ROC Curve -", best_kernel_name, "Kernel"),
    subtitle = paste("AUC =", round(auc_svm, 3)),
    x = "Specificity", y = "Sensitivity"
  ) +
  theme_minimal()

# Confusion matrix
cm_svm_df <- as.data.frame(cm_svm$table)
colnames(cm_svm_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_svm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(
    title = paste0("SVM Confusion Matrix - ", best_kernel_name,
                   " Kernel (thr = ", best_thr_svm, ")"),
    x = "Actual", y = "Predicted"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Final summary
svm_summary <- data.frame(
  Model             = paste0("SVM (", best_kernel_name, ")"),
  Main_Setting      = paste0("kernel=", best_kernel_name,
                             " | C=", best_kernel_row$Best_C),
  Threshold         = round(best_thr_svm, 2),
  Test_AUC          = round(auc_svm,  3),
  Balanced_Accuracy = round(bal_svm,  3),
  Sensitivity       = round(sens_svm, 3),
  Specificity       = round(spec_svm, 3)
)
print(svm_summary)

# Complete SVM comparison table
svm_full_comparison <- data.frame(
  Kernel   = c("Linear", "Radial", "Polynomial"),
  CV_AUC   = c(round(max(svm_linear$results$ROC), 3),
               round(max(svm_radial$results$ROC), 3),
               round(max(svm_poly$results$ROC),   3)),
  Test_AUC = c(round(as.numeric(auc(roc_linear)), 3),
               round(as.numeric(auc(roc_radial)), 3),
               round(auc_svm, 3)),
  Best_C   = c(svm_linear$bestTune$C,
               svm_radial$bestTune$C,
               svm_poly$bestTune$C),
  Best_Param = c(paste0("C=", svm_linear$bestTune$C),
                 paste0("C=", svm_radial$bestTune$C,
                        " | sigma=", svm_radial$bestTune$sigma),
                 paste0("C=", svm_poly$bestTune$C,
                        " | degree=", svm_poly$bestTune$degree))
)

print(svm_full_comparison)

# ROC curves — all three kernels together
roc_linear_obj <- roc_linear
roc_radial_obj <- roc_radial
roc_poly_obj   <- roc_svm

ggroc(list(Linear = roc_linear_obj,
           Radial = roc_radial_obj,
           Polynomial = roc_poly_obj),
      linewidth = 1) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  scale_color_manual(
    values = c("Linear" = "steelblue", "Radial" = "#41ab5d", "Polynomial" = "purple"),
    labels = c(
      paste0("Linear (AUC=",     round(as.numeric(auc(roc_linear_obj)), 3), ")"),
      paste0("Radial (AUC=",     round(as.numeric(auc(roc_radial_obj)), 3), ")"),
      paste0("Polynomial (AUC=", round(as.numeric(auc(roc_poly_obj)),   3), ")")
    )
  ) +
  labs(
    title    = "SVM: ROC Curves by Kernel",
    subtitle = "Test set evaluation",
    x = "Specificity", y = "Sensitivity",
    color = "Kernel"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

# Save everything
saveRDS(svm_full_comparison, "svm_full_comparison.rds")
saveRDS(roc_linear_obj,      "svm_roc_linear.rds")
saveRDS(roc_radial_obj,      "svm_roc_radial.rds")
saveRDS(roc_poly_obj,        "svm_roc_poly.rds")

cat("CV vs Test AUC gap:\n")
cat("Linear:     CV=", round(max(svm_linear$results$ROC), 3),
    "| Test=", round(as.numeric(auc(roc_linear)), 3), "\n")
cat("Radial:     CV=", round(max(svm_radial$results$ROC), 3),
    "| Test=", round(as.numeric(auc(roc_radial)), 3), "\n")
cat("Polynomial: CV=", round(max(svm_poly$results$ROC),   3),
    "| Test=", round(auc_svm, 3), "\n")





#===========================================================
#=================== NEURAL NETWORK =======================
#===========================================================
library(purrr)
# Neural network backend activations
install_tensorflow(version = "2.16")
keras3::keras$backend$backend()


# NN requires scaling — use prepare_data with scale = TRUE
all_features <- setdiff(names(mental_train), "Has_Mental_Health_Issue")
nn_data <- prepare_data(mental_train, mental_val, mental_test, all_features, scale = TRUE)

train_sc_nn <- nn_data$train
val_sc_nn   <- nn_data$val
test_sc_nn  <- nn_data$test

y_val_nn  <- val_sc_nn$Has_Mental_Health_Issue
y_test_nn <- test_sc_nn$Has_Mental_Health_Issue

# Prepare matrices
prep_nn_matrix <- function(df) {
  df_dummy <- model.matrix(Has_Mental_Health_Issue ~ ., data = df)[, -1]
  label    <- as.numeric(df$Has_Mental_Health_Issue == "Yes")
  list(x = df_dummy, y = label)
}

train_nn <- prep_nn_matrix(train_sc_nn)
val_nn   <- prep_nn_matrix(val_sc_nn)
test_nn  <- prep_nn_matrix(test_sc_nn)

n_features <- ncol(train_nn$x)
cat("Features:", n_features, "\n")
cat("Training rows:", nrow(train_nn$x), "\n")

# Helper: build model dynamically based on n_layers and units
build_nn <- function(n_layers, units, dropout, lr, n_features) {
  
  model <- keras_model_sequential()
  
  # First hidden layer — must specify input_shape
  model <- model %>%
    layer_dense(units = units, activation = "relu",
                input_shape = n_features) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout)
  
  # Additional hidden layers — units halved each time
  if (n_layers > 1) {
    for (l in 2:n_layers) {
      units_l <- max(8, units / (2^(l-1)))  # halve each layer, min 8
      model <- model %>%
        layer_dense(units = units_l, activation = "relu") %>%
        layer_batch_normalization() %>%
        layer_dropout(rate = dropout)
    }
  }
  
  # Output layer
  model <- model %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss      = "binary_crossentropy",
    metrics   = list(metric_auc(name = "auc"))
  )
  
  model
}

# Step 1: Grid search
n_layers_grid <- c(1, 2, 3)
units_grid    <- c(32, 64, 128)
dropout_grid  <- c(0.2, 0.4)
lr_grid       <- c(0.0001, 0.001, 0.01)

param_grid_nn <- expand.grid(
  n_layers   = n_layers_grid,
  units      = units_grid,
  dropout    = dropout_grid,
  lr         = lr_grid
)

cat("Total combinations:", nrow(param_grid_nn), "\n")

nn_grid_results <- map_dfr(seq_len(nrow(param_grid_nn)), function(i) {
  
  n_layers_i <- param_grid_nn$n_layers[i]
  units_i    <- param_grid_nn$units[i]
  dropout_i  <- param_grid_nn$dropout[i]
  lr_i       <- param_grid_nn$lr[i]
  
  set.seed(42)
  keras3::set_random_seed(42)
  
  model_i <- build_nn(n_layers_i, units_i, dropout_i, lr_i, n_features)
  
  history_i <- model_i %>% fit(
    x               = train_nn$x,
    y               = train_nn$y,
    epochs          = 100,
    batch_size      = 64,
    validation_data = list(val_nn$x, val_nn$y),
    callbacks       = list(
      callback_early_stopping(
        monitor              = "val_auc",
        patience             = 10,
        mode                 = "max",
        restore_best_weights = TRUE
      )
    ),
    verbose = 0
  )
  
  best_val_auc <- max(history_i$metrics$val_auc)
  best_epoch   <- which.max(history_i$metrics$val_auc)
  
  cat(sprintf("Combo %d/%d: layers=%d units=%d dropout=%.1f lr=%.4f | val_AUC=%.4f (epoch %d)\n",
              i, nrow(param_grid_nn), n_layers_i, units_i,
              dropout_i, lr_i, best_val_auc, best_epoch))
  
  tibble(
    n_layers   = n_layers_i,
    units      = units_i,
    dropout    = dropout_i,
    lr         = lr_i,
    best_epoch = best_epoch,
    val_AUC    = round(best_val_auc, 4)
  )
})
# buradayızz 22.53'te çalıştırdım
# Grid search results
print(nn_grid_results %>% arrange(desc(val_AUC)))

# Filter out combinations that stopped too early (likely unstable)
nn_grid_results_filtered <- nn_grid_results %>%
  filter(best_epoch >= 5) %>%
  arrange(desc(val_AUC))

print(nn_grid_results_filtered)

# Use filtered best params
best_nn_params <- nn_grid_results_filtered %>% slice(1)
print(best_nn_params)

# Save grid results immediately
saveRDS(nn_grid_results, "nn_grid_results.rds")
saveRDS(best_nn_params,  "nn_best_params.rds")
cat("Grid results saved!\n")

# Grid search plot
ggplot(nn_grid_results,
       aes(x = factor(units), y = val_AUC,
           color = factor(dropout), group = factor(dropout))) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_grid(n_layers ~ lr, labeller = label_both) +
  scale_color_manual(values = c("0.2" = "steelblue", "0.4" = "tomato")) +
  labs(
    title    = "Neural Network: Grid Search Results",
    subtitle = "Validation AUC across hyperparameter combinations",
    x = "Units", y = "Validation AUC",
    color = "Dropout"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

# Step 2: Final model with best parameters
set.seed(42)
keras3::set_random_seed(42)

nn_final <- build_nn(
  n_layers   = best_nn_params$n_layers,
  units      = best_nn_params$units,
  dropout    = best_nn_params$dropout,
  lr         = best_nn_params$lr,
  n_features = n_features
)

summary(nn_final)

history_final <- nn_final %>% fit(
  x               = train_nn$x,
  y               = train_nn$y,
  epochs          = 200,
  batch_size      = 64,
  validation_data = list(val_nn$x, val_nn$y),
  callbacks       = list(
    callback_early_stopping(
      monitor              = "val_auc",
      patience             = 15,
      mode                 = "max",
      restore_best_weights = TRUE
    )
  ),
  verbose = 1
)

# Step 3: Training curves
history_df <- data.frame(
  epoch      = seq_along(history_final$metrics$auc),
  train_auc  = history_final$metrics$auc,
  val_auc    = history_final$metrics$val_auc,
  train_loss = history_final$metrics$loss,
  val_loss   = history_final$metrics$val_loss
)

# AUC curve
ggplot(history_df %>% pivot_longer(cols = c(train_auc, val_auc),
                                   names_to = "set", values_to = "AUC"),
       aes(x = epoch, y = AUC, color = set)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("train_auc" = "steelblue", "val_auc" = "tomato"),
                     labels = c("Train", "Validation")) +
  labs(title    = "Neural Network: Training Curve (AUC)",
       subtitle = paste("Best val AUC =", round(max(history_df$val_auc), 4)),
       x = "Epoch", y = "AUC", color = "") +
  theme_minimal(base_size = 13)

# Loss curve
ggplot(history_df %>% pivot_longer(cols = c(train_loss, val_loss),
                                   names_to = "set", values_to = "Loss"),
       aes(x = epoch, y = Loss, color = set)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("train_loss" = "steelblue", "val_loss" = "tomato"),
                     labels = c("Train", "Validation")) +
  labs(title    = "Neural Network: Training Curve (Loss)",
       subtitle = "Early stopping applied",
       x = "Epoch", y = "Loss", color = "") +
  theme_minimal(base_size = 13)

# Step 4: Threshold selection on validation set
p_nn_val <- as.numeric(predict(nn_final, val_nn$x))

best_thr_nn_row <- pick_best_thr_bal(y_val_nn, p_nn_val)$best
best_thr_nn     <- best_thr_nn_row$threshold

print(best_thr_nn_row)

all_thr_nn <- pick_best_thr_bal(y_val_nn, p_nn_val)$all
plot_threshold_effect_one(all_thr_nn, "Neural Network", best_thr_nn)

# Step 5: Final evaluation on test set
p_nn_test <- as.numeric(predict(nn_final, test_nn$x))

pred_nn_test <- factor(
  ifelse(p_nn_test >= best_thr_nn, "Yes", "No"),
  levels = c("No", "Yes")
)

cm_nn  <- confusionMatrix(pred_nn_test, y_test_nn, positive = "Yes")
roc_nn <- roc(y_test_nn, p_nn_test, levels = c("No", "Yes"),
              direction = "<", quiet = TRUE)
auc_nn <- as.numeric(auc(roc_nn))

sens_nn <- as.numeric(cm_nn$byClass["Sensitivity"])
spec_nn <- as.numeric(cm_nn$byClass["Specificity"])
bal_nn  <- 0.5 * (sens_nn + spec_nn)
acc_nn  <- as.numeric(cm_nn$overall["Accuracy"])
f1_nn   <- as.numeric(cm_nn$byClass["F1"])
prec_nn <- as.numeric(cm_nn$byClass["Pos Pred Value"])

nn_test_summary <- data.frame(
  Model             = paste0("Neural Network (", best_nn_params$n_layers,
                             "L-", best_nn_params$units, "u)"),
  Test_AUC          = round(auc_nn,  3),
  Balanced_Accuracy = round(bal_nn,  3),
  Threshold         = round(best_thr_nn, 2),
  Accuracy          = round(acc_nn,  3),
  Sensitivity       = round(sens_nn, 3),
  Specificity       = round(spec_nn, 3),
  Precision         = round(prec_nn, 3),
  F1                = round(f1_nn,   3)
)
print(nn_test_summary)

cm_nn_table <- as.data.frame(cm_nn$table)
colnames(cm_nn_table) <- c("Prediction", "Reference", "Count")
print(cm_nn_table)

# ROC curve
ggroc(roc_nn, linewidth = 1, color = "#e74c3c") +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(title    = "Neural Network ROC Curve",
       subtitle = paste("AUC =", round(auc_nn, 3)),
       x = "Specificity", y = "Sensitivity") +
  theme_minimal()

# Confusion matrix
cm_nn_df <- as.data.frame(cm_nn$table)
colnames(cm_nn_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_nn_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 7, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#e74c3c") +
  labs(title = paste0("Neural Network Confusion Matrix (thr = ", best_thr_nn, ")"),
       x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

# Final summary
nn_summary <- data.frame(
  Model             = "Neural Network",
  Main_Setting      = paste0(best_nn_params$n_layers, " layers | units=",
                             best_nn_params$units, " | dropout=",
                             best_nn_params$dropout, " | lr=", best_nn_params$lr),
  Threshold         = round(best_thr_nn, 2),
  Test_AUC          = round(auc_nn,  3),
  Balanced_Accuracy = round(bal_nn,  3),
  Sensitivity       = round(sens_nn, 3),
  Specificity       = round(spec_nn, 3)
)
print(nn_summary)

# Save everything
nn_final %>% save_model("nn_final.keras")
saveRDS(history_df,      "nn_history.rds")
saveRDS(roc_nn,          "nn_roc.rds")
saveRDS(nn_test_summary, "nn_test_summary.rds")
saveRDS(nn_summary,      "nn_summary.rds")
saveRDS(best_thr_nn,     "nn_best_thr.rds")
saveRDS(p_nn_test,       "nn_probs_test.rds")

cat("All NN objects saved!\n")









#====================================================================
# FINAL ROC PLOT (All Machine Learning Models)
#====================================================================
roc_list_all <- list(
  "k-NN"                = roc_knn,
  "Decision Tree"       = roc_tree,
  "Bagging"             = roc_bag,
  "Random Forest"       = roc_rf,
  "SVM (Linear)"        = roc_linear_obj,
  "SVM (Radial)"        = roc_radial_obj,
  "SVM (Polynomial)"    = roc_poly_obj,
  "Neural Network"      = roc_nn
)

auc_vals_all <- sapply(roc_list_all, function(r) as.numeric(auc(r)))
auc_text_all <- paste0(names(auc_vals_all), ": AUC = ",
                       sprintf("%.3f", auc_vals_all), collapse = "\n")

ggroc(roc_list_all, linewidth = 1) +
  theme_minimal(base_size = 14) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray50") +
  scale_color_viridis_d(end = 0.9) +
  labs(title    = "Machine Learning Models: ROC Curves (Test Set)",
       subtitle = "All models evaluated on the same held-out test set",
       color    = "Model") +
  theme(legend.position = "right") +
  annotate("text", x = 0.45, y = 0.05,
           label = auc_text_all, hjust = 0, vjust = 0, size = 3.5)



final_comparison <- bind_rows(
  knn_summary,
  tree_summary,
  bag_summary,
  rf_summary,
  data.frame(
    Model             = "SVM (Linear)",
    Main_Setting      = paste0("kernel=Linear | C=", svm_linear$bestTune$C),
    Threshold         = round(pick_best_thr_bal(y_val_svm, p_linear_val)$best$threshold, 2),
    Test_AUC          = round(as.numeric(auc(roc_linear_obj)), 3),
    Balanced_Accuracy = round(0.5 * (as.numeric(cm_linear$byClass["Sensitivity"]) +
                                       as.numeric(cm_linear$byClass["Specificity"])), 3),
    Sensitivity       = round(as.numeric(cm_linear$byClass["Sensitivity"]), 3),
    Specificity       = round(as.numeric(cm_linear$byClass["Specificity"]), 3)
  ),
  data.frame(
    Model             = "SVM (Radial)",
    Main_Setting      = paste0("kernel=Radial | C=", svm_radial$bestTune$C,
                               " | sigma=", svm_radial$bestTune$sigma),
    Threshold         = round(pick_best_thr_bal(y_val_svm, p_radial_val)$best$threshold, 2),
    Test_AUC          = round(as.numeric(auc(roc_radial_obj)), 3),
    Balanced_Accuracy = round(0.5 * (as.numeric(cm_radial$byClass["Sensitivity"]) +
                                       as.numeric(cm_radial$byClass["Specificity"])), 3),
    Sensitivity       = round(as.numeric(cm_radial$byClass["Sensitivity"]), 3),
    Specificity       = round(as.numeric(cm_radial$byClass["Specificity"]), 3)
  ),
  data.frame(
    Model             = "SVM (Polynomial)",
    Main_Setting      = paste0("kernel=Polynomial | C=", svm_poly$bestTune$C,
                               " | degree=", svm_poly$bestTune$degree),
    Threshold         = round(best_thr_svm, 2),
    Test_AUC          = round(auc_svm, 3),
    Balanced_Accuracy = round(bal_svm, 3),
    Sensitivity       = round(sens_svm, 3),
    Specificity       = round(spec_svm, 3)
  ),
  nn_summary
)

print(final_comparison %>% arrange(desc(Test_AUC)))

