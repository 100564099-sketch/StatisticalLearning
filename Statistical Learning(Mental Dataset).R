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


# Load dataset
mental = read.csv("mental_health.csv")

# Check size and structure of data
dim(mental)     # number of rows and columns
str(mental)     # variable types


# 2. Data Preparation
# Convert target variable to factor (classification task)
#mental$Has_Mental_Health_Issue = as.factor(mental$Has_Mental_Health_Issue)

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

# 3. target distribution 
table(mental$Has_Mental_Health_Issue)
prop.table(table(mental$Has_Mental_Health_Issue))

# Notes: The target variable is highly imbalanced, with 7.84% of observations in class 0 and 92.16% in class 1. 

# Visualize
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

par(mfrow=c(3,3))

lapply(cat_vars, function(v) {
  
  # Calculate class proportions per category
  tab <- prop.table(table(mental[[v]],
                          mental$Has_Mental_Health_Issue),1)
  
  barplot(tab,
          col=c("lightblue","pink"),
          main=paste("Class Proportions by", v),
          ylab="Proportion")
})



# 5. EDA - Numeric Variables 
num_vars <- names(mental)[sapply(mental, is.numeric)]

# Separate numeric variables by type
binary_vars <- num_vars[sapply(mental[num_vars], function(x)
  length(unique(x)) == 2)]

discrete_vars <- num_vars[sapply(mental[num_vars], function(x)
  length(unique(x)) > 2 & length(unique(x)) <= 15)]

continuous_vars <- num_vars[sapply(mental[num_vars], function(x)
  length(unique(x)) > 15)]

# Binary numeric variables

par(mfrow=c(3,3))

lapply(binary_vars, function(v) {
  
  tab <- prop.table(table(mental[[v]], mental$Has_Mental_Health_Issue),1)
  
  barplot(t(tab),
          legend = TRUE,
          beside=TRUE,
          main=paste("Class Proportions by", v),
          col=c("lightblue","pink"))
})
# Discrete variables (0-10 scales)
par(mfrow=c(3,3))

lapply(discrete_vars, function(v) {
  
  boxplot(mental[[v]] ~ mental$Has_Mental_Health_Issue,
          main=paste("Distribution of", v, "by Class"),
          xlab="Mental Health Issue",
          ylab=v)
})

# Continuous variables (density)

mental_long <- mental %>%
  select(all_of(continuous_vars), Has_Mental_Health_Issue) %>%
  pivot_longer(-Has_Mental_Health_Issue,
               names_to="Variable",
               values_to="Value")

ggplot(mental_long,
       aes(x=Value, fill=Has_Mental_Health_Issue)) +
  geom_density(alpha=0.4) +
  facet_wrap(~Variable, scales="free") +
  theme_minimal() +
  labs(title="Distribution of Continuous Variables by Mental Health Class")


# EDA conclusion

# Demographic categorical variables (Gender, Country, Marital_Status)
# do not clearly separate the two classes.

# Psychological and stress-related numeric variables show clearer differences.

# Variables like Work_Stress_Level, Feeling_Sad_Down,
# Financial_Stress, and Anxious_Nervous look like strong predictors.



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

Model performance was estimated using repeated stratified cross-validation on the standardized, SMOTEd training set. Specifically, 5-fold cross-validation repeated 3 times was used to reduce variance in performance estimates. ROC-AUC was selected as the primary metric (metric = "ROC") because it is threshold-independent and suitable for imbalanced classification.

Four baseline classifiers were evaluated:

Logistic Regression (GLM, binomial)

Linear Discriminant Analysis (LDA)

Quadratic Discriminant Analysis (QDA)

Naive Bayes

Logistic regression, LDA, and QDA were trained via caret::train using the same resampling scheme and ROC-based summary function. Naive Bayes was evaluated using the identical fold indices in a manual loop to compute fold-level ROC-AUC values, which were then averaged.

The resulting cross-validated ROC-AUC scores were:

LogReg: 0.724

LDA: 0.724

QDA: 0.899

Naive Bayes: 0.811
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

Balanced Accuracy=(Sensitivity+Specificity) / 2

Balanced Accuracy was used because it gives equal importance to both classes by averaging true-positive rate (Sensitivity) and true-negative rate (Specificity), making it more reliable than Accuracy or F1 in imbalanced settings.

For each model, the threshold that maximized Balanced Accuracy (with tie-breaking toward higher Sensitivity and then higher Specificity) was selected. The resulting optimal thresholds on the validation set were:

Logistic Regression: threshold = 0.44, BalancedAcc = 0.663

LDA: threshold = 0.44, BalancedAcc = 0.660

Naive Bayes: threshold = 0.45, BalancedAcc = 0.638

QDA: threshold = 0.89, BalancedAcc = 0.569

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
  pred_label <- factor(ifelse(prob >= threshold, "Yes", "No"), levels = c("Yes", "No"))
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

# Print final results
print(Final_Results)


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
           x = 0.65, y = 0.25,   # istersen konumu değiştir
           label = auc_text,
           hjust = 0, vjust = 0,
           size = 4)



