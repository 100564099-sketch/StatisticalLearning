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
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3, index = folds, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final")


set.seed(42)

# logreg, LDA, QDA cross-validation as initial models
fit_logreg <- train(Has_Mental_Health_Issue ~ ., data=train_sc, method="glm", family=binomial(), metric="ROC", trControl=ctrl)
fit_lda    <- train(Has_Mental_Health_Issue ~ ., data=train_sc, method="lda", metric="ROC", trControl=ctrl)
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





# Threshold Selection
thr_grid <- seq(0.05, 0.95, by = 0.01)

calc_metrics <- function(y_true, prob_yes, thr) {
  pred <- factor(ifelse(prob_yes >= thr, "Yes", "No"), levels = c("Yes","No"))
  cm <- caret::confusionMatrix(pred, y_true, positive = "Yes")
  
  acc  <- as.numeric(cm$overall["Accuracy"])
  prec <- as.numeric(cm$byClass["Pos Pred Value"])
  sens <- as.numeric(cm$byClass["Sensitivity"])
  spec <- as.numeric(cm$byClass["Specificity"])
  f1   <- if (is.na(prec) || is.na(sens) || (prec + sens) == 0) NA_real_ else 2 * prec * sens / (prec + sens)
  
  data.frame(threshold = thr, Accuracy = acc, Precision = prec,
             Sensitivity = sens, Specificity = spec, F1 = f1)
}

pick_best_thr <- function(y_true, prob_yes) {
  tbl <- dplyr::bind_rows(lapply(thr_grid, function(t) calc_metrics(y_true, prob_yes, t))) |>
    dplyr::arrange(dplyr::desc(F1), dplyr::desc(Sensitivity), dplyr::desc(Accuracy))
  list(best = tbl[1, ], all = tbl)
}

# --- probabilities on VAL
p_logreg <- predict(fit_logreg, newdata = val_sc, type = "prob")[, "Yes"]
p_lda    <- predict(fit_lda,    newdata = val_sc, type = "prob")[, "Yes"]
p_qda    <- predict(fit_qda,    newdata = val_sc, type = "prob")[, "Yes"]

nb_train <- e1071::naiveBayes(Has_Mental_Health_Issue ~ ., data = train_sc)
p_nb     <- predict(nb_train, newdata = val_sc, type = "raw")[, "Yes"]

# --- pick best threshold per model (by F1)
best_logreg <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_logreg)$best; best_logreg$Model <- "LogReg"
best_lda    <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_lda)$best;    best_lda$Model    <- "LDA"
best_qda    <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_qda)$best;    best_qda$Model    <- "QDA"
best_nb     <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_nb)$best;     best_nb$Model     <- "NB"

val_threshold_summary <- dplyr::bind_rows(best_logreg, best_lda, best_qda, best_nb) |>
  dplyr::select(Model, threshold, Accuracy, Precision, Sensitivity, Specificity, F1) |>
  dplyr::arrange(dplyr::desc(F1), dplyr::desc(Sensitivity), dplyr::desc(Accuracy))

print(val_threshold_summary)

best_thresholds <- setNames(val_threshold_summary$threshold, val_threshold_summary$Model)

# >>> BURAYA KOY (best_thresholds satırının HEMEN ALTI) <<<
all_logreg <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_logreg)$all
all_lda    <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_lda)$all
all_qda    <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_qda)$all
all_nb     <- pick_best_thr(val_sc$Has_Mental_Health_Issue, p_nb)$all




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
  
  # default 0.5 line
  p <- p + geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray50", linewidth = 0.5)
  
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
    subtitle = "Dashed line = 0.5, Solid line = best threshold (by F1)"
  ) &
  theme(legend.position = "bottom")


# 1 = Yes Majority causes problems when selecting thresholds
# either find a metric that takes 0 into account or make No = 1 and Yes = 0
