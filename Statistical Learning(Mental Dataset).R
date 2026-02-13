setwd("/Users/utkusizanli/Desktop/UC3M/Statistical Learning/Midterm1")

# Load dataset
mental = read.csv("mental_health.csv")

# Check size and structure of data
dim(mental)     # number of rows and columns
str(mental)     # variable types

# Convert target variable to factor (classification task)
mental$Has_Mental_Health_Issue = as.factor(mental$Has_Mental_Health_Issue)

# Convert all character columns to factors automatically
mental[] = lapply(mental, function(x) if (is.character(x)) as.factor(x) else x)

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
  abs(mean(x[mental$Has_Mental_Health_Issue==1]) -
        mean(x[mental$Has_Mental_Health_Issue==0]))
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



# SOME CHANGES

