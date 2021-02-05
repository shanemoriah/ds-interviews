library(tidyverse) # for data manipulation and visualization
library(rio) # for data i/o
library(caret) # for model routines
library(plotROC) # for RoC routines


#==== general setup ====
# working directory
setwd("C:/Users/Wenyao/Desktop/R/ds-interviews/takehome")
set.seed(1)

options(warn = -1)
options(stringsAsFactors = FALSE) # this is no longer needed since R 4.0.0


#==== input ====
# first row contains invalid data
loan_data_raw <- import("LoanData.csv.gz", skip = 1)


#==== data exploration and cleaning ====
# check the possible values of the label variable
loan_data_raw %>% 
  count(loan_status)

# preliminary data cleaning
loan_data <- loan_data_raw %>% 
  # filter for the 2 cases that we care about in order to setup a binary classification context
  filter(loan_status %in% c("Charged Off", "Fully Paid")) %>% 
  # convert variables into their appropriate format
  mutate(
    loan_status = factor(loan_status, levels = c("Charged Off", "Fully Paid")),
    term = term %>% str_extract("\\d+") %>% as.numeric(),
    emp_length = emp_length %>% str_extract("\\d+") %>% as.numeric(),
    revol_util = revol_util %>% gsub("%", "", .) %>% as.numeric(),
    int_rate = int_rate %>% gsub("%", "", .) %>% as.numeric()
  )

# find the list of variables that don't seem to be useful
variables_to_delete <- c(
  # variables with too many missing values (>=90%)
  loan_data %>% 
    summarise_all(function(v){(is.na(v) | v == "") %>% sum()}) %>% 
    gather(variable, missing_count) %>% 
    filter(missing_count >= 0.9 * nrow(loan_data)) %>% 
    pull(variable),
  
  # numerical variables with 0 variance
  loan_data %>% 
    select_if(is.numeric) %>% 
    summarise_all(sd) %>% 
    gather(variable, sd) %>% 
    filter(sd == 0) %>% 
    pull(variable),
  
  # categorical variables with only 1 case
  loan_data %>% 
    select_if(is.character) %>% 
    summarise_all(function(v){v %>% unique() %>% length()}) %>% 
    gather(variable, cases) %>% 
    filter(cases == 1) %>% 
    pull(variable)
)

loan_data_final <- loan_data %>% 
  # exclude variables that aren't useful
  select(
    # previously-identified variables to delete
    -!!variables_to_delete,
    # other trivial variables
    -id, -issue_d, -url, -earliest_cr_line, 
    # date variables
    -ends_with("_d"), -contains("_date")
  ) %>% 
  # impute missing values with median
  mutate_if(
    is.numeric, 
    function(v){ifelse(is.na(v), median(v, na.rm = TRUE), v)}
  ) %>% 
  # convert categorical variables to factors
  mutate_if(
    is.character, as.factor
  )


#==== fit model ====
# training-testing split: 80-20
training_index <- createDataPartition(loan_data_final$loan_status, p = 0.8, list = FALSE)
training_data <- loan_data_final[training_index,]
testing_data <- loan_data_final[-training_index,]

# oversample the minority class due to the imbalanced nature
training_data_final <- upSample(
  x = training_data %>% select(-loan_status),
  y = training_data$loan_status,
  yname = "loan_status"
)

# train model
# this step can be iterative depending on 
# 1) subjective judgment (based prior knowledge of what tends to correlate with a default) and 
# 2) variable importance (as indicated by z-stats)
model_logistic_regression <- train(
  form = loan_status ~ loan_amnt + funded_amnt + term + int_rate +
    home_ownership + grade + sub_grade + annual_inc + dti + delinq_2yrs +
    fico_range_low + mths_since_last_delinq + revol_util +
    total_pymnt + chargeoff_within_12_mths + delinq_amnt +
    hardship_flag + debt_settlement_flag + verification_status,
  data = training_data_final,
  method = "glm",
  family = "binomial"
)

# variable importance (indicated by z-stats of the coefficient estimates)
varImp(model_logistic_regression)


#==== evaluate model =====
# model forecast on both training and testing data (i.e., scoring)
prediction <- 
  pmap(
    list(
      data = list(training_data, testing_data),
      type = c("in-sample", "out-of-sample")
    ),
    function(data, type){
      tibble(
        # actual label
        actual = data$loan_status,
        # predicted label
        predicted = predict(
          model_logistic_regression, newdata = data
        ),
        # predicted probability
        predict(
          model_logistic_regression, newdata = data, type = "prob"
        ),
        type = type
      )
    }
  ) %>% 
  bind_rows() %>% 
  mutate(
    type = factor(type, levels = c("in-sample", "out-of-sample")),
    correct = actual == predicted
  )

# in-sample confusion matrix
confusionMatrix(
  prediction %>% filter(type == "in-sample") %>% pull(predicted),
  prediction %>% filter(type == "in-sample") %>% pull(actual)
)$table
# out-of-sample confusion matrix
confusionMatrix(
  prediction %>% filter(type == "out-of-sample") %>% pull(predicted),
  prediction %>% filter(type == "out-of-sample") %>% pull(actual)
)$table


#==== plots ====
# roc plot
plot_roc <- ggplot(prediction, aes(m = `Fully Paid`, d = actual)) + 
  geom_roc(aes(color = type), n.cuts = 0, size = 2) +
  theme_minimal() +
  theme(
    text = element_text(size = 30),
    legend.position = "none",
    legend.title = element_blank(),
    strip.background = element_rect(fill = "gray90", color = NULL)
  ) + 
  facet_wrap(.~type) +
  ggtitle("RoC (AuC = 98.00%, 97.86%)")
calc_auc(plot_roc)

# visualize the confusion matrix
plot_confusion_matrix <- prediction %>% 
  group_by(type, actual) %>% 
  summarise(
    correct_ratio = sum(correct) / n(),
    incorrect_ratio = (n() - sum(correct)) / n()
  ) %>% 
  gather(category, value, -type, -actual) %>% 
  mutate(
    category = factor(
      category,
      levels = c("incorrect_ratio", "correct_ratio"),
      labels = c("incorrect", "correct")
    )
  ) %>% 
  group_by(type, actual) %>% 
  arrange(type, actual, desc(category)) %>% 
  mutate(
    text_y = value / 2 + lag(value, default = 0)
  ) %>% 
  ggplot(aes(x = actual)) +
  geom_bar(aes(weight = value, fill = category)) +
  geom_label(aes(label = paste0(round(value * 100, 2), "%"), y = text_y), size = 10) +
  scale_fill_manual(values = c("incorrect" = "tomato", "correct" = "dodgerblue")) +
  facet_wrap(.~type) +
  ggtitle("Model Performance") +
  theme_minimal() +
  theme(
    text = element_text(size = 30),
    strip.background = element_rect(fill = "gray90", color = NULL),
    legend.position = "top",
    legend.title = element_blank(),
    axis.title = element_blank()
  )


#==== output ====
png("plot_roc.png", width = 800, height = 500, type = "cairo", bg = "white")
print(plot_roc)
dev.off()

png("plot_confusion_matrix.png", width = 800, height = 500, type = "cairo", bg = "white")
print(plot_confusion_matrix)
dev.off()
