
# ============================================================
# Project 2: Modelling daily cycling demand in Edinburgh
# Authors: [Yuxiang Wu s2471309], [Mingshi Yue s2520285], [Yijing Wang s2315893]
# ============================================================

#
# Place the code needed in the Report_project02.Rmd, including documentation.
#
#############
# 0. Packages
############
library(ggplot2); library(dplyr); library(tidyr)
library(lubridate); library(knitr); library(kableExtra)
library(patchwork); library(broom)

##############
# 1. Load data
##############
load('cycle_daily_df.Rdata')

###################
# 2. Data Wrangling
###################
cycle_daily_df <- cycle_daily_df %>%
  mutate(
    # Task 1 — ordered factor month: preserves Jan..Dec order in boxplots and
    # avoids alphabetical mis-ordering; labels = month.abb for readable axes.
    month_fct = factor(month, levels = 1:12,
                       labels = month.abb, ordered = TRUE),

    # Task 2 — explicit dow levels Mon..Sun: original data used Sun<Mon<..;
    dow = factor(dow,
                 levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
                 ordered = TRUE),

    # Task 3 — trend as integer days since study start.
    trend = as.integer(date - as.Date("2020-01-01"))
  )

# Exploratory Data Analysis

# The Summary table
summary_stats <- cycle_daily_df %>%
  select(count, temp_mean, temp_min, temp_max) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    Min    = min(Value),
    Q1     = quantile(Value, 0.25),
    Median = median(Value),
    Mean   = mean(Value),
    Q3     = quantile(Value, 0.75),
    Max    = max(Value),
    SD     = sd(Value),
    .groups = "drop"
  ) %>%
  mutate(Variable = factor(Variable,
                           levels = c("count", "temp_mean", "temp_min", "temp_max"))) %>%
  arrange(Variable)

# Time series
p_timeseries <- ggplot(cycle_daily_df, aes(x = date, y = count)) +
  geom_line(alpha = 0.25, linewidth = 0.3) +
  geom_smooth(method = "loess", span = 0.15, colour = "steelblue",
              se = FALSE, linewidth = 1) +
  labs(x = "Date", y = "Daily cyclist count") +
  theme_bw()

# Boxplots of count by month and by day of week
p_box_month <- ggplot(cycle_daily_df, aes(x = month_fct, y = count)) +
  geom_boxplot(fill = "steelblue", alpha = 0.4, outlier.size = 0.8) +
  labs(x = "Month", y = "Daily cyclist count") +
  theme_bw()

p_box_dow <- ggplot(cycle_daily_df, aes(x = dow, y = count)) +
  geom_boxplot(fill = "coral", alpha = 0.4, outlier.size = 0.8) +
  labs(x = "Day of week", y = "Daily cyclist count") +
  theme_bw()

# gridExtra avoids the patchwork guide-alignment bug with ordered factors
p_box_combined <- gridExtra::arrangeGrob(p_box_month, p_box_dow, ncol = 2)

# Median count by month and by day of week 
#(used in EDA text where i producing the numbers after each month or day)
median_by_month <- cycle_daily_df %>%
  group_by(month_fct) %>%
  summarise(median_count = median(count), .groups = "drop")

median_by_dow <- cycle_daily_df %>%
  group_by(dow) %>%
  summarise(median_count = median(count), .groups = "drop")

# Scatter vs temp_mean (not min/max): mean is the covariate used in M0–M3;
p_scatter_temp <- ggplot(cycle_daily_df, aes(x = temp_mean, y = count)) +
  geom_point(alpha = 0.25, size = 1) +
  geom_smooth(method = "loess", colour = "red", se = FALSE) +
  labs(x = "Mean daily temperature (°C)", y = "Daily cyclist count") +
  theme_bw()

monthly_year <- cycle_daily_df %>%
  group_by(year, month_fct) %>%
  summarise(mean_count = mean(count), .groups = "drop")

p_monthly_year <- ggplot(monthly_year,
                         aes(x = month_fct, y = mean_count,
                             colour = factor(year), group = year)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  labs(x = "Month", y = "Mean daily count", colour = "Year") +
  theme_bw()

#######################
# 3. Model Fitting
#######################

# Use factor(month) in formulas for M1-M3

# M0
m0 <- lm(count ~ temp_mean + weekend + month, data = cycle_daily_df)

cycle_daily_df <- cycle_daily_df %>%
  mutate(dow_model = factor(dow, ordered = FALSE),
         is_covid  = as.integer(year %in% c(2020, 2021)))
# M1
m1 <- lm(count ~ temp_mean + weekend + trend + factor(month) + dow_model,
          data = cycle_daily_df)
# M2
m2 <- lm(count ~ temp_mean + I(temp_mean^2) + weekend + trend +
           factor(month) + dow_model, data = cycle_daily_df)
# M3
m3 <- lm(count ~ temp_mean + I(temp_mean^2) + weekend + trend +
           factor(month) + dow_model + is_covid, data = cycle_daily_df)

coef_m0 <- tidy(m0) %>%
  mutate(p.value = ifelse(p.value < 2.2e-16, "< 2.2e-16",
                          formatC(p.value, format = "e", digits = 2)))

coef_m2 <- tidy(m2) %>%
  mutate(p.value = ifelse(p.value < 2.2e-16, "< 2.2e-16",
                          formatC(p.value, format = "e", digits = 2)))

# The Diagnostic plots
diag_plots <- function(model, model_name) {
  aug <- augment(model)

  p_resid <- ggplot(aug, aes(x = .fitted, y = .resid)) +
    geom_point(alpha = 0.2, size = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
    labs(x = "Fitted values", y = "Residuals",
         title = paste(model_name, "— Residuals vs Fitted")) +
    theme_bw()

  p_qq <- ggplot(aug, aes(sample = .resid)) +
    stat_qq(alpha = 0.2, size = 0.8) +
    stat_qq_line(colour = "red") +
    labs(x = "Theoretical quantiles", y = "Sample quantiles",
         title = paste(model_name, "— Normal QQ")) +
    theme_bw()

  gridExtra::arrangeGrob(p_resid, p_qq, ncol = 2)
}

diag_m0 <- diag_plots(m0, "M0")
diag_m1 <- diag_plots(m1, "M1")
diag_m2 <- diag_plots(m2, "M2")
diag_m3 <- diag_plots(m3, "M3")

##################################
# 4. Cross-Validation Functions
##################################

calc_scores <- function(y, mu, sigma, alpha = 0.05) {
  # y     : vector of observed values
  # mu    : vector of predictive means (from predict(fit, newdata=test)$fit)
  # sigma : vector of predictive SDs. Combine residual error and mean uncertainty:
  #         sigma = sqrt(summary(fit)$sigma^2 + predict(fit, newdata=test, se.fit=TRUE)$se.fit^2)
  # Returns a named list with RMSE, MAE, DS, IS
  rmse <- sqrt(mean((y - mu)^2))
  mae  <- mean(abs(y - mu))
  v    <- sigma^2
  ds   <- mean(log(v) + (y - mu)^2 / v)
  z    <- qnorm(1 - alpha / 2)
  lwr  <- mu - z * sigma
  upr  <- mu + z * sigma
  is_vec <- (upr - lwr) + (2 / alpha) * pmax(0, lwr - y) +
            (2 / alpha) * pmax(0, y - upr)
  is   <- mean(is_vec)
  list(RMSE = rmse, MAE = mae, DS = ds, IS = is)
}

loy_cv_one <- function(formula, data, model_label) {
  yrs <- sort(unique(data$year))
  fold_list <- vector("list", length(yrs))
  oos_list  <- vector("list", length(yrs))
  for (i in seq_along(yrs)) {
    y <- yrs[i]
    tr <- data[data$year != y, , drop = FALSE]
    te <- data[data$year == y, , drop = FALSE]
    fit <- lm(formula, data = tr)
    pr  <- predict(fit, newdata = te, se.fit = TRUE)
    mu  <- as.numeric(pr$fit)
    sig <- sqrt(summary(fit)$sigma^2 + as.numeric(pr$se.fit)^2)
    sc  <- calc_scores(te$count, mu, sig)
    fold_list[[i]] <- tibble::tibble(
      model = model_label, year = y,
      RMSE = sc$RMSE, MAE = sc$MAE, DS = sc$DS, IS = sc$IS
    )
    ds_i <- log(sig^2) + (te$count - mu)^2 / sig^2
    oos_list[[i]] <- tibble::tibble(
      model = model_label, month = te$month, y = te$count, mu = mu, ds_i = ds_i
    )
  }
  list(
    folds = dplyr::bind_rows(fold_list),
    oos   = dplyr::bind_rows(oos_list)
  )
}

f_cv_m0 <- count ~ temp_mean + weekend + month
f_cv_m1 <- count ~ temp_mean + weekend + trend + month + dow_model
f_cv_m2 <- count ~ temp_mean + I(temp_mean^2) + weekend + trend + month + dow_model
f_cv_m3 <- count ~ temp_mean + I(temp_mean^2) + weekend + trend + month + dow_model + is_covid

cv_m0 <- loy_cv_one(f_cv_m0, cycle_daily_df, "M0")
cv_m1 <- loy_cv_one(f_cv_m1, cycle_daily_df, "M1")
cv_m2 <- loy_cv_one(f_cv_m2, cycle_daily_df, "M2")
cv_m3 <- loy_cv_one(f_cv_m3, cycle_daily_df, "M3")

cv_folds_all <- dplyr::bind_rows(
  cv_m0$folds, cv_m1$folds, cv_m2$folds, cv_m3$folds
)

# Table 1 (Evaluation): mean score per model across held-out years
cv_model_summary <- cv_folds_all %>%
  dplyr::group_by(model) %>%
  dplyr::summarise(
    RMSE = mean(RMSE),
    MAE  = mean(MAE),
    DS   = mean(DS),
    IS   = mean(IS),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    RMSE = round(RMSE, 3),
    MAE  = round(MAE, 3),
    DS   = round(DS, 3),
    IS   = round(IS, 3)
  )

cv_results_list <- list(M0 = cv_m0, M1 = cv_m1, M2 = cv_m2, M3 = cv_m3)
oos_final <- cv_results_list[["M2"]]$oos

# Table 2 (Evaluation): RMSE and DS by calendar month for final model
cv_monthly_table <- oos_final %>%
  dplyr::group_by(month) %>%
  dplyr::summarise(
    RMSE = sqrt(mean((y - mu)^2)),
    DS   = mean(ds_i),
    .groups = "drop"
  ) %>%
  dplyr::arrange(month) %>%
  dplyr::mutate(
    Month = month.abb[month],
    RMSE  = round(RMSE, 1),
    DS    = round(DS, 2)
  ) %>%
  dplyr::select(Month, RMSE, DS)

# Table 3
days_per_year <- 365L
trend_m123_table <- tibble::tibble(
  Model = c("M1", "M2", "M3"),
  daily = c(
    unname(coef(m1)["trend"]),
    unname(coef(m2)["trend"]),
    unname(coef(m3)["trend"])
  )
) %>%
  dplyr::mutate(
    `Daily change (cyclists/day)`   = round(daily, 3),
    `Annual change (cyclists/year)` = round(daily * days_per_year, 0)
  ) %>%
  dplyr::select(Model, `Daily change (cyclists/day)`, `Annual change (cyclists/year)`)

#########################
# 5. Planning Questions
#########################

# 5.1 Temperature variable comparison

f_cv_m2_tmin <- count ~ temp_min + I(temp_min^2) + weekend + trend + month + dow_model
f_cv_m2_tmax <- count ~ temp_max + I(temp_max^2) + weekend + trend + month + dow_model

cv_m2_tmin <- loy_cv_one(f_cv_m2_tmin, cycle_daily_df, "temp_min")
cv_m2_tmax <- loy_cv_one(f_cv_m2_tmax, cycle_daily_df, "temp_max")

temp_cv_table <- dplyr::bind_rows(
  cv_m2$folds %>% dplyr::mutate(model = "temp_mean"),
  cv_m2_tmin$folds %>% dplyr::mutate(model = "temp_min"),
  cv_m2_tmax$folds %>% dplyr::mutate(model = "temp_max")
) %>%
  dplyr::group_by(model) %>%
  dplyr::summarise(
    RMSE = round(mean(RMSE), 1),
    MAE  = round(mean(MAE), 1),
    DS   = round(mean(DS), 2),
    IS   = round(mean(IS), 1),
    .groups = "drop"
  )

m2_tmax <- lm(count ~ temp_max + I(temp_max^2) + weekend + trend +
                factor(month) + dow_model, data = cycle_daily_df)

beta1_temp <- unname(coef(m2_tmax)["temp_max"])
beta2_temp <- unname(coef(m2_tmax)["I(temp_max^2)"])
marginal_5  <- beta1_temp + 2 * beta2_temp * 5
marginal_15 <- beta1_temp + 2 * beta2_temp * 15

# 5.3 Trend and extrapolation

trend_coef_m2 <- unname(coef(m2)["trend"])
trend_se_m2   <- summary(m2)$coefficients["trend", "Std. Error"]
trend_pval_m2 <- summary(m2)$coefficients["trend", "Pr(>|t|)"]
annual_change_m2 <- trend_coef_m2 * 365

baseline_row  <- cycle_daily_df[which.min(abs(cycle_daily_df$date -
                                                as.Date("2020-01-01"))), , drop = FALSE]
baseline_pred <- unname(predict(m2, newdata = baseline_row))
pct_annual    <- annual_change_m2 / baseline_pred * 100

summer_temp_mean <- mean(cycle_daily_df$temp_mean[cycle_daily_df$month %in% 6:8])

extrap_years <- 2020:2035
extrap_july  <- as.Date(paste0(extrap_years, "-07-01"))
extrap_trend <- as.integer(extrap_july - as.Date("2020-01-01"))

extrap_nd <- data.frame(
  temp_mean = summer_temp_mean,
  weekend   = 0L,
  trend     = extrap_trend,
  month     = 7L,
  dow_model = factor("Wed", levels = levels(cycle_daily_df$dow_model))
)

extrap_pr  <- predict(m2, newdata = extrap_nd, interval = "prediction", level = 0.95)
extrap_df  <- data.frame(
  year = extrap_years,
  fit  = extrap_pr[, "fit"],
  lwr  = extrap_pr[, "lwr"],
  upr  = extrap_pr[, "upr"]
)

planning_threshold <- 8000
below_idx      <- which(extrap_df$fit < planning_threshold)
threshold_year <- if (length(below_idx) > 0) extrap_df$year[below_idx[1]] else NA_integer_

p_extrapolation <- ggplot(extrap_df, aes(x = year, y = fit)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), alpha = 0.15, fill = "steelblue") +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_point(colour = "steelblue", size = 2) +
  geom_hline(yintercept = planning_threshold, linetype = "dashed", colour = "red") +
  annotate("text", x = 2020.5, y = planning_threshold + 300,
           label = paste0(format(planning_threshold, big.mark = ","), " cyclists/day"),
           hjust = 0, colour = "red", size = 3.5) +
  scale_x_continuous(breaks = seq(2020, 2035, by = 2)) +
  scale_y_continuous(labels = scales::comma) +
  labs(x = "Year",
       y = "Predicted daily count\n(July weekday, mean summer temp.)") +
  theme_bw()

# 5.4 Poorly-fit period

m2_aug <- broom::augment(m2, data = cycle_daily_df)

resid_month_year <- m2_aug %>%
  dplyr::group_by(year, month_fct) %>%
  dplyr::summarise(
    mean_resid = mean(.resid),
    rmse       = sqrt(mean(.resid^2)),
    n          = dplyr::n(),
    .groups    = "drop"
  )

worst_period <- resid_month_year %>%
  dplyr::arrange(dplyr::desc(rmse)) %>%
  dplyr::slice(1)

worst_direction <- ifelse(worst_period$mean_resid > 0, "under-predicts", "over-predicts")

p_resid_time <- ggplot(m2_aug, aes(x = date, y = .resid, colour = factor(year))) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Date", y = "Residual (observed - fitted)", colour = "Year") +
  theme_bw() +
  theme(legend.position = "bottom")

# 5.5 Original diagnostic: month x year residual heatmap

p_resid_heatmap <- ggplot(resid_month_year,
                          aes(x = factor(year), y = month_fct, fill = mean_resid)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, name = "Mean\nresidual",
                       labels = scales::comma) +
  geom_text(aes(label = round(mean_resid, 0)), size = 2.5, colour = "grey30") +
  labs(x = "Year", y = "Month") +
  theme_minimal() +
  theme(panel.grid = element_blank())
