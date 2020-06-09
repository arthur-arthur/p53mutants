# =============================================================================
# Big Data Science: Project
# =============================================================================
#
# Task 1: p53 mutants classification
# Analysis & visualizations of output generated during optimization
# -----------------------------------------------------------------------------
# NOTE: all files are sourced relative to the working directory
#
# Arthur Leloup
# 26/05/2020
# =============================================================================

# setup
library(tidyverse)
library(cowplot)
library(plot3D)
theme_set(theme_bw())

# set wd and source helper functions
source("helper_functions/analysis_results.R")

# =============================================================================
# Exploratory data analysis
# =============================================================================

#' Data for the visualizations are outputted by the 0-EDA notebook.

DIR <- "0__EDA"

# plot some strongly correlated feature pairs
df <- read_csv(file.path(DIR, "pairwise_corr.csv"))

# inactive class is downsampled
ggplot(df, aes(x1, x2, color=class)) +
  geom_point(size=0.5, alpha=0.3) +
  geom_point(data=subset(df, class == "active"), 
             mapping=aes(x1, x2), size=0.5, alpha=0.3) +
  facet_wrap(~pair, scales="free") +
  scale_color_manual(values = c("grey30", "skyblue")) +
  xlab("Feature 1") +
  ylab("Feature 2") +
  guides(color = guide_legend(override.aes = list(size=3))) +
  labs(color = "Class label") +
  theme(legend.position="bottom")

# low-dimensional representations: first 3 components of PCA
# for all features
pca_all <- read_csv(file.path(DIR, "PCA_allfeatures.csv"))
# for feature 4827:5408 (3d distances)
pca_3d <- read_csv(file.path(DIR, "PCA_3Dfeatures.csv"))

PCs <- list(pca_all, pca_3d)

par(mfrow = c(1, 2), mar=rep(1, 4))
for (i in seq_along(PCs)) {
  pc = PCs[[i]]
  scatter3D(
    pc$PC1, pc$PC2, pc$PC3, 
    colvar = pc$class, pch=19, bty = "f", cex=0.5,
    col = c("skyblue", "grey30"), alpha=0.75,
    colkey = list(addlines=TRUE, at=c(0, 1), length=0.05),
    phi=20, theta=40,
    ticktype = "detailed"
  )
}

# =============================================================================
# Logistic Regression Classifier (LRC)
# =============================================================================

#' Feature selection / dimension reduction
#' ----------------------------------------------------------------------------
#' Visual exploration of output from 1-LRC-DimRed-A notebook, i.e. an initial
#' (coarse) grid search on the downsampled ds.

DIR <-"1__DimRed/LRC/A/v1"
df1 <- read_results(DIR) %>%
  mutate(remove_corr_n = 0) %>%
  select(remove_corr_n, scaler:val_balanced_acc) %>%
  mutate(ref = ifelse(scaler == "no_scaling", 0.339, 
                      ifelse(scaler == "StandardScaler", 0.369, 0.414)))

DIR <-"1__DimRed/LRC/A/v2"
df2 <- read_results(DIR) %>%
  select(remove_corr_n:val_balanced_acc) %>%
  mutate(ref=NA)

# join v1-v2
names(df1) == names(df2)
df <- rbind(df1, df2)

df <- df %>%
  mutate(fs_model = ifelse(fs_model == "na", "no PP", fs_model)) %>%
  mutate(scaler = ifelse(scaler == "no_scaling", "No scaling", scaler)) %>%
  separate(fs_model, into = c("fs_model", "fs_hyperparam"), sep = "_") %>%
  mutate(fs_m = as.numeric(fs_m),
         fs_univ_k = as.numeric(fs_univ_k))

# reshape to long format
df_long = to_long(df)

# plot train vs test
f1 <- ggplot(subset(df_long, fs_model != "no PP" & remove_corr_n == 0),
             aes(tot_n_features, mcc, color = fold)) +
  geom_point(alpha=0.25) +
  geom_smooth() +
  geom_hline(aes(yintercept = ref), color="black", size=0.5, alpha=0.6) +
  facet_grid(cols=vars(fs_model), rows=vars(scaler)) +
  scale_y_continuous(limits=0:1) +
  xlab("Total # of features") +
  ylab("MCC") +
  theme(legend.position = "bottom") +
  labs(color="Fold"); f1

# plot effect of model-based selection
ggplot(df, aes(tot_n_features, val_mcc, color=fs_model)) +
  geom_jitter(alpha=0.45, size=1) +
  geom_hline(aes(yintercept = ref), color="purple", size=1, alpha=0.5) +
  #geom_smooth() +
  scale_x_continuous(limits=c(0, 3000)) +
  xlab("Total # of features") +
  ylab("Validation fold MCC") +
  facet_grid(rows=vars(scaler)) +
  theme(legend.position = "bottom") +
  labs(color="FS-model")

# effect of removing redudant features (~Pearson correlation)
ggplot(df, aes(remove_corr_n, val_mcc, color=fs_model)) +
  geom_jitter(alpha=0.45) +
  geom_hline(aes(yintercept = ref), color="purple", size=1, alpha=0.5) +
  #geom_smooth() +
  ylab("Validation fold MCC") +
  facet_grid(rows=vars(scaler)) +
  theme(legend.position = "bottom") +
  labs(color="FS-model")
  # limited effect when scaled, but also no major reduction of performance
  # with a gain in efficiency, i.e. consider for further optimization steps

# continue with StandardScaler and LinearSVC fs-model
df2 <- df %>%
  filter(fs_model=="LinearSVC" & scaler == "StandardScaler")

lr_pp <- ggplot(df2, aes(fs_m, val_mcc, color=fs_univ_k)) +
  geom_jitter(alpha=0.8, size=1) +
  geom_smooth(alpha=0.5, color="grey60") +
  facet_grid(cols=vars(pca_n)) +
  xlab("LinearSVC threshold (m)") +
  ylab("Validation MCC") +
  labs(color = "Univ threshold (k)") +
  ggtitle("Preprocessing and feature-selection of the LR pipeline") +
  theme(legend.position="bottom") ; lr_pp

# optimal m around 500
df3 <- df2 %>%
  filter(fs_m == 500)

ggplot(df3, aes(fs_univ_k, val_mcc, color=pca_n)) +
  geom_point()+
  facet_grid(cols=vars(pca_n))

# focus on low range of k
ggplot(subset(df3, fs_univ_k < 200), 
       aes(fs_univ_k, val_mcc)) +
  geom_point(size=2)+
  facet_grid(cols=vars(pca_n))
  # optimal seems to be a low nr of pc's (10), and ~50 original 
  # features based on univariate criteria

df3 %>% arrange(desc(val_mcc))

ggplot(df3, aes(val_balanced_acc, val_mcc)) +
  geom_point() +
  geom_label(aes(label=paste("n =", pca_n)), size=2.5) +
  geom_label(aes(label=paste("k =", fs_univ_k)), 
             col='orange', nudge_y = -0.008, size=2.5) +
  geom_label(aes(label=paste("tot =", tot_n_features)), 
             col='darkred', nudge_y = 0.008, size=2.5) +
  xlab("Balanced accuracy") +
  ylab("Validation fold MCC") +
  scale_x_continuous(limits = c(0.75, 0.825)) +
  scale_y_continuous(limits=c(0.46, 0.57)); f3

plot_grid(f1, f2, nrow = 1, labels="AUTO", rel_widths = c(1, 2))

#' Resampling algorithms
#' ----------------------------------------------------------------------------
#' Visual exploration of output from 2-LRC-Resampling notebook, i.e. 
#' resampling of the dimension-reduced ds (optimized in LRC-DimRed-A and B.
#' SMOTE, SVMSMOTE, ADASYN oversampling and random over- and undersamplers
#' were tested to obtain different class ratio's.

DIR <- "2__Resampling/LRC"
df_csl <- read_csv(file.path(DIR, "results_resampling.csv")) %>%
  mutate(csl = TRUE) %>%
  select(seed, csl, upsampler, downsampler, up_class_ratio, 
         down_vs_up_ratio, final_class_ratio, train_mcc, val_mcc)
df_nocsl <- read_csv(file.path(DIR, "results_resampling_cost_insensitive.csv")) %>%
  mutate(csl = FALSE) %>%
  select(seed, csl, upsampler, downsampler, up_class_ratio, 
         down_vs_up_ratio, final_class_ratio, train_mcc, val_mcc)

df <- rbind(df_csl, df_nocsl)

# average over the random seeds
conditions <- names(df)[2:7]
df_summ <- df %>%
  group_by_at(conditions) %>%
  summarize(n = n(),
            train_mcc = mean(train_mcc),
            val_mcc = mean(val_mcc))

ggplot(df_summ, aes(final_class_ratio, val_mcc, color=down_vs_up_ratio)) +
  geom_point()
  # downsampling generally reduces performance

ggplot(df_summ, aes(down_vs_up_ratio, val_mcc)) +
  geom_point()

df2 <- df %>%
  filter(down_vs_up_ratio == 1)

ggplot(df2, aes(upsampler, val_mcc, fill=upsampler)) +
  geom_boxplot(alpha=0.5) +
  facet_grid(cols=vars(final_class_ratio)) +
  scale_y_continuous(limits = c(0.3, 0.7)) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = "bottom")

#' Evaluation of best-performing pipelines (LRC)
#' ----------------------------------------------------------------------------
#' Visual exploration of output from 4-LRC-Eval notebook, i.e. the estimated
#' extra-sample classification performance based on the held-out test set.

# load data
DIR <- "4__Eval/LRC"
files <- file.path(DIR, dir(DIR))
# summary measures
eval <- read.csv(files[2])
# precision-recall curves
pr <- read.csv(files[1])

# clean/tidy/reshape
eval_long <- eval %>%
  select(-X) %>%
  pivot_longer(cols = -model, names_to = "metric", values_to = "value")

eval_ <- eval %>%
  mutate(resampling = ifelse(
    str_starts(model, "SVMSMOTE"), "RS", "No RS"),
    CSL = ifelse(
      str_ends(model, "CSL"), "CSL", "No CSL")) %>%
  filter(model != "LR (no PP)")

pr_ <- pr %>%
  mutate(resampling = ifelse(
    str_starts(model, "SVMSMOTE"), "RS", "No RS"),
    CSL = ifelse(str_ends(model, "CSL"), "CSL", "No CSL")) %>%
  filter(model != "LR (no PP)")

# reference: no preprocessing or CSL
noPP <- pr %>%
  filter(model == "LR (no PP)")

# plot PR curves
f3 <- ggplot(pr_, aes(recall, precision)) +
  geom_path(size=1, alpha=0.8, color="orange") +
  geom_path(data=noPP, color="skyblue", size=0.5, alpha=0.8) +
  facet_grid(rows=vars(resampling), cols = vars(CSL)) +
  theme(legend.position = "none") +
  xlab("Recall") +
  ylab("Precision") +
  geom_label(
    data=eval_, 
    mapping=aes(
      x=0.75, y=0.9, 
      label = paste("MCC:", round(mcc, 2))),
    size=3,
    alpha=0.75); f3

# confusion matrices
conf <- eval_long %>%
  filter(metric %in% c('tp', "fp", "fn", "tn")) %>%
  rename(count = value) %>%
  filter(model != "LR (no PP)") %>%
  mutate(true_class = ifelse(metric %in% c("tp", "fn"), "active", "inactive"),
         pred_class = ifelse(metric %in% c("fp", "tp"), "active", "inactive"),
         resampling = ifelse(
           str_starts(model, "SVMSMOTE"), "RS", "No RS"),
         CSL = ifelse(str_ends(model, "CSL"), "CSL", "No CSL"))

conf <- conf %>%
  left_join(
    conf %>%
      group_by(model, true_class) %>%
      summarize(true_tot = sum(count))
  ) %>%
  mutate(prop = count / true_tot)

f4 <- ggplot(conf, aes(true_class, pred_class, fill = prop, label=count)) +
  geom_tile() +
  geom_label(fill = "white", size=3, alpha = 0.75) +
  xlab("True class label") +
  ylab("Predicted class label") +
  scale_x_discrete(limits=c("inactive", 'active')) +
  scale_y_discrete(limits=c("inactive", 'active')) + 
  facet_grid(rows=vars(resampling), cols=vars(CSL)) +
  labs(fill = "Proportion") +
  theme(legend.position = "bottom"); f4

conf_matrix_legend <- get_legend(f4)
evalLRC <- plot_grid(
  f3, f4 + theme(legend.position = "none"),
  nrow = 1, labels = "AUTO", rel_widths = c(1.3, 1)); evalLRC

# =============================================================================
# Random Forest Classifier (RFC)
# =============================================================================

#' Feature selection / dimension reduction
#' ----------------------------------------------------------------------------
#' Visual exploration of output from 1-RFC-DimRed notebook, i.e. an initial
#' (coarse) grid search on a downsampled dataset.

# setup
DIR <- "1__DimRed/RFC/A"
df <- read_results(DIR)

table(df$fs_model)

df <- df %>%
  separate(fs_model, into = c("fs_model", "fs_hyperparam"), sep = "_")

top <- df %>%
  arrange(desc(val_mcc))

df_long <- to_long(df)

ggplot(df_long, aes(tot_n_features, mcc, color=fold))  +
  geom_jitter(alpha=0.3) +
  geom_smooth()

# plot effect of model-based selection
f2 <- ggplot(df, aes(tot_n_features, val_mcc, color=fs_model)) +
  geom_jitter(alpha=0.45) +
  #geom_smooth() +
  xlab("Total # of features") +
  ylab("Validation fold MCC") +
  facet_grid(rows=vars(scaler)) +
  theme(legend.position = "bottom") +
  labs(color="FS-model"); f2

ggplot(df, aes(tot_n_features, val_mcc, color=fs_model)) +
  geom_point(alpha=0.5) +
  facet_grid(cols=vars(scaler))

ggplot(subset(df, scaler != "no_scaling"),
       aes(scaler, val_mcc, color=scaler, shape=factor(max_depth))) +
  geom_point() +
  geom_line(aes(group=interaction(fs_model, fs_m, max_depth, 
                                  tot_n_features, n_estimators)))

ggplot(subset(df, val_mcc > 0.5), 
       aes(tot_n_features, val_mcc, color=scaler, shape=fs_model)) +
  geom_point(size=3, alpha=0.5)

# filter
df2 <- df %>%
  filter(fs_model=="ExtraTreesClf" & scaler=="RobustScaler")

ggplot(df2, aes(fs_m, val_mcc,color=max_depth))+
  geom_point(size=4, alpha=0.5)

# select m == 100
df3 <- df2 %>%
  filter(fs_m == 100)

ggplot(df3, aes(n_estimators, val_mcc, shape=factor(max_depth), color=fs_hyperparam))+
  geom_jitter(size=5, alpha=0.5)

df3 <- df2 %>%
  filter(n_estimators %in% c(500, 1000) & max_depth %in% c("10", "20"))

ggplot(df3, aes(fs_hyperparam, val_mcc, color=factor(max_depth))) +
  geom_point()

df3 <- df3 %>%
  filter(fs_hyperparam == 50 & max_depth==20)

ggplot(df3, aes(pca_n, val_mcc, color=factor(max_depth))) +
  geom_point(size=5)

df3 %>% filter(pca_n == 0 & fs_m == 100)

#' Resampling
#' ----------------------------------------------------------------------------
#' Visual exploration of output from 2-RFC-Resampling notebook

DIR <- "2__Resampling/RFC"
df <- read_results(DIR)

ref = mean(df[df$upsampler == "no_resampling", "val_mcc", drop = TRUE])

rfc1 <- ggplot(subset(df, !is.na(down_vs_up_ratio)), 
       aes(final_class_ratio, val_mcc, color=upsampler)) +
  geom_point(size=0.5, alpha=0.5) +
  geom_hline(yintercept = ref) +
  geom_smooth() +
  xlab("Resampled class ratio") +
  ylab("Validation MCC") +
  labs(color="Upsampling algorithm") +
  theme(legend.position = "bottom") +
  ggtitle("Resampling strategies for the RFC pipeline") +
  facet_grid(cols=vars(down_vs_up_ratio)); rfc1

ggplot(subset(df, upsampler != "no_resampling"),
       aes(upsampler, val_mcc)) +
  geom_boxplot() +
  facet_grid(cols=vars(up_class_ratio), rows=vars(down_vs_up_ratio)) +
  geom_hline(yintercept = ref) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = ref, 
           fill = "grey", alpha = .3)

# less resampling is better, downsampling reduces performance for all
# conditions

df2 <- df %>% 
  filter(upsampler %in% c("no_resampling", "SVMSMOTE") &
           up_class_ratio == 0.01)

ggplot(df2, aes(factor(final_class_ratio), val_mcc, color=down_vs_up_ratio)) +
  geom_boxplot() +
  geom_hline(yintercept =ref)

# conclusion: SVMSMOTE: upsample to 0.01, no downsampling

#' Evaluation of best-performing pipelines (RFC)
#' ----------------------------------------------------------------------------
#' Visual exploration of output from 4-RFC-Eval notebook, i.e. the estimated
#' extra-sample classification performance based on the held-out test set.

# load data
DIR <- "4__Eval/RFC"
files <- file.path(DIR, dir(DIR))
# summary measures
eval <- read.csv(files[2])
# precision-recall curves
pr <- read.csv(files[1])

# clean/tidy/reshape
eval_long <- eval %>%
  select(-X) %>%
  pivot_longer(cols = -model, names_to = "metric", values_to = "value")

eval_ <- eval %>%
  mutate(resampling = ifelse(
    str_starts(model, "SVMSMOTE"), "RS", "No RS"),
    CSL = ifelse(
      str_ends(model, "CSL"), "CSL", "No CSL")) %>%
  filter(model != "RFC (no PP)")

pr_ <- pr %>%
  mutate(resampling = ifelse(
    str_starts(model, "SVMSMOTE"), "RS", "No RS"),
    CSL = ifelse(str_ends(model, "CSL"), "CSL", "No CSL")) %>%
  filter(model != "RFC (no PP)")

# reference: no preprocessing or CSL
noPP <- pr %>%
  filter(model == "RFC (no PP)")

table(pr$model)

# plot PR curves
f1 <- ggplot(pr_, aes(recall, precision)) +
  geom_path(size=1, alpha=0.8, color="orange") +
  geom_path(data=noPP, color="skyblue", size=0.5, alpha=0.8) +
  facet_grid(rows=vars(resampling), cols = vars(CSL)) +
  theme(legend.position = "none") +
  xlab("Recall") +
  ylab("Precision") +
  geom_label(
    data=eval_, 
    mapping=aes(
      x=0.75, y=0.9, 
      label = paste("MCC:", round(mcc, 2))),
    size=3,
    alpha=0.75); f1

# confusion matrices
conf <- eval_long %>%
  filter(metric %in% c('tp', "fp", "fn", "tn")) %>%
  rename(count = value) %>%
  filter(model != "RFC (no PP)") %>%
  mutate(true_class = ifelse(metric %in% c("tp", "fn"), "active", "inactive"),
         pred_class = ifelse(metric %in% c("fp", "tp"), "active", "inactive"),
         resampling = ifelse(
           str_starts(model, "SVMSMOTE"), "RS", "No RS"),
         CSL = ifelse(str_ends(model, "CSL"), "CSL", "No CSL"))

conf <- conf %>%
  left_join(
    conf %>%
      group_by(model, true_class) %>%
      summarize(true_tot = sum(count))
  ) %>%
  mutate(prop = count / true_tot)

f2 <- ggplot(conf, aes(true_class, pred_class, fill = prop, label=count)) +
  geom_tile() +
  geom_label(fill = "white", size=3, alpha = 0.75) +
  xlab("True class label") +
  ylab("Predicted class label") +
  scale_x_discrete(limits=c("inactive", 'active')) +
  scale_y_discrete(limits=c("inactive", 'active')) + 
  facet_grid(rows=vars(resampling), cols=vars(CSL)) +
  labs(fill = "Proportion") +
  theme(legend.position = "none"); f2

evalRFC <- plot_grid(
  f1, f2, nrow = 1, labels = c("C", "D"),
  rel_widths = c(1.3, 1)); evalRFC

# =============================================================================
# Neural Network (NN) classifier
# =============================================================================

#' Optimization of the NN architecture & PP steps
#' ----------------------------------------------------------------------------
#' Visual exploration of output of the 1-NN-Opt notebook, i.e. optimization
#' of the NN architecture & hyperparameters

# read data: only scaling
DIR <- "NN/1__Opt/v1"
df1 <- read_results(DIR) %>%
  mutate(n_remove = 0) %>%
  select(fold, n_remove, scaler:mcc)
# highly correlated features removed during preprocessing
DIR <- "NN/1__Opt/v2"
df2 <- read_results(DIR) %>%
  select(fold, n_remove, scaler:mcc)

# combine
df <- rbind(df1, df2)
names(df)
# aggregate folds
conditions <- names(df)[2:12]; conditions
metrics <- names(df)[13:16]; metrics
df_summ <- df %>%
  group_by_at(conditions) %>%
  summarise_at(metrics, list(mean=mean, sd=sd)) %>%
  mutate(hidden_layers = ifelse(u2 == "na", 1, 2))

ggplot(df_summ, aes(patience, mcc_mean, color=n_remove, shape=factor(d1))) +
  geom_point(alpha=0.5, size=3) +
  facet_grid(cols=vars(epoch), rows=vars(hidden_layers))

df_summ2 <- df_summ %>%
  filter(batch == 4096 & epoch == 75, patience %in% c(20, 30))

ggplot(df_summ2, 
       aes(patience, mcc_mean, color=factor(hidden_layers), 
           shape=factor(clip))) +
  geom_point(alpha=0.5, size=4, position = position_dodge(5)) +
  facet_grid(cols=vars(epoch))

# focus on single-hidden-layer NN
df_summ3 <- df_summ2 %>%
  filter(clip == 3 & hidden_layers == 1)

ggplot(df_summ3, aes(n_remove, mcc_mean, color=d1, shape=factor(scaler))) +
  geom_point(position = position_dodge(5), size=3, alpha=0.5) +
  facet_grid(cols=vars(lr))

df_summ4 <- df_summ3 %>%
  filter(scaler == "StandardScaler" & d1 == 0.5 & u1 == 256)

ggplot(df_summ4, aes(n_remove, mcc_mean, color=lr)) +
  geom_point() +
  geom_errorbar(aes(ymin=mcc_mean - mcc_sd, ymax=mcc_mean+mcc_sd))
# manually removing features does not seem to improve downstream performance

# conclusion: SS - clip(3) - h=256 (Dropout(0.5))
# LR w/ Adam optimizer 0.01-0.001

# finer grid search on LR
DIR <- "NN/1__Opt"
df3 <- read_results(DIR)

# aggregate folds
df3 %>%
  select(-X, -i) %>%
  filter(n_remove == 0) %>%
  group_by(n_remove, scaler, clip, u1, d1, batch, epoch, patience, lr) %>%
  summarize(n=n(), mcc=mean(mcc))
  # NN architecture: u1=256 (dropout p=0.5), lr=0.001 (~75 epochs w/
  # early stopping)

#' CSL & resampling
#' ----------------------------------------------------------------------------
#' Visual exploration of output of the 2-NN-CSL-(no)-resampling notebooks,
#' i.e. tuning of the class_weight hyperparameter and/or resampling strategies

# cost-sensitive learning: no resampling
DIR <- "NN/2__CSL/no_resampling"
df <- read_results(DIR)

df_summ <- df %>%
  select(-X) %>%
  group_by(seed, upsampler, up_ratio, cw_0) %>%
  summarize(n=n(), mcc=mean(mcc))

table(df_summ$n)

# general
ggplot(df_summ, aes(cw_0, mcc))+
  geom_point() +
  geom_smooth()

df_summ %>% group_by(upsampler, up_ratio, cw_0) %>%
  summarize(mcc = mean(mcc),
            n = n()) %>% # avg over 3 random seeds
  arrange(desc(mcc))
# optimal: {0:0.15, 1: 0.85} w/ validation fold MCC of ~0.56

# cost-sensitive learning: resampling
DIR <- "NN/2__CSL/resampling"
df <- read_results(DIR) %>%
  select(-X) %>%
  group_by(seed, upsampler, up_ratio, cw_0) %>%
  summarize(mcc = mean(mcc), # average validation fold mcc per seed (1-4)
            n = n())

table(df$up_ratio)
table(df$n)

ggplot(df, aes(factor(cw_0), mcc)) +
  geom_boxplot()
# => SVMSMOTE(0.2)

#' Evaluation of the NN classifiers
#' ----------------------------------------------------------------------------
#' Plot PR curves & confusion matrices: output from 3-NN-Evaluation notebook

# load data
DIR <- "NN/3__Eval"
files <- file.path(DIR, dir(DIR))

# precision-recall curves
pr <- read.csv(files[1])
# summary measures
eval <- read.csv(files[2])

# clean/tidy/reshape
eval_long <- eval %>%
  select(-X) %>%
  pivot_longer(cols = -model, names_to = "metric", values_to = "value")

eval_ <- eval %>%
  mutate(
    resampling = ifelse(
      str_starts(model, "Resampling"), "RS", "No RS"),
    CSL = ifelse(
      str_ends(model, "No CSL"), "No CSL", "CSL")) %>%
  filter(model != "NN (no PP)")

pr_ <- pr %>%
  mutate(
    resampling = ifelse(
      str_starts(model, "Resampling"), "RS", "No RS"),
    CSL = ifelse(
      str_ends(model, "No CSL"), "No CSL", "CSL")) %>%
  filter(model != "NN (no PP)")

# reference: no preprocessing or CSL
noPP <- pr %>%
  filter(model == "NN (no PP)")

table(pr$model)

# plot PR curves
f1 <- ggplot(pr_, aes(recall, precision)) +
  geom_path(size=1, alpha=0.8, color="orange") +
  geom_path(data=noPP, color="skyblue", size=0.5, alpha=0.8) +
  facet_grid(rows=vars(resampling), cols = vars(CSL)) +
  theme(legend.position = "none") +
  xlab("Recall") +
  ylab("Precision") +
  geom_label(
    data=eval_, 
    mapping=aes(
      x=0.75, y=0.9, 
      label = paste("MCC:", round(mcc, 2))),
    size=3,
    alpha=0.75); f1

# confusion matrices
conf <- eval_long %>%
  filter(metric %in% c('tp', "fp", "fn", "tn")) %>%
  rename(count = value) %>%
  filter(model != "NN (no PP)") %>%
  mutate(true_class = ifelse(metric %in% c("tp", "fn"), "active", "inactive"),
         pred_class = ifelse(metric %in% c("fp", "tp"), "active", "inactive"),
         resampling = ifelse(
           str_starts(model, "Resampling"), "RS", "No RS"),
         CSL = ifelse(str_ends(model, "No CSL"), "No CSL", "CSL"))

conf <- conf %>%
  left_join(
    conf %>%
      group_by(model, true_class) %>%
      summarize(true_tot = sum(count))
  ) %>%
  mutate(prop = count / true_tot)

f2 <- ggplot(conf, aes(true_class, pred_class, fill = prop, label=count)) +
  geom_tile() +
  geom_label(fill = "white", size=3, alpha = 0.75) +
  xlab("True class label") +
  ylab("Predicted class label") +
  scale_x_discrete(limits=c("inactive", 'active')) +
  scale_y_discrete(limits=c("inactive", 'active')) + 
  facet_grid(rows=vars(resampling), cols=vars(CSL)) +
  labs(fill = "Proportion") +
  theme(legend.position = "none"); f2

evalNN <- plot_grid(
  f1, f2, nrow = 1, labels = c("E", "F"), 
  rel_widths = c(1.3, 1)); evalNN

# single panel with results
plot_grid(evalLRC, evalRFC, evalNN,
          conf_matrix_legend,
          nrow = 4, rel_heights = c(1, 1, 1, 0.2))
