read_results <- function(DIR) {
  
  #' returns a tibble with all csv files from the given directory (DIR)
  #' concatened (assuming matching columns)
  
  if(!"dplyr" %in% tolower((.packages()))){
    library("dplyr")
  }
  
  files <- file.path(DIR, dir(DIR))
  files <- files[grep("csv$", files)]
  
  data <- list()
  for (i in seq_along(files)) {
    cat(paste("|--", files[i], "\n"))
    data[[i]] <- read.csv(files[i], stringsAsFactors = F)
  }
   
  cat("|--> MERGED")
  do.call(rbind, data) %>% as_tibble()
  
}

to_long <- function(df) {
  
  #' Returns standard results.csv format (as outputted by the attached
  #' notebooks) into long format by training/validation fold MCC or BA metric.
  #' Input is not checked
  
  df_long_mcc <- df %>%
    pivot_longer(cols=ends_with("mcc"), names_to="fold", values_to="mcc") %>%
    mutate(fold=str_remove(fold, "_mcc")) %>%
    select(-ends_with("balanced_acc"))
  
  df_long_ba <- df %>%
    pivot_longer(cols=ends_with("balanced_acc"), 
                 names_to = "fold", 
                 values_to = "acc") %>%
    mutate(fold = str_remove(fold, "_balanced_acc")) %>%
    select(-ends_with("mcc"))
  
  inner_join(df_long_mcc, df_long_ba)
}

aggregate_folds <- function(df, not=NULL) {
  
  #' aggregates fold of NN optimizatio output: manual averaging
  #' of CV folds per condition that is included in the df.
  #' Aggregates over all columns excl X, seed, fold and metrics
  
  `%notin%` <- Negate(`%in%`)
  names <- names(df)
  
  if (names[1] %in% c("X", "X1")) {
    df <- df %>%
      select(-1)
    names <- names(df)
  }
  
  excl <- c(
    c("fold", "precision", "recall", "auc", "mcc"),
    not)
  conditions <- names[names %notin% excl]
  metrics <- names[names %in% c("precision", "recall", "auc", "mcc")]
  
  out <- df %>%
    group_by_at(conditions) %>%
    #summarize(n=n())
    summarise_at(metrics, list(mean=mean, sd=sd))

  out
}
