rm(list=ls(all=TRUE))

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(recommenderlab)

##=================================================
## MY CHOSEN DATASET:
## Data from here https://www.kaggle.com/CooperUnion/anime-recommendations-database

anime_load <- tempfile()
download.file("https://drive.google.com/uc?export=download&id=1f-TzvOOsNXlCVAnWPuhLl4PU7qWxDoSV", anime_load)

anime <- read.csv(anime_load)

rating_load <- tempfile()
download.file("https://www.dropbox.com/s/6ok9gkt4onxv8kb/rating.csv?raw=1", rating_load)

rating <- read.csv(rating_load)

rating %>% nrow() ## number of ratings (including implicit)
rating %>% group_by(user_id) %>% summarise(n()) %>% nrow() ## number of users

## Decided to only look at TV shows (the dataset also includes films)
## Filtered out "-1"s which are when a user has watched the show but not rated it 
anime_rating_all <- rating %>% left_join(anime, by="anime_id") %>% filter(rating.x != -1, type == "TV")

##=================================================
## My intention is to use the recommenderlab package
## I was unable to use this for the movielens project as my laptop did not have enough memory
## I will therefore reduce the size of the anime dataset so that the package can be used
##=================================================
## Data visualisation - before reducing the data

all_anime_avg_hist <- anime_rating_all %>% group_by(anime_id) %>% summarise(rating_m = mean(rating.x, na.rm = TRUE)) %>% 
	ggplot(aes(rating_m)) +
	geom_histogram(bins = 200) +
	xlim(0, 10) +
	xlab("Average anime rating") +
	theme_bw()

all_user_avg_hist <- anime_rating_all %>% group_by(user_id) %>% summarise(rating_m = mean(rating.x, na.rm = TRUE)) %>% 
	ggplot(aes(rating_m)) +
	geom_histogram(bins = 200) +
	xlim(0, 10) +
	xlab("Average user rating") +
	theme_bw()

##=================================================
## Datasets to be reduced in size so they are manageable for the recommenderlab package

## Filter users and animes to reduce the size of the dataset (so recommenderlab package can be used)
user_list <- anime_rating_all %>% group_by(user_id) %>% summarise(n = n()) %>% filter(n >= 200)    ## only users with 200 or more ratings
anime_list <- anime_rating_all %>% group_by(anime_id) %>% summarise(n = n()) %>% filter(n >= 2000) ## only animes with 2000 or more ratings

anime_rating <- anime_rating_all %>%               ## Dataset ~750,000 reviews
	filter(user_id %in% user_list$user_id & anime_id %in% anime_list$anime_id)

##=================================================
## Data visualisation (for reduced data-set):
## Data is split into test and training sets so that exploration is only performed on the training set

##=================================================
## Splitting into training and test datasets

# Test set will be 10% of the dataset
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = anime_rating$rating.x, times = 1, p = 0.1, list = FALSE)
train <- anime_rating[-test_index,]
temp <- anime_rating[test_index,]

# Make sure user and anime ids in the test set set are also in edx set
test <-  temp %>% 
	semi_join(train, by = "anime_id") %>%
	semi_join(train, by = "user_id")

removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(test_index, temp, removed) ## to reduce memory

##=================================================
## Data visualisation on the train part of the reduced dataset

anime_avg_hist <- train %>% group_by(anime_id) %>% summarise(rating_m = mean(rating.x, na.rm = TRUE)) %>% 
	ggplot(aes(rating_m)) +
	geom_histogram(bins = 200) +
	xlim(0, 10) +
	xlab("Average anime rating") +
	theme_bw()

user_avg_hist <- train %>% group_by(user_id) %>% summarise(rating_m = mean(rating.x, na.rm = TRUE)) %>% 
	ggplot(aes(rating_m)) +
	geom_histogram(bins = 200) +
	xlim(0, 10) +
	xlab("Average user rating") +
	theme_bw()

##=================================================
## PREPARING DATA FOR RECOMMENDERLAB PACKAGE
##=================================================
## The recommenderlab packages allows training, test and validation within the same analysis
## To use the Recommenderlab package we need a RealRatingMatrix
## This is a matrix of users vs. animes filled with ratings

anime_matrix <- anime_rating %>% 
	select(user_id, anime_id, rating.x) %>%
	pivot_wider(names_from = "anime_id", values_from = "rating.x") %>%
	as.matrix(.)

## View(anime_matrix) ## can see in matrix form using the View command

rownames(anime_matrix)<- anime_matrix[,1] ## makes the row names the user_ids
anime_matrix <- anime_matrix[,-1] ## Once the rownames are listed, remove the user_id column

anime_matrix_view <- anime_matrix[1:10, 1:10]

anime_matrix <- sweep(anime_matrix, 2, colMeans(anime_matrix, na.rm=TRUE)) ## this sweeps out the summary statistic (i.e. removes the average rating of anime across users)
anime_matrix <- sweep(anime_matrix, 1, rowMeans(anime_matrix, na.rm=TRUE)) ## this sweeps out the average user rating across films

anime_matrix_view_z <- round(anime_matrix[1:10, 1:10], 2)

anime_rrm <- as(anime_matrix, "realRatingMatrix")

isS4(anime_rrm) ## to check it is in the correct format

## The data is now in a format that can be used by the recommenderlab package

##==============================================
## Explore the data:

dim(anime_rrm@data) ## look at the dimensions of the real ratings matrix
## 4687 users, 633 animes

sum(table(as.vector(anime_rrm@data))) ## total number of ratings - should match the product of the dimensions:
## 2966871
## 4687*633 = 2966871 They match

##==============================================
#### RECOMMENDERLAB PACKAGE
##==============================================

## Recommender is built using evaluationScheme()
## This package splits the dataset into training, test and validation sets

set.seed(1, sample.kind="Rounding")
anime_eval <- evaluationScheme(data = anime_rrm, 
																method = "cross-validation", 
																k = 10,           ## number of cross-validation folds
																given = -15,       ## this is the number of ratings to withhold from the test set while validating
																goodRating = 7.5) ## chosen a rating over 7.5 as good

## I have chosen a given of -1 which means that all but 1 ratings are withheld to test the model
## Models that perform well with lower values of given are better as user ratings are often sparse

## Ths evaluation creates 3 datasets - train and test sets
## Within the test set the function creates known and unknown data sets
	## known - ratings specified by given
  ## unknown - remaining ratings which are used to validate the known predictions

anime_eval_train   <- getData(anime_eval, "train")
anime_eval_known   <- getData(anime_eval, "known")
anime_eval_unknown <- getData(anime_eval, "unknown")

## Now we wish to consider the recommender algorithms in the recommenderlab package
## These are:
  ## IBCF, UBCF, POPULAR, RANDOM, SVD, SVDF

## Let us go through these in order and compare the outputs
## We will start with RANDOM as this is the baseline approach (no clustering)

## RANDOM
anime_rand <- 
	anime_eval_train %>%
	Recommender(method = "RANDOM") 

anime_rand_eval <- anime_rand %>% 
	predict(anime_eval_known, type = "ratings") %>% 
	calcPredictionAccuracy(anime_eval_unknown)

## IBCF
anime_ibcf <- 
	anime_eval_train %>%
	Recommender(method = "IBCF") 

anime_ibcf_eval <- anime_ibcf %>% 
	predict(anime_eval_known, type = "ratings") %>% 
	calcPredictionAccuracy(anime_eval_unknown)

## UBCF
anime_ubcf <- 
	anime_eval_train %>%
	Recommender(method = "UBCF") 

anime_ubcf_eval <- anime_ubcf %>% 
	predict(anime_eval_known, type = "ratings") %>% 
	calcPredictionAccuracy(anime_eval_unknown)

## POPULAR
anime_pop <- 
	anime_eval_train %>%
	Recommender(method = "POPULAR") 

anime_pop_eval <- anime_pop %>% 
	predict(anime_eval_known, type = "ratings") %>% 
	calcPredictionAccuracy(anime_eval_unknown)

## SVD
anime_svd <- 
	anime_eval_train %>%
	Recommender(method = "SVD") 

anime_svd_eval <- anime_svd %>% 
	predict(anime_eval_known, type = "ratings") %>% 
	calcPredictionAccuracy(anime_eval_unknown)

## SVDF
anime_svdf <- 
	anime_eval_train %>%
	Recommender(method = "SVDF") 

anime_svdf_eval <- anime_svdf %>% 
	predict(anime_eval_known, type = "ratings") %>% 
	calcPredictionAccuracy(anime_eval_unknown)

## Now provide a comparison table for use in the report
comparison <- transpose(data.frame(RANDOM = anime_rand_eval,
												 IBCF = anime_ibcf_eval,
												 UBCF = anime_ubcf_eval,
												 POPULAR = anime_pop_eval,
												 SVD = anime_svd_eval,
												 SVDF = anime_svdf_eval)) %>% 
												rename(RMSE = V1, MSE = V2, MAE = V3) %>%
	mutate(Algorithms = c("RANDOM", "IBCF", "UBCF", "POPULAR", "SVD", "SVDF"))

comparison <- comparison[, c(4, 1, 2, 3)]

RANDOM <- comparison[1, ]
IBCF <- comparison[1:2, ]
UBCF <- comparison[1:3, ]
POPULAR <- comparison[1:4, ]
SVD <- comparison[1:5, ]
SVDF <- comparison[1:6, ]

## Plot this:

algorithm_comparison <- comparison %>% ggplot(aes(RMSE, reorder(Algorithms, RMSE), fill = Algorithms)) +
	geom_col(show.legend = FALSE) +
	ylab("Recommenderlab algorithms") +
	theme_bw() 

## SVDF > SVD > POPULAR > UBCF > IBCF > RANDOM

