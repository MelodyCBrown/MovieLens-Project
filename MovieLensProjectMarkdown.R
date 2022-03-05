## ----setup, include= FALSE------------------------------------------------------------------------------
knitr::opts_chunk$set(include = FALSE,
                      warning= TRUE,
                      error=TRUE,
                      message= FALSE)


## ----librariesoptions, include=FALSE--------------------------------------------------------------------
library(tidyverse)
library(dplyr)
library(caret)
library(data.table)
library(tinytex)
options(dplyr.summarise.inform= FALSE)
options(digits=4)
options(pillar.sigfig = 4)


## ----Setup, include=TRUE--------------------------------------------------------------------------------
## Read in data file
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

## Rename and reformat Columns
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

## Convert to a data frame and join new columns
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                            title = as.character(title),
                                            genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

## Set Validation set as 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

#Confirm userId and movieId in validation set are also in edx set
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



## ----datahead, include=TRUE-----------------------------------------------------------------------------
# Glimpse of data set
glimpse(edx)

# Range of ratings
descriptives <-edx %>% 
  summarize(n_users = n_distinct(userId), 
            n_movies= n_distinct(movieId), 
            mean_rating = mean(rating),
            min_rating = min(rating),
            max_rating = max(rating))

knitr::kable(descriptives, caption="Descriptives Summary Table")


## ----Freq Ratings Sample, include=TRUE------------------------------------------------------------------
#Calculate frequency of ratings by title
topratingsdf <-edx %>% group_by(title) %>% 
  summarize(count = n()) %>%
  top_n(25) %>%
  arrange(desc(count)) 

#Create table of top 25 rated movie titles by count
trs <-head(topratingsdf, 25)
knitr::kable(trs, caption="Top 25 Rated Movie Titles by Count")


## ---- freq exploration, include= TRUE-------------------------------------------------------------------
##Convert frequencies to level and then plot to see distribution ratings frequency in a visually meaningful way
topratingsdf <- edx %>% 
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange (desc(count)) %>%
  as.data.frame()

#Convert count to numeric
topratingsdf$count <- as.numeric(unlist(topratingsdf$count))

#Add Ratings Frequency Levels
ratingsfreqdf <-topratingsdf %>% mutate(RatingsFreq = cut(count, breaks=c(0, 5000, 10000, 15000, 20000, 25000, 30000, 40000), labels= c("Under_5k", "5k_10k", "10k_15k", "15k_20k", "20k_25k", "25k_30k", "Over_30k")))

#Percentage of Ratings by Level
ratingsfreqtbl <- prop.table(table(ratingsfreqdf$RatingsFreq))* 100

knitr::kable(ratingsfreqtbl, caption="Percentage of Ratings by Level")

#Plot frequency of ratings
ggplot(ratingsfreqdf, aes(RatingsFreq)) +
  labs(title= "Frequency of Ratings Counts", x= "Ratings Frequency Levels", y="Count of Ratings")+
  geom_bar(color="blue", fill="blue")

rm(descriptives, topratingsdf, ratingsfreqtbl, trs)



## ----Genre Sample, include= TRUE------------------------------------------------------------------------
## Remove timestamp and title columns from edx and validation sets to optimize processing time
edx <- within(edx, rm(timestamp, title))
validation <- within(validation, rm(timestamp, title))

#Split genres into distinct rows for analysis and ability to calculate information based on genres
edx <- edx %>% separate_rows(genres, sep = "\\|")
validation <- validation %>% separate_rows(genres, sep = "\\|")

# Unique genres
uniquegenresdf <- edx %>% 
  group_by(genres) %>%
  summarize(count=n()) %>%
  arrange(desc(count)) 
  
knitr::kable(uniquegenresdf, caption="Unique Genres by Count")

#Plot of genres by count
ggplot(uniquegenresdf, aes(reorder(x =genres, -count), y = count)) +
  labs(title = "Counts by Genre", x = "Genres", y = "Counts") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))+
  geom_bar(stat = "identity", color = 'blue', fill= 'blue') +
  theme(axis.text.x = element_text(angle=45, hjust=1.0))

rm(uniquegenresdf)


## ----averages, include=TRUE-----------------------------------------------------------------------------
#Calculate average rating by movie ID
avgrating <- edx %>% 
  group_by(movieId) %>% 
  summarize(avgrating = mean(rating))

#Visualize average rating by movie distribution
ggplot(avgrating, aes(x=avgrating))+ 
  labs(title="Distribution of Average Rating by Movie") +
  geom_histogram(col="white", fill="blue")

#Boxplot of Ratings by genres
grplot <-edx %>% 
  ggplot(aes(reorder(x = genres, -rating, median), y = rating, fill = genres)) + 
  labs(title = "Ratings by Genre", x = "Genres", y = "Ratings") +
  geom_boxplot()+
  theme(axis.text.x = element_text(angle=45, hjust=1.0))
grplot

rm(grplot, avgrating)


## ---- Partition, include= TRUE--------------------------------------------------------------------------
## Set test set as 10% of the edx data set
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
edx_temp <- edx[test_index,]

# Confirm that userId and movieId are in the train and test sets
test_set <- edx_temp %>% 
      semi_join(train_set, by = "movieId") %>%
      semi_join(train_set, by = "userId")

# Add rows removed from the test set into train set
edx_removed <- anti_join(edx_temp, test_set)
train_set <- rbind(train_set, edx_removed)

rm(edx_temp, test_index, edx_removed)



## ---- RMSE, include=TRUE--------------------------------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



## ---- BaseModel, include=TRUE---------------------------------------------------------------------------
## Calculate mu, the average of all ratings
mu <- mean(train_set$rating)

##Create base model
basemodel <- RMSE(test_set$rating, mu)

#Create test results table 
RMSE_Results <- tibble(Method= "Base Model", RMSE = basemodel)

knitr::kable(RMSE_Results, caption ="Model Results Table")


## ---- MovieModel, include=TRUE--------------------------------------------------------------------------
## Calculate average user's rating by movie & calculating b_i
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating-mu))
  
##Check prediction against test set
prediction_bi <- mu + test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

#Calculate RMSE Results
MovieModel <- RMSE(prediction_bi, test_set$rating)

#Create test results table 
RMSE_Results <- tibble(Method= c("Base Model", "Movie Model"), RMSE = c(basemodel, MovieModel))

knitr::kable(RMSE_Results, caption ="Model Results Table")


## ---- UserMovieModel, include=TRUE----------------------------------------------------------------------

## Calculate average user's rating by movie & calculating b_i
user_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId")%>%
  group_by(userId) %>%
  summarize(b_u = mean(rating-mu -b_i))

##Check prediction against test set
prediction_bu <- test_set %>%
  left_join(movie_avgs, by ="movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  summarize(pred= mu+b_i +b_u) %>% 
  pull(pred)

#Calculate RMSE Results
UserMovieModel <- RMSE(prediction_bu, test_set$rating)

#Create test results table 
RMSE_Results <- tibble(Method= c("Base Model", "Movie Model", "Movie + User Model"), RMSE = c(basemodel, MovieModel, UserMovieModel))

knitr::kable(RMSE_Results, caption ="Model Results Table")
                       


## ---- GenreModel, include=TRUE--------------------------------------------------------------------------
## Calculate average user's rating by genre & calculating b_g
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  group_by(userId, genres) %>%
  summarize(b_g = mean(rating-mu-b_i-b_u)) 

##Check prediction against test set
prediction_bg <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by ="userId", "genre") %>%
  left_join(genre_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
  
#Calculate RMSE Results
MovieGenreModel <- RMSE(prediction_bg, test_set$rating)
MovieGenreModel

#Create test results table 
RMSE_Results <- tibble(Method= c("Base Model", "Movie Model", "Movie + User Model", "Movie + User + Genre Model"), RMSE = c(basemodel, MovieModel, UserMovieModel, MovieGenreModel))

knitr::kable(RMSE_Results, caption ="Model Results Table")


## ---- FinalBaseModel, include=TRUE----------------------------------------------------------------------
## Calculate mu, the average of all ratings
edx_mu <- mean(edx$rating)

##Create base model
Finalbasemodel <- RMSE(validation$rating, edx_mu)

#Create test results table 
RMSE_Results_Final <- tibble(Method= "Base Model", RMSE = Finalbasemodel)

knitr::kable(RMSE_Results_Final, caption ="Final Model Results Table")


## ---- FinalMovieModel, include=TRUE---------------------------------------------------------------------
## Calculate average user's rating by movie & calculating b_i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating-edx_mu))
  
##Check prediction against validation set
prediction_bi <- edx_mu + validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

#Calculate RMSE Results
FinalMovieModel <- RMSE(prediction_bi, validation$rating)

#Create test results table 
RMSE_Results_Final <- tibble(Method= c("Base Model", "Movie Model"), RMSE = c(Finalbasemodel, FinalMovieModel))

knitr::kable(RMSE_Results_Final, digits = 4, caption ="Final Model Results Table")


## ---- FinalMovieUserModel, include=TRUE-----------------------------------------------------------------
## Calculate average user's rating by movie & calculating b_i
user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId")%>%
  group_by(userId) %>%
  summarize(b_u = mean(rating-edx_mu -b_i))
  
##Check prediction against validation set
prediction_bu <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId")%>%
  mutate(pred= edx_mu + b_i + b_u) %>%
  pull(pred)

#Calculate RMSE Results
FinalUserMovieModel <- RMSE(prediction_bu, validation$rating)

#Create test results table 
RMSE_Results_Final <- tibble(Method= c("Base Model", "Movie Model", "Movie + User Model"), RMSE = c(Finalbasemodel, FinalMovieModel, FinalUserMovieModel))

knitr::kable(RMSE_Results_Final, caption ="Final Model Results Table")


## ---- FinalModel, include=TRUE, eval=FALSE--------------------------------------------------------------
## ###NOT CALCULATED AS FINAL MODEL DUE TO LACK OF IMPROVEMENT
## 
## ## Calculate average user's rating by genre & calculating b_g
## genre_avgs <- edx %>%
##   left_join(movie_avgs, by="movieId") %>%
##   left_join(user_avgs, by="userId") %>%
##   group_by(userId, genres) %>%
##   summarize(b_g = mean(rating-edx_mu-b_i-b_u))
## 
## 
## ##Check prediction against validation set
## prediction_bg <- validation %>%
##   left_join(movie_avgs, by="movieId") %>%
##   left_join(user_avgs, by ="userId", "genre") %>%
##   left_join(genre_avgs, by = "userId") %>%
##   mutate(pred = edx_mu + b_i + b_u + b_g) %>%
##   pull(pred)
## 
## 
## #Calculate RMSE Results
## FinalMovieGenreModel <- RMSE(prediction_bg, validation$rating)
## 
## #Create test results table
## RMSE_Results_Final <- tibble(Method= c("Base Model", "Movie Model", "Movie + User Model", "Movie + User + Genre Model", RMSE = c(Finalbasemodel, FinalMovieModel, FinalUserMovieModel, FinalMovieGenreModel)))
## 
## knitr::kable(RMSE_Results_Final, caption ="Final Model Results Table")

