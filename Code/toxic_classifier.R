# /*FILE: toxic_classifier.R
#
# Programer: Dave <sudevschiz@gmail.com>
# Edited By : NA
#
# Date: 26 February 2018
#
# Code Objectives : Toxic comments need to be classfied into 6 different type of toxicities
#
# Version: Original
#
# Revision-History:
#
# Comments:
# 26 Feb 2018 : Load the file, make the benchmark submission.
# 06 Mar 2018 : Add more features
# 12 Mar 2018 : Text2Vec

# ###############

#Load necessary libraries
library(data.table)
library(stringdist)
library(ranger)
library(xgboost)
library(coreNLP)
library(tm)
library(tidytext)
library(stringr)

#Set the directory
setwd("~/kaggle/toxic_comments/")

train <- fread("Data/train.csv",stringsAsFactors = F)
test  <- fread("Data/test.csv",stringsAsFactors = F)

#Load the sample submission file too

samp_sub <- fread("Data/sample_submission.csv")


####DATA CLEANING#####

cleanerFunc <- function(text_comments){
  comm_corpus <-  Corpus(VectorSource(text_comments))
  comm_corpus <- tm_map(comm_corpus,removePunctuation)
  comm_corpus <- tm_map(comm_corpus, tolower)
  comm_corpus <- tm_map(comm_corpus, removeWords,stopwords('en'))
  comm_corpus <- tm_map(comm_corpus, removeNumbers)
  comm_corpus <- tm_map(comm_corpus, stripWhitespace)

  
  return(comm_corpus)
}

weighWords <- function(subset){
  clean_corpus <- cleanerFunc(subset)
  
  dtm <- DocumentTermMatrix(clean_corpus,control = list(weighting = weightTfIdf))
  dtm <- removeSparseTerms(dtm,sparse = 0.99)
  mat <- as.matrix(dtm)
  
  imp <- colSums(mat)
  
  return(data.frame(Words = names(imp),Weight = imp))

}

####Naive benchmark submission####

train[,3:8] <- lapply(train[,3:8], as.factor)


#Find weightage of words in each category - 

##Toxic
comm_subset <- train[toxic == 1, comment_text]
toxic_imp <- weighWords(comm_subset)


##Severe_toxic
comm_subset <- train[severe_toxic == 1, comment_text]
sev_toxic_imp <- weighWords(comm_subset)

##Obscene
comm_subset <- train[obscene == 1, comment_text]
obscene_imp <- weighWords(comm_subset)

##Threat
comm_subset <- train[threat == 1, comment_text]
threat_imp <- weighWords(comm_subset)


##Insult
comm_subset <- train[insult == 1, comment_text]
insult_imp <- weighWords(comm_subset)

##Identity hate

comm_subset <- train[identity_hate == 1, comment_text]
identity_hate_imp <- weighWords(comm_subset)


#For each comment in train (and then test) dataset, find the aggregated weights based on words in that sentence
train$toxic_weight <- 0
train$sev_toxic_weight <- 0
train$obscene_weight <- 0
train$threat_weight <- 0
train$insult_weight <- 0
train$identity_hate_weight <- 0


token_list <- strsplit(train$comment_text,split = " ")

train$toxic_weight <- unlist(lapply(token_list, function(x){w <- toxic_imp[toxic_imp$Words %in% x,"Weight"]; return(sum(w))}))
train$sev_toxic_weight <- unlist(lapply(token_list, function(x){w <- sev_toxic_imp[sev_toxic_imp$Words %in% x,"Weight"]; return(sum(w))}))
train$obscene_weight <- unlist(lapply(token_list, function(x){w <- obscene_imp[obscene_imp$Words %in% x,"Weight"]; return(sum(w))}))
train$threat_weight <- unlist(lapply(token_list, function(x){w <- threat_imp[threat_imp$Words %in% x,"Weight"]; return(sum(w))}))
train$insult_weight <- unlist(lapply(token_list, function(x){w <- insult_imp[insult_imp$Words %in% x,"Weight"]; return(sum(w))}))
train$identity_hate_weight <- unlist(lapply(token_list, function(x){w <- identity_hate_imp[identity_hate_imp$Words %in% x,"Weight"]; return(sum(w))}))

train[NA] <- 0

m_toxic <- ranger(data = train[,c(9:14,3)],num.trees = 301,mtry = 2,write.forest = T,probability = T,dependent.variable.name = "toxic")
m_sev_toxic <- ranger(data = train[,c(9:14,4)],num.trees = 301,mtry = 2,write.forest = T,probability = T,dependent.variable.name = "severe_toxic")
m_obscene <- ranger(data = train[,c(9:14,5)],num.trees = 301,mtry = 2,write.forest = T,probability = T,dependent.variable.name = "obscene")
m_threat <- ranger(data = train[,c(9:14,6)],num.trees = 301,mtry = 2,write.forest = T,probability = T,dependent.variable.name = "threat")
m_insult <- ranger(data = train[,c(9:14,7)],num.trees = 301,mtry = 2,write.forest = T,probability = T,dependent.variable.name = "insult")
m_identity_hate <- ranger(data = train[,c(9:14,8)],num.trees = 301,mtry = 2,write.forest = T,probability = T,dependent.variable.name = "identity_hate")


#On the test data 

token_list <- strsplit(test$comment_text,split = " ")

test$toxic_weight <- unlist(lapply(token_list, function(x){w <- toxic_imp[toxic_imp$Words %in% x,"Weight"]; return(sum(w))}))
test$sev_toxic_weight <- unlist(lapply(token_list, function(x){w <- sev_toxic_imp[sev_toxic_imp$Words %in% x,"Weight"]; return(sum(w))}))
test$obscene_weight <- unlist(lapply(token_list, function(x){w <- obscene_imp[obscene_imp$Words %in% x,"Weight"]; return(sum(w))}))
test$threat_weight <- unlist(lapply(token_list, function(x){w <- threat_imp[threat_imp$Words %in% x,"Weight"]; return(sum(w))}))
test$insult_weight <- unlist(lapply(token_list, function(x){w <- insult_imp[insult_imp$Words %in% x,"Weight"]; return(sum(w))}))
test$identity_hate_weight <- unlist(lapply(token_list, function(x){w <- identity_hate_imp[identity_hate_imp$Words %in% x,"Weight"]; return(sum(w))}))

test[NA] <- 0

pred <- predict(m_toxic,data = test[,c(3:8)])
samp_sub$toxic <- pred$predictions[,2]

pred <- predict(m_sev_toxic,data = test[,c(3:8)])
samp_sub$severe_toxic <- pred$predictions[,2]

pred <- predict(m_obscene,data = test[,c(3:8)])
samp_sub$obscene <- pred$predictions[,2]

pred <- predict(m_threat,data = test[,c(3:8)])
samp_sub$threat <- pred$predictions[,2]

pred <- predict(m_insult,data = test[,c(3:8)])
samp_sub$insult <- pred$predictions[,2]

pred <- predict(m_identity_hate,data = test[,c(3:8)])
samp_sub$identity_hate <- pred$predictions[,2]



#Write this samp_sub to a file for submission
write.csv(samp_sub,"Test_submission_1.csv",row.names = F)


###ACTUAL CLASSIFICATION TRIALS - ####

#Merge train and test for cleaning and imputing. A flag needs to be added before that

train$DATASET <- "TRAIN"
test$DATASET <- "TEST"

all_data <- rbind(train,test,fill= T)

table(all_data$DATASET)


####POS tag the comments and store them########

initCoreNLP(mem = "4g")
pos_list <- list(nrow(all_data))

t1 <- Sys.time()
for (i in 1:nrow(all_data)) {
  
  annot <- annotateString(as.character(all_data[i,2]))
  pos_list[[i]] <- annot
  pos_list[[i]]$unq_id <- all_data[i,1]
  
  if(i%%100 == 0){
    print(i)
  }
}

print(Sys.time() - t1)

saveRDS(pos_list,"POS_Tagged_2602.RDS")






####Capital-Ratio feature####

temp_cap <- str_match_all(all_data$comment_text,pattern = "[A-Z]")
temp_cap <- unlist(lapply(temp_cap,length))
temp_nchar <- nchar(all_data$comment_text)

#Feature is created by dividing total number of capital letters in the comment by total number of characters in the comment

all_data$Cap_Ratio <- temp_cap/temp_nchar







####Cleaning####

#Need some thorough cleaning of the data. First look through the data and list down things to do.
#Cleaning could be done after the captial letters feature is extracted from the text
#1. Remove numbers, remove special characters etc.

cleaned_corpus <- cleanerFunc(all_data$comment_text)
cleaned_text <- sapply(cleaned_corpus, content)






