# FakeReviewDetectionSystem


                                           FAKE CONSUMER REVIEW DETECTION

 
 
 ABSTRACT :
 Consumers’ reviews on ecommerce websites, online services, ratings and experience stories
are useful for the user as well as the vendor. The reviewer can increasetheir brand’s loyalty and 
help other customers understand their experience with the product. Similarly reviews help the
vendors gain more profiles by increasing their sale ofproducts, if consumers leave positive
feedback on their product review. Butunfortunately, these review mechanisms can be misused 
by vendors.
 For example, one may create fake positive reviews to promote brand’s reputationor try to
demote competitor’s products by leaving fake negative reviews on their product.Existing 
solutions with supervised include application of different machine learning algorithms and 
different tools like Weka.
 Unlike the existing work, instead of using a constrained dataset I chose to have awide variety
of vocabulary to work on such as different subjects of datasets combined asone big data set. 
Sentiment analysis has been incorporated based on emojis and text content in the reviews. Fake
reviews are detected and categorized. The testing results areobtained through the application of
Naïve Bayes, Linear SVC, Support Vector Machineand Random forest algorithms. The 
implemented (proposed) solution is to classify these reviews into fake or genuine. The highest 
accuracy is obtained by using Naïve Bayes by including sentimentclassifier.
KEYWORDS: Machine learning, fake, reviews, Logistic Regression….


Introduction :
Everyone can freely express his/her views and opinions anonymously and without the fear of 
consequences. Social media and online posting have made it even easier to post confidently and 
openly. These opinions have both pros and cons while providing the right feedback to reach the 
right person which can help fix the issue and sometimes a con when these get manipulated 
These opinions are regarded as valuable. This allows people with malicious intentions to easily 
make the system to give people the impression of genuineness and post opinions to promote 
their own product or to discredit the competitor products and services, without revealing 
identity of themselves or the organization they work for. Such people are called opinion 
spammers and these activities can be termed as opinion spamming.
There are few different types of opinion spamming. One type is giving positive opinions to 
some products with intention to promote giving untrue or negative reviews to products to 
damage their reputation. Second type consists of advertisements with no opinions on product. 
There is lot of research work done in field of sentiment analysis and created models while using 
different sentiment analysis on data from various sources, but the primary focus is on the 
algorithms and not on actual fake review detection.
The use of Opinion Mining, a type of language processing to track the emotion and thought 
process of the people or users about a product which can in turn help research work. Opinion 
mining, which is also called sentiment analysis, involves building a system to collect and 
examine opinions about the product made in social media posts, comments, online product and 
service reviews or even tweets. Automated opinion mining uses machine learning, a component 
of artificial intelligence. An opinion mining system can be built using a software that can 
extract knowledge from dataset and incorporate some other data to improve its performance.
One of the biggest applications of opinion mining is in the online and e-commerce reviews of 
consumer products, feedback and services. As these opinions are so helpful for both the user as 
well as the seller the e-commerce web sites suggest their customers to leave a feedback and 
review about their product or service they purchased. These reviews provide valuable 
information that is used by potential customers to know the opinions of previous or current 
users before they decide to purchase that product from that seller. Similarly, the seller or 
service providers use this information to identify any defects or problems users face with their 
products and to understand the competitive information to know the difference about their 
similar competitors’ products.


1.Related work :
Most of the research work on spam review detection falls into two categories . One group of 
research focus on only the content of the reviews. On the other hand, other group of researchers concentrate on reviewers behavior instead of review content .But a combination of 
both approaches gives the best result.
• In Jindal ,et al .claimed they are the first to attempt to study review spam and spam 
detection. They collected 2.14 million review for amazon for their research work. They 
found a large number of duplicate and near-duplicate reviews written by the same 
reviewers on different products or by different reviewers on the same products or 
different products. They proposed to perform spam detection based on duplicate finding 
and classification. They used logistic regression to learn a predictive model. Using10-
fold cross-validation on the data they got average area under the ROC curve(AUC) 
value of 78%.
• A method was proposed by E.I Elmurngi and A. Gherbi [1] usingan open source
software tool called ‘Weka tool’ to implement machine learning algorithms using 
sentiment analysis to classify fair and unfair reviews from amazon reviews based on 
three different categories positive, negative and neutral words. In this research work, the 
spam reviews are identified by only including the helpfulness votes voted by the 
customers along with the rating deviation are considered which limits the overall
performance of the system. Also, as per the researcher’s observations andexperimental 
results, the existing system uses Naive Bayes classifier for spam and non- spam 
classification where the accuracy is quite low which may not provide accurate results
for the user.
• Initially N. O’Brien [4] and J. C. S. Reis, A. Correia, F. Murai, A. Veloso, and F.
Benevenuto [5] have proposed solutions that depends only on the features used in the data
set with the use of different machine learning algorithms in detecting fake news on social
media. Though different machine learning algorithms the approach lacks in showing how
accurate the results are.
• B. Wagh,J.V.Shinde, P.A.Kale [6] worked on twitter to analyze the tweets postedby users 
using sentiment analysis to classify twitter tweets into positive and negative. They made
use of K-Nearest Neighbor as a strategy to allot them sentiment labels by training
and testing the set using feature vectors. But the applicability of their approachto other 
type of data has not been validated
• In, Li, et al. manually labeled nearly 6000 reviews. They collected a dataset from 
Epinions website. They employed ten college students for tagging all the review 
.Students were first instructed to read books and articles about how spam review looks 
like then they were asked to label those reviews. They first used supervised learning 
algorithm and analyze the effectiveness of different features in review spam 
identification .They also used a two-view semi-supervised methodology to exploit a 
large amount of unlabeled data. The experiment results show that two-view co-training 
algorithms can achieve better results than the single-view algorithm.
• In Luca, et al. worked on restaurant reviews that are identified by Yelp’s filteringalgorithm as suspicious or fake. They found that nearly one out of five reviews is 
marked as fake by Yelp’s algorithm. These reviews tend to be more extreme than other 
reviews and are written by reviewers with less established reputation. Moreover, their 
finding suggests that economic incentives factor heavily into the decision to commit 
fraud. Organizations are more likely to game the system when they are facing increased 
competition and when they have poor or less established reputaions.

2.Proposed work :
To solve the major problem faced by online websites due to opinion spamming, this project 
proposes to identify any such spammed fake reviews by classifying them into fake and genuine. 
The method attempts to classify the reviews obtained from freely available datasets from 
various sources and categories including service based, product based, customer feedback, 
experience based and the crawled Amazon dataset with a greater accuracy using Naïve Bayes 
[7], Linear SVC, SVM, Random forest, Decision Trees algorithm. A classifier is built based on 
the identified features. And those features are assigned a probability factor or a weight 
depending on the classified training sets. This is a supervised learning technique applying 
different Machine learning algorithms to detect the fake or genuine reviews,the problem is 
solved in the following six step:

Data collection : 
 Consumer review data collection-Raw data review was collected from different sources, such 
as Amazon ,websites for booking air lines, Hotels and Restaurant, Caarguns,etc,..review. Doing 
so was to increase diversity of the review data. A data of more than 1000 was created . But 
here we are using a less number of review.

Data Preprocess :
Processing and refining the data by removal of irrelevant and redundant information as well as 
noisy and unreliable data from the review dataset.
Step 1 : Sentence tokenization
The entire review is given as input and it is tokenized into sentences using NLTK package.
Step 2 : Removal of punctuation marks
Punctuation marks used at the starting and ending of the reviews are removed along with 
additional white spaces.
Step 3 : Word Tokenization
Each individual review is tokenized into words and stored in a list for easier retrieval. Step 4 
Removal of stop words
Affixes are removed from the stem. For example, the stem of "cooking" is "cook", and the 
stemming algorithm knows that the "ing" suffix can be removed.

Feature Extraction :
The preprocessed data is converted into a set of features by applying certain parameters. The 
following features are extracted:
Normalized length of the review-Fake reviews tend to be of smaller length. Reviewer ID- A
reviewer posting multiple reviews with the same Reviewer ID.
Rating-Fake reviews in most scenarios have 5 out of 5 stars to entice the customer or have the 
lowest rating for the competitive products thus it plays an important role in fake detection.
Verified Purchase-Purchase reviews that are fake have lesser chance of it being verified 
purchase than genuine reviews .Thus these combination of features are selected for identifying 
the fake reviews. This in turn improves the performance of the prediction models.

Sentiment Analysis :
Classifying the reviews according to their emotion factor or sentiments being positive, negative
or neutral. It includes predicting the reviews being positive or negativeaccording to the words 
used in the text, ratings given to the review and so on. Related research [8] shows that fake 
reviews has stronger positive or negative emotions than true reviews. The reasons are that, fake 
reviews are used to affect people opinion, and it is more significant to convey opinions than to 
plainly describe the facts. The Subjective vs Objective ratio matters: Advertisers post fake 
reviews with more objective information, giving more emotions such as how happy it made 
them than conveying how the product is or what it does. Positive sentiment vs negative 
sentiment: The sentiment of the review is analyzed which in turn help in making the decision of 
it being a fake or genuine review.

Fake Review Detection :
Classification assigns items in a collection to target categories or classes. The goal of 
classification is to accurately predict the target class for each case in the data. Each data in the 
review file is assigned a weight and depending upon which it is classified into respective 
classes - Fake and Genuine.

SKlearn Based Classifiers :
The Sklearn based classifiers were also used for classification and compared 
which algorithm to get better and accurate results.
a. Multinomial Naïve Bayes: Naive Bayes classifier [7] is used in natural language
processing (NLP) problems by predicting the tag of text, calculate probability of each tag
of a text and then output choose the highest one.
b.LinearSVC: This classifier classifies data by providing the best fit hyper plane that
can be used to divide the data into categories SVC: Different studies have shown If you 
use the default kernel in SVC (), the RadialBasis Function (rbf) kernel, then you probably 
used a more nonlinear decision boundaryon the case of the dataset, this will vastly 
outperform a linear decision boundary
c. Random Forest: This algorithm has also been used for classifying which is provided
by sklearn library by creating multiple decision trees set randomly on subset of training
data.

EXPERIMENT AND RESULT :
Steps to Implement :
1. Import the modules and the libraries. For this project, we are 
importing the libraries numpy, pandas, and sklearn and metrics.
2. We are reading our dataset. And we are printing our dataset
3. We are dropping an unused column that is unnamed.
4. We are printing the head of the dataset.
5. We dropping all the null value rows in the dataset.
6. We are defining a function which will convert the text taken into 
input and will remove all the punctuation and the will convert into 
small letters. Then, we are converting our dataset into training and 
testing dataset.
7. We are defining our Pipeline and we are passing our function and our 
model which is Random Forest Classifier into this pipeline.
8. We are fitting our training values into the pipeline.
9. We are passing the testing dataset and we are predicting the accuracy 
of the model.
10. We are doing the same task as above. But this time our model is 
SVC.
11. We are fitting our training values into the pipeline.
12. We are passing the testing dataset and we are predicting the 
accuracy of the model.
13. The predicted accuracy of the model is 86.57% .

Conclusion :
The fake review detection is designed for filtering the fake reviews. In this research work SVM 
classification provided a better accuracy of classifying than the Naïve Bayes classifier for 
testing dataset. On the other hand, the Naïve Bayes classifier has performed better than other 
algorithms on the training data. Revealing that it can generalize better and predict the fake 
reviews efficiently. This method can be applied over other sampled instances of the dataset. 
The data visualization helped in exploring the dataset and the features identified contributed to 
the accuracy of the classification. The various algorithms used, and their accuracies show how 
each of them have performed based on their accuracy factors.
Also, the approach provides the user with a functionality to recommend the most truthful 
reviews to enable the purchaser to make decisions about the product. Various factors such as 
adding new vectors like ratings, emojis, verified purchase have affected the accuracy of 
classifying the data better.

usages:
Individual consumers: A consumer can also compare the summaries with competing products 
before taking a decision without missing out on any other better products available in the 
market.
Businesses/Sellers: Opinion mining helps the sellers to reach their audience and understand 
their perception about the product as well as the competitors. Such reviews
also help the sellers to understand the issues or defects so that they can improve later versions 
of their product. In today’s generation this way of encouraging the consumers to write a review 
about a product has become a good strategy for marketing their product through real audience’s 
voice. Such precious information has been spammed and manipulated. Out of many researches 
one fascinating research was done to identify the deceptive opinion spam [3].

References :
1. E. I. Elmurngi and A.Gherbi, “Unfair Reviews Detection on Amazon Reviews using
Sentiment Analysis with Supervised Learning Techniques,” Journal of Computer
Science, vol. 14, no. 5, pp. 714–726, June 2018.
2. J. Leskovec, “WebData Amazon reviews,” [Online]. Available:
http://snap.stanford.edu/data/web-Amazon-links.html [Accessed: October 2018].
3. J. Li, M. Ott, C. Cardie and E. Hovy, “Towards a General Rule for Identifying Deceptive
Opinion Spam,” in Proceedings of the 52nd Annual Meeting of the Association for
Computational Linguistics, Baltimore, MD, USA, vol. 1, no. 11, pp. 1566-1576,
November 2014.
4. N. O’Brien, “Machine Learning for Detection of Fake News,” [Online]. Available:
https://dspace.mit.edu/bitstream/handle/1721.1/119727/1078649610-MIT.pdf
[Accessed: November 2018].
5. J. C. S. Reis, A. Correia, F. Murai, A. Veloso, and F. Benevenuto, “Supervised Learning
for Fake News Detection,” IEEE Intelligent Systems, vol. 34, no. 2, pp. 76-81, May 2019.
6. B. Wagh, J. V. Shinde and P. A. Kale, “A Twitter Sentiment Analysis Using NLTK and
Machine Learning Techniques,” International Journal of Emerging Research in
Management and Technology, vol. 6, no. 12, pp. 37-44, December 2017.
7. A. McCallum and K. Nigam, “A Comparison of Event Models for Naive Bayes Text
Classification,” in Proceedings of AAAI-98 Workshop on Learning for Text
Categorization, Pittsburgh, PA, USA, vol. 752, no. 1, pp. 41-48, July 1998.
8. B. Liu and M. Hu, “Opinion Mining, Sentiment Analysis and Opinion Spam Detection,”
[Online]. Available: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
[Accessed: January 2019].
9. C. Hill, “10 Secrets to Uncovering which Online Reviews are Fake,” [Online]. Available:
https://www.marketwatch.com/story/10-secrets-to-uncovering-which-online-reviewsare-fake-2018-09-21 [Accessed: March 2019].
10. J. Novak, “List archive Emojis,” [Online]. Available: https://li.st/jesseno/positivenegative-and-neutral-emojis-6EGfnd2QhBsa3t6Gp0FRP9 [Accessed: June 2019].
11. P. K. Novak, J. Smailović, B. Sluban and I. Mozeti, “Sentiment of Emojis,” Journal of
Computation and Language, vol.10, no. 12, pp. 1-4, December 2015.
12. P. K. Novak, “Emoji Sentiment Ranking,” [Online]. Available:
http://kt.ijs.si/data/Emoji_sentiment_ranking/ [Accessed: July 2019]


