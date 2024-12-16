##Cross-Domain Recommendation Systems Hackathon
###Overview
This hackathon focused on developing cross-domain recommendation systems using datasets from Zvuk (a music streaming service) and MegaMarket (an online retail platform). The event saw participation from 100 teams, and our solution secured a prize-winning position.

###Hackathon Objective
Recommendation systems have become integral to our daily lives, helping us find the right products, music, and movies effortlessly, while generating billions in additional revenue for their owners. One of the cutting-edge areas in recommendation systems is cross-domain recommendations, where user actions in one domain (e.g., an online store) are used to suggest relevant content in another domain (e.g., a music streaming service).

The challenge was to develop a recommendation algorithm that outperforms the machine learning algorithms used by Sber's recommendation system. The training data consisted of real user interactions from Sber's ecosystem, including MegaMarket and Zvuk.

###Our Solution
For our solution, we started with a basic ALS model applied separately to the two datasets. We then introduced a hybrid model that combined ALS on the two datasets separately and ALS on the two datasets jointly. The final model incorporated all the above components and included a final rescoring of predictions. We used NDCG@10 as our evaluation metric, and the highest score was achieved with the separate models, specifically 0.0772. This result placed us at the top of the leaderboard for three days and ultimately secured us the second place in the final leaderboard standings.

###Results
Our innovative solution not only met but exceeded the performance of existing algorithms, earning us a top position in the hackathon. The success of our model demonstrates the potential of cross-domain recommendations in providing personalized and relevant suggestions to users.

