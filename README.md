# Entropy Analysis of Recipe Instructions Across Cuisines and Complexity Levels

**Project Goal**  
Apply information-theoretic measures (unigram/bigram entropy, multi-gram entropy, mutual information, and Zipf’s law) to a large recipe dataset, revealing how textual diversity and complexity vary with cuisine type, recipe length, preparation time, and ingredient count. Analyses include tokenization of the `steps`, classification into short/medium/long categories, and plotting rank-frequency distributions for Zipf’s law.

**Dataset**  
- I use `RAW_recipes.csv` from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).  
- Contains ~230k+ recipes with fields like `steps`, `tags`, `minutes`, `n_ingredients`, etc.
