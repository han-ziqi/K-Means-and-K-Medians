## K-Means and K-Medians

- This project is a implementation about two **cluster algorithm** [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) and [K-Medians](https://en.wikipedia.org/wiki/K-medians_clustering). 

- The **kmeans.py** can cluster words belonging to four files: *animals*, *countries*, *fruits* and *veggies*.
- The standalone .py file including:
  - Implement K-Means and K-Medians clustering algorithm
  - Vary K value from 1-9, then calculate the B-CUBED precision, recall, and F-score for each set of clusters. Then plot the result.
  - Normalise each data object (vector) to unit l2 length before clustering, then re-run the algrithm and plot  B-CUBED precision, recall, and F-score

---

## How to run kmeans.py

For best result, you can clone this repository and open in [PyCharm](https://www.jetbrains.com/pycharm/),  I used [pandas](https://en.wikipedia.org/wiki/Pandas_(software)) to load the data, PLEASE **make sure the four dataset is in same directory** before you run.

### Implement K-Means and K-Medians

You can run the results of implement K-Means and K-Medians clustering algorithm as default settings.
#### You can see this demo for single run



![This is a demo for K-means](https://github.com/han-ziqi/K-Means-and-K-Medians/raw/master/demo/K-Means-2.jpg"K-Means demo")
![This is a demo for K-medians](https://github.com/han-ziqi/K-Means-and-K-Medians/raw/master/demo/K-Medians-2.jpg"K-Medians demo")

### Vary K value, and Normalise each data object (vector) to unit l2 length before clustering

1. First, please comment line 261 and Line 263.

2. Remove block comments symbol from line 264 to line 331

3. Wait the program, then you will see 4 plots, they They correspond to K-Means and K-Medians about:

   - Vary K value from 1-9, then calculate the B-CUBED precision, recall, and F-score for each set of clusters. Then plot the result.

   - Normalise each data object (vector) to unit l2 length before clustering, then re-run the algrithm and plot  B-CUBED precision, recall, and F-score
![This is a demo for 4 diargams](https://github.com/han-ziqi/K-Means-and-K-Medians/raw/master/demo/Compare.jpeg)
