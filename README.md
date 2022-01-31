# ML Algorithms Web App
Machine Learning Web Application to try on different Dataset and different Algorithms. With Many addition features to be analysed and visualized

ML Algorithm is a Web App created using `Python Streamlit`.

The main purpose of this project is to play around with Machine Learning Algorithm and `Hyperparameter`
This web contains 4 demo datasets, and performs `EDA` using `Pandas`, `Matplotlib` and `Seaborn` modules.

Once EDA is performed, the user can choose a Machine Learning Classification Algorithm from the sidebar.
The Classification Algorithms Used Are:
```
Decision Tree, KNN, LogisticRegression, Naive Bayes And Random Forest
```



## Steps To Deploy On Heroku

Once you have pushed your Repository On GitHub, follow these steps to deploy on Heroku.
| :warning: WARNING                                     |
|:------------------------------------------------------|
| requirements.txt, setup.sh, Procfile is must          |

```py
heroku login
```

**heroku: Press any key to open up the browser to login or q to exit:**
Login to your `Heroku` account

```py
heroku create mlalgoapp
```

You can provide anyname in place of mlalgoapp

The last step is to `push` our files to `Heroku`

```py
git push heroku master
```

Check out the deployed site and play around with Machine Learning Algorithms:
[ML Algo Web App](https://mlalgoapp.herokuapp.com/)
