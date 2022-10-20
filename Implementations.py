import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


#features selected from Feature importance method using Random Forest

# Importing the dataset
Feature =pd.read_pickle("FMat.pkl")
Features=Feature[['BC_Actd_', 'BC_Vctd_', 'PR_Vctd_', 'Auth_Vctd_', 'Hub_Vctd_', 'OUT_Vctd_', 'IN_Vctd_', 'Auth_Pctd_', 'IN_Pctd_', 'OUT_Actd_', 'CC_Pctd_', 'PR_Actd_', 'Hub_Pctg_', 'Auth_Actd_', 'OUT_Pctd_', 'CC_Actg_', 'ACC_Pctd_', 'PR_Vctg_', 'Hub_Vctg_', 'IN_Actd_', 'ACC_Vctd_', 'CC_Vctd_', 'Hub_Actd_']]


y = Features["y"].values

# Encoding categorical data

labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y)

#Define x and normalize values
X =Features.drop('y',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

log_regression = LogisticRegression(solver='lbfgs', max_iter=1000)
log_regression.fit(X_train,y_train)
y_pred_log = log_regression.predict(X_test)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_test)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
y_pred_dtree = dtree.predict(X_test)

clf = RandomForestClassifier(max_depth=2,n_estimators=150, random_state=0)
clf.fit(X_train, y_train)
y_pred_rfc = clf.predict(X_test)



d_train = lgb.Dataset(X_train, label=y_train)

lgbm_params = {'learning_rate':0.05, 'boosting_type':'gbdt',    #Try dart for better accuracy
              'objective':'binary',
              'metric':['auc', 'binary_logloss'],
              'num_leaves':100,
              'max_depth':10}

clf = lgb.train(lgbm_params, d_train, 50) 

y_pred_lgbm=clf.predict(X_test)

#convert into binary values 0/1 for classification
for i in range(0, X_test.shape[0]):
    if y_pred_lgbm[i]>=.5:       # setting threshold to .5
       y_pred_lgbm[i]=1
    else:  
       y_pred_lgbm[i]=0      
       

# cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
# sns.heatmap(cm_lgbm, annot=True) 


dtrain=xgb.DMatrix(X_train,label=y_train)


#setting parameters for xgboost
parameters={'max_depth':10, 
            'objective':'binary:logistic',
            'eval_metric':'auc',
            'learning_rate':.05}

xg=xgb.train(parameters, dtrain, 50)

#now predicting the model on the test set 
dtest=xgb.DMatrix(X_test)
y_pred_xgb = xg.predict(dtest) 

#Converting probabilities into 1 or 0  
for i in range(0, X_test.shape[0]): 
    if y_pred_xgb[i]>=.5:       # setting threshold to .5 
       y_pred_xgb[i]=1 
    else: 
       y_pred_xgb[i]=0 

#cm_xgb = confusion_matrix(y_test, y_pred_xgb)
# sns.heatmap(cm_xgb, annot=True)

print("####################      Approach IV       ###############################")
print("####################F1-score of Models################################ ")

print ("f1 score with Knn = ", metrics.f1_score(y_pred_knn,y_test))
print ("f1 score with logistic Regression= ", metrics.f1_score(y_pred_log, y_test))
print ("f1 score with Naive Bayes = ", metrics.f1_score(y_pred_nb,y_test))
print ("f1 score with DecisionTree= ", metrics.f1_score(y_pred_dtree, y_test))
print ("f1 score with Random Forest = ", metrics.f1_score(y_pred_rfc,y_test))
print ("f1 score with LightGBM= ", metrics.f1_score(y_pred_lgbm, y_test))
print ("f1 score with XGB= ", metrics.f1_score(y_pred_xgb, y_test))

print("######################################################################")
print("####################AUC of Models################################ ")

print("AUC score with Knn is: ", roc_auc_score(y_pred_knn,y_test))
print("AUC score with logistic Regression is: ", roc_auc_score(y_pred_log, y_test))
print("AUC score with Naive Bayes is: ", roc_auc_score(y_pred_nb,y_test))
print("AUC score with DecisionTree is: ", roc_auc_score(y_pred_dtree, y_test))
print("AUC score with XGBoost is: ", roc_auc_score(y_pred_xgb, y_test))
print("AUC score with LGBM is: ", roc_auc_score(y_pred_lgbm,y_test))



























































