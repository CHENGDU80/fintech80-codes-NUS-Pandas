import pandas as pd
import numpy as np 
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import shap
import json


def preprocessing(df):
    # we have other variable related to time, so remove raw 
    df = df.drop(columns='Time')

    # categorical -> int
    le_vin = LabelEncoder()
    df['VIN'] = le_vin.fit_transform(df['VIN'])
    
    le_ctr = LabelEncoder()
    df['Country'] = le_ctr.fit_transform(df['Country'])

    le_vmk = LabelEncoder()
    df['Vehicle_Make'] = le_vmk.fit_transform(df['Vehicle_Make'])
    
    le_mdl = LabelEncoder()
    df['Vehicle_Model'] = le_mdl.fit_transform(df['Vehicle_Model'])

    le_mdl = LabelEncoder()
    df['Vehicle_Model'] = le_mdl.fit_transform(df['Vehicle_Model'])

    df['Autonomy_Level'] = df['Autonomy_Level'].apply(lambda x: int(x.replace('Level ','')))

    return df

def param_tune(X, y, n_iter=100, n_jobs=8, verbose=1):
    '''
    find best parameter
    '''
    # parameter tuning
    params = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'min_child_weight': (1, 10),
    'max_depth': (3,15),
    'colsample_bytree': (0.1, 1.0, 'uniform'),
    'gamma': (1e-9, 1.0, 'log-uniform'),
    'min_child_weight': (1, 10),
    'n_estimators': (100, 250),
    }

    clf = XGBClassifier()
    clf.fit(X,y)
    
    bayesian_search = BayesSearchCV(
    clf, 
    search_spaces = params,
    scoring='roc_auc',
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    ),
    n_jobs = n_jobs,
    n_iter = n_iter,  
    verbose = verbose,
    refit = True,
    random_state = 42
    )

    bayesian_search.fit(X,y)
    print('best_score : ',bayesian_search.best_score_)
    best_params = bayesian_search.best_params_
    
    return best_params

def train_xgboost(X, y, n_iter=100, n_jobs=8, verbose=1, display=False):
    '''
    train xgboost with best parameter
    '''
    best_params = param_tune(X, y, n_iter, n_jobs, verbose)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    clf = XGBClassifier(
        learning_rate= best_params['learning_rate'],
        max_depth = best_params['max_depth'],
        colsample_bytree = best_params['colsample_bytree'],
        min_child_weight = best_params['min_child_weight'],
        n_estimators = best_params['n_estimators'],
    )
    clf.fit(X_train, y_train)
    # accuracy
    if display:
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)
        print(f'Accuracy: {accuracy:.2f}')    
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
    return clf

def pred_xgboost(clf, X):
    predictions = clf.predict(X)
    return predictions

def pred_single(single_sample, clf):
    '''
    input : single dataset
    return : class(0 or 1), probability
    '''
    pred_class = clf.predict(single_sample)
    pred_proba = clf.predict_proba(single_sample)
    
    return_proba = pred_proba[0].max()

    return pred_class[0], round(return_proba*100, 3)

def global_explain(clf, X, display=False):
    # SHAP explainer creation
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    if display:
        shap.summary_plot(shap_values, X, show=True)
    else:
        # save as image
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        # save as image
        plt.savefig('./shap_global_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    return explainer, shap_values

# def local_explain(X, explainer, shap_values, display=False):
#     shap_values = explainer.shap_values(X)

#     if display:
#         shap.waterfall_plot(shap.Explanation(values=shap_values[X_index], 
#                                         base_values=explainer.expected_value, 
#                                         data=X.iloc[X_index]),
#                                         )
#     else:
#         plt.figure()
#         shap.waterfall_plot(shap.Explanation(values=shap_values[X_index], 
#                                             base_values=explainer.expected_value, 
#                                             data=X.iloc[X_index]),
#                                             show=display)
#         # save as image
#         plt.savefig('./shap_local_plot.png', dpi=300, bbox_inches='tight')
#         plt.close()

def local_explain(X, explainer, display=False):
    # Get shap values and explanation for the input sample
    shap_explanation = explainer(X)
    shap_values = shap_explanation.values[0]         # For a single prediction
    base_value = shap_explanation.base_values[0]
    data = X.iloc[0] if hasattr(X, 'iloc') else X    # Handle both DataFrame and array input

    if display:
        shap.waterfall_plot(shap.Explanation(values=shap_values, 
                                             base_values=base_value, 
                                             data=data))
    else:
        plt.figure()
        shap.waterfall_plot(shap.Explanation(values=shap_values, 
                                             base_values=base_value, 
                                             data=data),
                                             show=False)
        # Save as image
        plt.savefig('./shap_local_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

def main(clf, explainer, data):
    df = json.loads(data)
    df = pd.DataFrame(df)
    df = preprocessing(df)
    acc_class, proba = pred_single(df, clf)

    # explain
    local_explain(df, explainer, display=True)

    return acc_class, proba

if __name__ == 'main':
    main()