# Osnove-strojnog-u-enja---lv

- za pretvorbu kategorickih u num:
ohe = OneHotEncoder ()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type'] = X_encoded
- drugi nacin:- X['Sex'].replace({'male' : 0,
                        'female' : 1
                        }, inplace = True)

Za računala na labosima:
pip install (-U) numpy --user
pip install (-U) matplotlib --user
pip install (-U) pandas --user
pip install (-U) scikit-learn --user
pip install (-U) tensorflow --user

keras==2.12.0
matplotlib==3.7.1
numpy==1.23.4
pandas==1.5.3
scikit-learn==1.2.2
seaborn==0.12.2
tensorboard==2.12.2
tensorflow==2.12.0
