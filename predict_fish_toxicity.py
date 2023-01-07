import pandas as pd
import joblib

def input_data(x1, x2, x3, x4, x5, x6):
    input_data = [[x1, x2, x3, x4, x5, x6]]
    new_data = pd.DataFrame(input_data, columns = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'NdsCH', 'NdssC',	'MLOGP'])
    return new_data

def prediction(model, data):
    result = model.predict(data)
    return result[0]

if __name__ == "__main__" :
    model = joblib.load("model/qsar_fish_toxicity_model.sav")
    print('\n----------------Fish Toxicity Prediction----------------\n')
    print('Predict acute aquatic toxicity towards the fish Pimephales promelas based on their molecular descriptor\n')
    print('Please input the molecular descriptors: ')
    var_1 = input('1. CIC0 : ')
    var_2 = input('2. SM1_Dz(Z) : ')
    var_3 = input('3. GATS1i : ')
    var_4 = input('4. NdsCH : ')
    var_5 = input('5. NdssC : ')
    var_6 = input('6. MLOGP : ')
    new_data = input_data(var_1, var_2, var_3, var_4, var_5, var_6)
    predict = prediction(model, new_data)
    print('\nToxicity Prediction :', predict)