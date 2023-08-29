
import pandas as pd
import logging
import os
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Configuración de registro
log_file = 'train.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file)

class ModelMetrics:
    @staticmethod
    def calculate_metrics(train_x, train_y, y_value, x_value, pred, model):
        mse_train = metrics.mean_squared_error(train_y, model.predict(train_x))
        mse_training = mse_train**2
        r2_train = model.score(train_x, train_y)
        
        mse_val = metrics.mean_squared_error(y_value, pred)
        mse_validation = mse_val**2
        r2_val = model.score(x_value, y_value)
        model_intercept = model.intercept_
        
        logging.info('Model evaluation metrics:')
        logging.info(f'TRAINING: RMSE: {mse_training:.2f} - R2: {r2_train:.4f}')
        logging.info(f'VALIDATION: RMSE: {mse_validation:.2f} - R2: {r2_val:.4f}')

        logging.info('\nModel coefficients:')
        logging.info(f'Intersection: {model_intercept:.2f}')
        
        coef = pd.DataFrame(train_x.columns, columns=['variables'])
        coef['Estimated coefficients'] = model.coef_
        logging.info(coef)
        coef.sort_values(by='Estimated coefficients').set_index('variables').plot(
            kind='bar', title='Importance of variables', figsize=(12, 6))
        
        plt.show()

class ModelWriter:
    @staticmethod
    def write_model_data(dataframe, output_file):
        try:
            dataframe.to_csv(output_file, index=False)
            logging.info(f'Saved {output_file}')
        except Exception as e:
            logging.error(f'Error writing to {output_file}: {e}')

class ModelTrainingPipeline:
    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self):
        try:
            train_file = 'features.csv'
            train_data = os.path.join(self.input_path, train_file)
            logging.info(f'Loading data from: {train_data}')
            return pd.read_csv(train_data)
        except (FileNotFoundError, PermissionError, OSError) as e:
            logging.error(f'Error opening file: {e}')
            return pd.DataFrame()

    def train_model(self, df):
        logging.info('Training model...')
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
        
        df_train = dataset[dataset['Set'] == 'train']
        df_test = dataset[dataset['Set'] == 'test']
        
        df_train.drop(columns=['Set'], inplace=True)
        df_test.drop(columns=['Item_Outlet_Sales', 'Set'], inplace=True)
        
        ModelWriter.write_model_data(df_train, os.path.join(self.model_path, 'train_final.csv'))
        ModelWriter.write_model_data(df_test, os.path.join(self.model_path, 'test_final.csv'))

        seed = 28
        model = LinearRegression()

        X = df_train.drop(columns='Item_Outlet_Sales')
        y = df_train['Item_Outlet_Sales']
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=seed)

        trained_model = model.fit(x_train, y_train)
        predicted_model = model.predict(x_val)

        ModelMetrics.calculate_metrics(x_train, y_train, y_val, x_val, predicted_model, model)

        return trained_model

    def save_model(self, model_trained):
        try:
            trained_file = 'trained_model.pkl'
            model_output = os.path.join(self.model_path, trained_file)
            with open(model_output, 'wb') as output:
                pickle.dump(model_trained, output)
            logging.info('Saved trained model.')
        except Exception as e:
            logging.error(f'Error writing trained model: {e}')

    def run(self):
        df = self.read_data()
        model_trained = self.train_model(df)
        self.save_model(model_trained)

if __name__ == '__main__':
    input_path = 'data/'
    model_path = 'model/'
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    ModelTrainingPipeline(input_path=input_path, model_path=model_path).run()
