"""
predict.py


"""

# Imports
import logging
import joblib
import pandas as pd

class MakePredictionPipeline:
    def __init__(self, input_path, input_path_test, output_path, model_path=None):
        self.input_path = input_path
        self.input_path_test = input_path_test
        self.output_path = output_path
        self.model_path = model_path

    def load_data(self, data_path):
        try:
            logging.info(f"Loading data from: {data_path}")
            return pd.read_csv(data_path)
        except FileNotFoundError:
            logging.error(f"File not found at: {data_path}")
            return pd.DataFrame()
        except (PermissionError, OSError) as error_load_file:
            logging.exception("Error loading data: %s", error_load_file)
            return pd.DataFrame()

    def load_model(self):
        try:
            logging.info(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            logging.error(f"Model file not found at: {self.model_path}")
        except (PermissionError, OSError) as error_load_model:
            logging.exception("Error loading the model: %s", error_load_model)

    def make_predictions(self, data):
        try:
            logging.info("Making predictions")
            data.drop(['Unnamed: 0', 'Outlet_Identifier', 'Item_Identifier',
                       'Item_Outlet_Sales', 'Set'], axis=1, inplace=True, errors='ignore')
            predictions = self.model.predict(data)
            return predictions
        except Exception as error_predictions:
            logging.exception("Error making predictions: %s", error_predictions)
            return pd.DataFrame()

    def write_predictions(self, predictions, filename="predict.csv"):
        try:
            logging.info(f"Writing predictions to: {self.output_path}/{filename}")
            df_predictions = pd.DataFrame(predictions, columns=['Prediction'])
            df_predictions.to_csv(f"{self.output_path}/{filename}", index=False)
        except (PermissionError, OSError) as error_write_file:
            logging.exception("Error writing predictions: %s", error_write_file)

    def run(self):
        data = self.load_data(self.input_path)
        self.load_model()
        predictions = self.make_predictions(data)
        self.write_predictions(predictions)

        data_test = self.load_data(self.input_path_test)
        data_test["pred_Sales"] = self.make_predictions(data_test)
        self.write_predictions(data_test, "test_predictions.csv")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = MakePredictionPipeline(
        input_path='data/features.csv',
        input_path_test='model/test_final.csv',
        output_path='predict/',
        model_path='model/trained_model.pkl')
    pipeline.run()