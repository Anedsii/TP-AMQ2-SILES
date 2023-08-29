"""
feature_engineering.py

DESCRIPTION: 

AUTHORS: Angela Siles

DATE: 2023-08-21

"""

# Imports
import pandas as pd
import os
class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Identifies and combines the train and test data.
        
        Args:
        data_train (pd.DataFrame): DataFrame of training data.
        data_test (pd.DataFrame): DataFrame of test data.
        
        Returns:
        combined_df (pd.DataFrame): Combined train and test data.
        """
        try:
            data_train = pd.read_csv(self.input_path + 'Train_BigMart.csv')
            data_test = pd.read_csv(self.input_path + 'Test_BigMart.csv')
            data_train['Set'] = 'train'
            data_test['Set'] = 'test'
            combined_df = pd.concat([data_train, data_test], 
                                    ignore_index=True, sort=False)
        
        except FileNotFoundError:
            raise FileNotFoundError("One or both files were not found.")
        
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError("One or both files are empty.")
     
        return combined_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data and apply EDA process.

        Args:
            df (pd.DataFrame): DataFrame to be transformed.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        # Describing the DataFrame
        df_description = df.describe()

        # Calculating Outlet_Establishment_Year
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        # Unifying labels for "Item_Fat_Content"
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'low fat': 'Low Fat', 
            'LF': 'Low Fat', 
            'reg': 'Regular'
        })

        # LIMPIEZA: Missing data of Item_Weight variables
        missing_weights = df[df['Item_Weight'].isnull()]['Item_Identifier'].unique()
        for item in missing_weights:
            mode_weight = df[df['Item_Identifier'] == item]['Item_Weight'].mode().iloc[0]
            df.loc[df['Item_Identifier'] == item, 'Item_Weight'] = mode_weight

        # CLEANING DATA: Missing data on Outlet_Size variables
        missing_outlets = df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique()
        for outlet in missing_outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] = 'Small'

        # New category item fat
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # FEATURES ENGINEERING: Generating categories for 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({
            'Others': 'Non perishable',
            'Health and Hygiene': 'Non perishable',
            'Household': 'Non perishable', 
            'Seafood': 'Meats', 
            'Meat': 'Meats', 
            'Baking Goods': 'Processed Foods',
            'Frozen Foods': 'Processed Foods', 
            'Canned': 'Processed Foods', 
            'Snack Foods': 'Processed Foods', 
            'Breads': 'Starchy Foods', 
            'Breakfast': 'Starchy Foods', 
            'Soft Drinks': 'Drinks', 
            'Hard Drinks': 'Drinks', 
            'Dairy': 'Drinks'
        })

        # Codificación de los precios de productos
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels=[1, 2, 3, 4])

        # Codificación de variables ordinales
        data = df.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
        data['Outlet_Size'] = data['Outlet_Size'].replace({
            'High': 2, 
            'Medium': 1, 
            'Small': 0
        })
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace({
            'Tier 1': 2, 
            'Tier 2': 1, 
            'Tier 3': 0
        })

        # Codificación de variables nominales
        data_transformed = pd.get_dummies(data, columns=['Outlet_Type'], dtype=int)

        return data_transformed


    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write the prepared data to a CSV file named 'features.csv'.

        Args:
            transformed_dataframe (pd.DataFrame): The DataFrame with prepared data.
        """
        try:
            # Completing the docstring if necessary
            """
            This function writes the transformed data to a CSV file named 'features.csv'.

            Args:
                transformed_dataframe (pd.DataFrame): The DataFrame with prepared data.

            Returns:
                None
            """
            
            # Completing with the appropriate code
            output_file = 'features.csv'
            output_path = os.path.join(self.output_path, output_file)
            transformed_dataframe.to_csv(output_path, index=False)
            
            print(f"Transformed data written to {output_path}")
        except Exception as e:
            print("Error while writing prepared data:", e)

        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        #print("Column Names:", df_transformed.columns.tolist())  # Mostrar los nombres de las columnas
        #print("Transformed DataFrame (first 5 rows):\n", df_transformed.head(5))
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = 'data/',
                               output_path = 'data/').run()
    
