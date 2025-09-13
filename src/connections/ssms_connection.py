#to connect to SQL Server (ODBC).
import pyodbc
# to load query results into a DataFrame.
import pandas as pd
# to read connection settings from a JSON file.
import json
# to work with file paths.
import os

# Defines a function main that accepts a path to a JSON config file.
def main(config_path="config.json"):
    """
    Fetches data from a SQL Server table and returns it as a DataFrame.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    # Get the directory of the current script
    # Resolves the absolute folder path where this .py file lives.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script path: {script_dir}")
    # Builds the absolute path to the JSON config by joining the script’s folder with config_path.
    config_file = os.path.join(script_dir, config_path)
    # Prints the final config file path.
    print(f"Config path: {config_file}")

    # Load configuration from JSON
    with open(config_file, "r") as file:
        config = json.load(file)
    
    # Read SQL Server connection details
    # Extracts the connection parameters (and the table name) from the JSON.
    server = config["sql_server"]["server"]
    database = config["sql_server"]["database"]
    table = config["sql_server"]["table"]

    print(f"Server: {server}, database: {database}, table: {table}")

    # Build an ODBC connection string
    # Define connection string for Windows Authentication
    connection_string = (
        # DRIVER={{SQL Server}} picks the default “SQL Server” ODBC driver on Windows. (Often you’ll want a specific one like ODBC Driver 17 for SQL Server.)
        f"DRIVER={{SQL Server}};"
        # SERVER and DATABASE come from config.
        f"SERVER={server};"
        f"DATABASE={database};"
        # means Windows Integrated Auth (no username/password in the string). For SQL auth you’d instead add UID=...;PWD=...; and drop Trusted_Connection.
        f"Trusted_Connection=yes;"
    )

    print(f"{connection_string}")
    # Tries to open an ODBC connection using that string.
    try:
        # Establish connection
        conn = pyodbc.connect(connection_string)
        # Basic check/feedback on whether the connection object was created.
        if conn:
            print("Connection to SQL Server successful!")
        else:
            print("Could not connect to SSMS")

        # Fetch data from the specified table
        # Builds a simple SQL query to read the whole table.
        query = f"SELECT * FROM {table}"
        # Uses pandas.read_sql to execute it and load results directly into a DataFrame df.
        df = pd.read_sql(query, conn)
        # Closes the DB connection, logs success, and returns the DataFrame.
        conn.close()
        print(f"Data fetched successfully from table '{table}'.")
        return df
    except Exception as e:
        # Catches any error (connection/driver/query issues), prints it, and returns None.
        print(f"Error connecting to SQL Server or fetching data: {e}")
        return None














