import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import plotly.express as px
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression # this could be any ML method
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import plotly.figure_factory as ff

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import pandas as pd
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error)  # Custom metric for RMSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.linear_model import LinearRegression
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Movies")
st.write('Welcome to my Streamlit app!')

import zipfile
import pandas as pd

import hashlib
import json
import streamlit as st


def load_data():
    zip_file_path = 'example.csv.zip'  # Path to the ZIP archive

    # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # List all files in the archive
        file_names = z.namelist()
        print("Files in the archive:", file_names)  # Debugging step

        # Ensure the file you want to load exists
        csv_file_name = 'example.csv'  # Correct name of the CSV file inside the ZIP
        if csv_file_name in file_names:
            with z.open(csv_file_name) as csv_file:
                # Read the CSV file
                data = pd.read_csv(csv_file)
        else:
            raise KeyError(f"'{csv_file_name}' not found in the archive.")
    
    return data

df = load_data()


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from category_encoders import LeaveOneOutEncoder
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section", ["KNN & Random Forest", "Linear Regression", 'SVM', 'Predictions'])

if option == "KNN & Random Forest":
    st.title("Movie Rating Prediction with Random Forest and KNN")


    # Sample and preprocess
    df_sampled = df.sample(frac=0.1, random_state=42)

# Extract the release year from the 'title' column using a regular expression
    df_sampled['release_year'] = df_sampled['title'].str.extract(r'\((\d{4})\)', expand=False)
    df_sampled['release_year'] = pd.to_numeric(df_sampled['release_year'], errors='coerce')  # Handle non-numeric cases

# Fill missing 'release_year' values with the median year
    df_sampled['release_year'] = df_sampled['release_year'].fillna(df_sampled['release_year'].median())

# Split the 'genres' column into lists of individual genres
    df_sampled['genres'] = df_sampled['genres'].str.split('|')

# Flatten the list of genres to find the most common genres
    genres_exploded = df_sampled['genres'].explode()
    top_genres = genres_exploded.value_counts().nlargest(20).index.tolist()  # Top 20 genres

# Create one-hot encoded columns for the top genres
    for genre in top_genres:
        df_sampled[f'genre_{genre}'] = df_sampled['genres'].apply(lambda x: int(genre in x))

# Define features (release year and genre columns) and target (ratings)
    feature_columns = ['release_year'] + [f'genre_{genre}' for genre in top_genres]
    X = df_sampled[feature_columns].copy()  # Feature matrix
    y = df_sampled['rating'].copy()  # Target variable

# Apply Leave-One-Out Encoding for 'userId' and 'movieId'
    encoder = LeaveOneOutEncoder(cols=['userId', 'movieId'])
    X_encoded_ids = encoder.fit_transform(df_sampled[['userId', 'movieId']], y)

# Concatenate the encoded IDs with the main feature matrix
    X = pd.concat([X_encoded_ids, X], axis=1)
    
# Add the encoded IDs back to the DataFrame for interpretability
    df_sampled['encoded_userId'] = X['userId']
    df_sampled['encoded_movieId'] = X['movieId']

# Split the dataset into training and testing sets for the Random Forest model
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Initialize and train a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train_rf, y_train_rf)

# Predict ratings using the Random Forest model
    y_pred_rf = rf.predict(X_test_rf)

# Evaluate the Random Forest model using various metrics
    mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
    rmse_rf = root_mean_squared_error(y_test_rf, y_pred_rf)
    mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
    r2_rf = r2_score(y_test_rf, y_pred_rf)

    df_test_rf = X_test_rf.copy()
    df_test_rf['Actual'] = y_test_rf.values
    df_test_rf['Predicted'] = y_pred_rf

# Standardize features for the K-Nearest Neighbors (KNN) model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets for the KNN model
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

# Initialize and train a KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_knn, y_train_knn)

# Predict ratings using the KNN model
    y_pred_knn = knn.predict(X_test_knn)

# Evaluate the KNN model using various metrics
    mse_knn = mean_squared_error(y_test_knn, y_pred_knn)
    rmse_knn = root_mean_squared_error(y_test_knn, y_pred_knn)
    mae_knn = mean_absolute_error(y_test_knn, y_pred_knn)
    r2_knn = r2_score(y_test_knn, y_pred_knn)

    df_test_knn = pd.DataFrame(X_test_knn, columns=X.columns)
    df_test_knn['Actual'] = y_test_knn.values
    df_test_knn['Predicted'] = y_pred_knn

    # Model performance comparison
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'KNN'],
        'MSE': [mse_rf, mse_knn],
        'RMSE': [rmse_rf, rmse_knn],
        'MAE': [mae_rf, mae_knn],
        'R²': [r2_rf, r2_knn]
    })

    st.write("Random Forest Metrics:")
    st.write(metrics_df.iloc[0])

    st.write("KNN Metrics:")
    st.write(metrics_df.iloc[1])

    # Plot performance comparison
    fig = make_subplots(rows=2, cols=2, subplot_titles=('MSE', 'RMSE', 'MAE', 'R²'))
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MSE'], text=metrics_df['MSE']), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], text=metrics_df['RMSE']), row=1, col=2)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'], text=metrics_df['MAE']), row=2, col=1)
    fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['R²'], text=metrics_df['R²']), row=2, col=2)
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(title_text='Model Performance Comparison', showlegend=False, height=700)

    st.plotly_chart(fig)

    fig1 = px.histogram(df_sampled, x='rating', nbins=20, title='Distribution of Actual Ratings')
    st.plotly_chart(fig1)

    fig2 = px.histogram(df_test_rf, x='Predicted', nbins=20, title='Random Forest: Predicted Rating Distribution')
    st.plotly_chart(fig2)
    fig3 = px.histogram(df_test_knn, x='Predicted', nbins=20, title='KNN: Predicted Rating Distribution')
    st.plotly_chart(fig3)

    fig4 = px.scatter(
    df_test_rf, x='Actual', y='Predicted',
    title='Random Forest: Actual vs. Predicted Ratings',
    labels={'Actual': 'Actual Rating', 'Predicted': 'Predicted Rating'}
    )
    fig4.add_shape(
        type="line", line=dict(dash='dash'),
        x0=df_test_rf['Actual'].min(), y0=df_test_rf['Actual'].min(),
        x1=df_test_rf['Actual'].max(), y1=df_test_rf['Actual'].max()
    )
    st.plotly_chart(fig4)

    fig5 = px.scatter(
        df_test_knn, x='Actual', y='Predicted',
        title='KNN: Actual vs. Predicted Ratings',
        labels={'Actual': 'Actual Rating', 'Predicted': 'Predicted Rating'}
    )
    fig5.add_shape(
        type="line", line=dict(dash='dash'),
        x0=df_test_knn['Actual'].min(), y0=df_test_knn['Actual'].min(),
        x1=df_test_knn['Actual'].max(), y1=df_test_knn['Actual'].max()
    )
    st.plotly_chart(fig5)

    df_test_rf['Residuals'] = df_test_rf['Actual'] - df_test_rf['Predicted']
    fig6 = px.histogram(df_test_rf, x='Residuals', nbins=20, title='Random Forest: Residual Distribution')
    st.plotly_chart(fig6)

    df_test_knn['Residuals'] = df_test_knn['Actual'] - df_test_knn['Predicted']
    fig7 = px.histogram(df_test_knn, x='Residuals', nbins=20, title='KNN: Residual Distribution')
    st.plotly_chart(fig7)

    fig8 = px.scatter(
        df_test_rf, x='Predicted', y='Residuals',
        title='Random Forest: Residuals vs. Predicted Values',
        labels={'Predicted': 'Predicted Rating', 'Residuals': 'Residuals'}
    )

    fig8.add_shape(
        type='line',
        x0=df_test_rf['Predicted'].min(),
        x1=df_test_rf['Predicted'].max(),
        y0=0,
        y1=0,
        line=dict(color='red', dash='dash')
    )
    st.plotly_chart(fig8)


    fig9 = px.scatter(
        df_test_knn, x='Predicted', y='Residuals',
        title='KNN: Residuals vs. Predicted Values',
        labels={'Predicted': 'Predicted Rating', 'Residuals': 'Residuals'}
    )
    fig9.add_shape(type='line', x0=df_test_knn['Predicted'].min(), x1=df_test_knn['Predicted'].max(), y0=0,y1=0, 
                   line=dict(color='red', dash='dash'))

    st.plotly_chart(fig9)


    avg_rating_by_year = df_sampled.groupby('release_year')['rating'].mean().reset_index()
    fig10 = px.line(avg_rating_by_year, x='release_year', y='rating', title='Average Actual Rating by Year')

    st.plotly_chart(fig10)

    genre_cols = [f'genre_{genre}' for genre in top_genres]
    genre_ratings = {}
    for genre in genre_cols:
        mean_rating = df_sampled[df_sampled[genre] == 1]['rating'].mean()
        genre_ratings[genre] = mean_rating

    genre_ratings_df = pd.DataFrame(list(genre_ratings.items()), columns=['Genre', 'Average Rating'])
    fig12 = px.bar(genre_ratings_df, x='Genre', y='Average Rating', title='Average Actual Rating by Genre')

    st.plotly_chart(fig12)




    user_residuals_rf = df_test_rf.groupby('userId')['Residuals'].mean().reset_index()
    fig13 = px.histogram(user_residuals_rf, x='Residuals', nbins=20, title='Random Forest: User Average Residual Distribution')
    st.plotly_chart(fig13)

    movie_residuals_rf = df_test_rf.groupby('movieId')['Residuals'].mean().reset_index()
    fig14 = px.histogram(movie_residuals_rf, x='Residuals', nbins=20, title='Random Forest: Movie Average Residual Distribution')
    st.plotly_chart(fig14)

    importances = rf.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for feature importance and sort
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(20)

    # Plot the feature importance
    fig16 = px.bar(
        feature_importance_df,
        x='Feature',
        y='Importance',
        title='Random Forest: Feature Importance',
    )

    st.plotly_chart(fig16)
    
elif option == "Linear Regression":
    st.title("Movie Rating Prediction with Linear Regression")

    df_sampled = df.sample(frac=0.1, random_state=42)

    df_sampled['release_year'] = df_sampled['title'].str.extract(r'\((\d{4})\)', expand=False)
    df_sampled['release_year'] = pd.to_numeric(df_sampled['release_year'], errors='coerce')

    df_sampled['release_year'] = df_sampled['release_year'].fillna(df_sampled['release_year'].median())

    df_sampled['genres'] = df_sampled['genres'].str.split('|')

    genres_exploded = df_sampled['genres'].explode()
    top_genres = genres_exploded.value_counts().nlargest(20).index.tolist()

    for genre in top_genres:
        df_sampled[f'genre_{genre}'] = df_sampled['genres'].apply(lambda x: int(genre in x))

    feature_columns = ['release_year'] + [f'genre_{genre}' for genre in top_genres]
    X = df_sampled[feature_columns].copy()
    y = df_sampled['rating'].copy()

    encoder = LeaveOneOutEncoder(cols=['userId', 'movieId'])
    X_encoded_ids = encoder.fit_transform(df_sampled[['userId', 'movieId']], y)

    X = pd.concat([X_encoded_ids, X], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)


    lr = LinearRegression()

    lr.fit(X_train_lr, y_train_lr)

    y_pred_lr = lr.predict(X_test_lr)

    mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr) 
    mae_lr = mean_absolute_error(y_test_lr, y_pred_lr)
    r2_lr = r2_score(y_test_lr, y_pred_lr)

    metrics_df_lr = pd.DataFrame({
        'Model': ['Random Forest', 'KNN'],
        'MSE': [mse_lr, mse_lr],
        'RMSE': [rmse_lr, rmse_lr],
        'MAE': [mae_lr, mae_lr],
        'R²': [r2_lr, r2_lr]})

    st.write("Random Forest Metrics:")
    st.write(metrics_df_lr.iloc[0])

    st.write("KNN Metrics:")
    st.write(metrics_df_lr.iloc[1])

    train_indices, test_indices = train_test_split(
        np.arange(len(df_sampled)), test_size=0.2, random_state=42)

    # Use these indices to split the scaled features and target variable
    X_train_lr, X_test_lr = X_scaled[train_indices], X_scaled[test_indices]
    y_train_lr, y_test_lr = y.iloc[train_indices], y.iloc[test_indices]

    # Map test indices back to the original DataFrame
    test_data = df_sampled.iloc[test_indices]

    # Create the df_results DataFrame
    df_results = pd.DataFrame({
        "Actual": y_test_lr.values,          # Actual ratings
        "Predicted": y_pred_lr,              # Predicted ratings
        "Residual": y_test_lr.values - y_pred_lr,  # Residuals
        "userId": test_data["userId"].values,   # User IDs
        "movieId": test_data["movieId"].values })

    st.write("Results")
    st.dataframe(df_results.head())

    fig1 = px.scatter(df_results, x="Actual", y="Predicted", title="Actual vs. Predicted Ratings",
                 labels={"Actual": "Actual Ratings", "Predicted": "Predicted Ratings"},
                 hover_data=["userId", "movieId"])
    fig1.update_traces(marker=dict(size=7, opacity=0.6))
    fig1.add_shape(type="line", x0=0, y0=0, x1=5, y1=5, line=dict(color="red", dash="dash"))  # Ideal line
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1)


    fig2 = px.scatter(df_results, x="Predicted", y="Residual", title="Residuals vs. Predicted Ratings",
                 labels={"Predicted": "Predicted Ratings", "Residual": "Residuals"},
                 hover_data=["userId", "movieId"])
    fig2.update_traces(marker=dict(size=7, opacity=0.6))
    fig2.add_shape(type="line", x0=0, y0=0, x1=5, y1=0, line=dict(color="red", dash="dash"))  # Zero line
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2)

    fig3 = px.histogram(df_results, x="Residual", nbins=30, histnorm="density",
                   title="Distribution of Residuals",
                   labels={"Residual": "Residuals"})
    fig3.update_traces(marker=dict(color="blue", line=dict(width=1, color="black")))
    fig3.update_layout(yaxis_title="Density")
    st.plotly_chart(fig3)

    user_residuals = df_results.groupby("userId")["Residual"].mean().reset_index()

    fig4 = px.bar(user_residuals, x="userId", y="Residual",
                 title="Average Residuals by User",
                 labels={"userId": "User ID", "Residual": "Average Residual"})
    fig4.update_traces(marker_color="orange")
    st.plotly_chart(fig4)

    movie_residuals = df_results.groupby("movieId")["Residual"].mean().reset_index()

    fig5 = px.bar(movie_residuals, x="movieId", y="Residual",
                 title="Average Residuals by Movie",
                 labels={"movieId": "Movie ID", "Residual": "Average Residual"})
    fig5.update_traces(marker_color="green")
    st.plotly_chart(fig5)

elif option == "SVM":
    # Sample data
    df_sampled = df.sample(frac=0.01, random_state=1)
    # Extract and process release year
    df_sampled['release_year'] = df_sampled['title'].str.extract(r'\((\d{4})\)', expand=False)
    df_sampled['release_year'] = pd.to_numeric(df_sampled['release_year'], errors='coerce')
    df_sampled['release_year'] = df_sampled['release_year'].fillna(df_sampled['release_year'].median())
 
    # Process genres
    df_sampled['genres'] = df_sampled['genres'].str.split('|')
    genres_exploded = df_sampled['genres'].explode()
    top_genres = genres_exploded.value_counts().nlargest(20).index.tolist()
    for genre in top_genres:
        df_sampled[f'genre_{genre}'] = df_sampled['genres'].apply(lambda x: int(genre in x))
 
    # Define features and target
    feature_columns = ['release_year'] + [f'genre_{genre}' for genre in top_genres]
    X = df_sampled[feature_columns].copy()
    y = df_sampled['rating'].copy()
 
    # Label encode IDs instead of one-hot encoding
    le_user = LabelEncoder()
    le_movie = LabelEncoder()
    X['userId'] = le_user.fit_transform(df_sampled['userId'])
    X['movieId'] = le_movie.fit_transform(df_sampled['movieId'])
 
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
    # Grid search for hyperparameters
    svr = SVR()
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1]
    }
 
    grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
 
    # Train model with best parameters
    best_params = grid_search.best_params_
    best_svr = SVR(**best_params)
    best_svr.fit(X_train, y_train)
 
    # Make predictions
    y_pred = best_svr.predict(X_test)
 
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
 
    # Display results
    st.write("Best Parameters:", best_params)
    st.write("MSE:", mse)
    st.write("RMSE:", rmse)
    st.write("MAE:", mae)
    st.write("R^2:", r2)
 
    # Create results dataframe
    results_df = pd.DataFrame({
        'Actual Ratings': y_test,
        'Predicted Ratings': y_pred
    })
    results_df['Residuals'] = results_df['Actual Ratings'] - results_df['Predicted Ratings']
 
    # Plot distributions
    actuals = px.histogram(
        results_df.melt(var_name="Type", value_name="Ratings", 
                       value_vars=["Actual Ratings", "Predicted Ratings"]),
        x="Ratings",
        color="Type",
        title="Distribution of Actual vs. Predicted Ratings",
        labels={"Ratings": "Ratings", "Type": "Rating Type"},
        barmode="overlay",
        opacity=0.75
    )
    st.plotly_chart(actuals)
 
    residuals = px.histogram(
        results_df,
        x="Residuals",
        title="Distribution of Residuals",
        labels={"Residuals": "Residuals"},
        nbins=30
    )
    st.plotly_chart(residuals)

if option == "Predictions": 
    import hashlib
    import json
    import streamlit as st

    CREDENTIALS_FILE = "Users.json"

    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    # Load and save credentials
    def load_credentials():
        try:
            with open(CREDENTIALS_FILE, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # Handle the case where the file is not found or the JSON is invalid
            st.warning("Credentials file is missing or corrupted. Initializing a new one.")
            return {}

    def save_credentials(credentials):
        with open(CREDENTIALS_FILE, "w") as file:
            json.dump(credentials, file)

    # Login functionality
    def login():
        st.subheader("Log In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Log In"):
            credentials = load_credentials()
            hashed_password = hash_password(password)

            if username in credentials and credentials[username] == hashed_password:
                st.success(f"Welcome, {username}!")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
            else:
                st.error("Invalid username or password.")

    # Sign-up functionality
    def signup():
        st.subheader("Sign Up")
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                credentials = load_credentials()
                if username in credentials:
                    st.error("Username already exists.")
                else:
                    credentials[username] = hash_password(password)
                    save_credentials(credentials)
                    st.success("Sign up successful! You can now log in.")

    def main():
        st.title("User Authentication")

        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False

        if st.session_state["logged_in"]:
            st.success(f"Logged in as {st.session_state['username']}")
            if st.button("Log Out"):
                st.session_state["logged_in"] = False
        else:
            option = st.sidebar.selectbox("Choose an option", ["Log In", "Sign Up"])

            if option == "Log In":
                login()
            elif option == "Sign Up":
                signup()

    if __name__ == "__main__":
        main()




    st.title("Recommender System Web App")

# Sample data loading (replace with your data loading logic)
# Assuming df has columns: 'userId', 'movieId', 'title', 'rating'
# Example dataframe structure:
# df = pd.read_csv('your_movie_ratings.csv')

# Sidebar input for user ID
    user_id = st.sidebar.text_input("Enter User ID", value="1")
    user_id = int(user_id)
    user_ratings = df[df['userId'] == user_id]
    st.subheader(f"Movies Rated by User {user_id}")
    st.dataframe(user_ratings)


    st.sidebar.header("Input Preferences")
    user_input = st.sidebar.text_input("Enter a movie title or genre:", value="")

# Generate TF-IDF matrix for the 'title' column
    tfidf = TfidfVectorizer(stop_words='english')
    df['title'] = df['title'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['title'])

# Compute cosine similarity between movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
    def get_recommendations(input_value, k=5):
        if input_value:
            matching_movies = df[df['title'].str.contains(input_value, case=False, na=False)]
            if matching_movies.empty:
                st.warning("No movies found matching your input.")
                return []

            st.subheader(f"Movies matching '{input_value}':")
            st.dataframe(matching_movies[['movieId', 'title']].head(5))

            movie_idx = matching_movies.index[0]
            sim_scores = list(enumerate(cosine_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k + 1]

            recommendations = [(df.iloc[i[0]]['movieId'], i[1]) for i in sim_scores]
            return recommendations
        else:
            st.warning("Please enter a movie title or genre.")
            return []

# Button to trigger recommendations
    if st.sidebar.button("Get Recommendations"):
        recommendations = get_recommendations(user_input)
        if recommendations:
            st.subheader("Top Movie Recommendations:")
            for movie_id, similarity in recommendations:
                movie_title = df[df['movieId'] == movie_id]['title'].values[0]
                st.write(f"{movie_title} - Similarity Score: {similarity:.2f}")





    



