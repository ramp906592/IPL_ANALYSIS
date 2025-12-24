🏏 IPL Analytics Dashboard

🚀 IPL Analytics Dashboard is an interactive web application that combines Machine Learning–based match prediction with raw statistical player comparison in a single, clean dashboard. The project is built using Python and Streamlit and demonstrates an end-to-end workflow from data handling and ML model integration to deployment-ready UI development.

The first core feature of the dashboard is 🏏 Match Prediction. A trained Machine Learning model is used to predict the winning probability of both teams based on the current match situation. Users can select the batting team, bowling team, city, target score, current score, overs completed, and wickets lost. The model uses predict_proba to calculate win probabilities, which are displayed clearly using percentage cards and progress bars for better visual understanding. All models and encoders are loaded using Joblib.

The second major feature is 📊 Player Comparison. This module allows users to compare two players using raw (non-normalized) career statistics. The comparison is divided into three clear sections — Batting 🏏, Bowling 🎯, and Overall 🏆. For every metric, the better-performing player is highlighted with a ✅ emoji, making the comparison intuitive and easy to read. An Overall Summary and Final Verdict is also provided to determine the overall better player based on total metric wins. This module is intentionally analytical and does not use any ML model, ensuring a clear separation between predictive and descriptive analytics.

The application also includes a 🏠 Home Page, which serves as the landing screen for the dashboard. It displays the total number of players available, the model load status, and a brief overview of the available features. Navigation between Home, Match Prediction, and Player Comparison is handled smoothly using the sidebar.

🧠 Two different datasets are used in this project by design — historical match-level data for match prediction and raw career statistics for player comparison. This separation follows industry best practices and makes the system modular, clean, and easy to explain during interviews.

🛠️ The project is developed using Python, Streamlit, Pandas, NumPy, Scikit-learn, and Joblib. The folder structure is kept simple to support easy deployment. The application requires only app.py, requirements.txt, trained .pkl model files, and the CSV dataset.

To run the application locally, install the dependencies listed in requirements.txt and run streamlit run app.py. The project is fully deployment-ready and can be hosted on platforms such as Streamlit Cloud, Render, or Railway without any code changes.
