# InvestNinja
AI-Powered Patent Insights

For running the Streamlit application:
`streamlit run main.py`


Steps for running `test_vertex.py`:
1. Install Google Cloud SDK 
2. Select the `investninja` project during init
    - `gcloud init`
    - `gcloud auth application-default login`
3. Set up the `GOOGLE_APPLICATION_CREDENTIALS` envionment variable in .bashrc or .zshrc file so that it points to the application_default_credentials.json file
4. Install the Python packages using the command `pip install -r requirements.txt`
5. Run `python test_vertex.py`
