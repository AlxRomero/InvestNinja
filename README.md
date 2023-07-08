# InvestNinja
AI-Powered Patent Insights

Steps for running `test_vertex.py`:
1. Setup up Google Cloud SDK and select the `investninja` project
    - `gcloud init`
    - `gcloud auth application-default login`
2. Set up the `GOOGLE_APPLICATION_CREDENTIALS` envionment variable in .bashrc or .zshrc file so that it points to the application_default_credentials.json file
3. Run `python test_vertex.py`