# DL4DS_FinalProject
This repo contains the code for my DL4DS final project, a Sudan Agricultural Advising Consultant.

In order to reproduce the results, first make sure you have an OpenAI account, and set the OpenAI key in your environment variable for this to work. An example of how to do this would be to enter the following code into the terminal:

```
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', '<Your-API-Key>', [System.EnvironmentVariableTarget]::User)
```

Once that is all set, install the dependencies:

```
pip install -r requirements.txt
```

Then create the Chroma database:

```
python create_database.py
```

And finally, you can open up the chatbot interface within Streamlit to begin querying:

```
streamlit run app.py
```

If you ever want to check the results of the TruLens evaluation functions after querying, close the running chatbot application with Ctrl+C in the terminal, then type:

```
trulens-eval
```

Clicking on the network URL link that pops up will allow you to view the evaluation results in a new Streamlit tab.
