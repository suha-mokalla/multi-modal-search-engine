# multi-modal-search-engine
A search engine that takes multiple modes of input

poetry is used for dependency management. To install dependencies, point to the main folder (this consits of the file `pyproject.toml`) and run 

```
poetry install
```

### CLI text search

To run the CLI text search, run the following command:

```
poetry run python main.py --pdf_folder <path_to_pdf_folder> --query <query_text> --num_results <number_of_results>
```

Here is an example of how to run the CLI text search:

```
poetry run python main.py --pdf_folder ./pdf_folder --query "beginner friendly" --num_results 3
```

### Text Search App

To use the web application, run this:

```
poetry run streamlit run app.py
```
This will open the web application in your default browser. Upload the PDFs, enter the search query and get results, as simple as that. Below are the screenshots from the app.

![image](./static/text_app1.png)
![image](./static/text_app2.png)


### To-Do:

- ~~Add a web interface for the CLI text search~~
- Add cli for image search
- Add a web interface for the image search
- Add cli for audio search
- Add a web interface for the audio search







