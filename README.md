# (Local) RAG Experiment

A python script that is an experiment in using local files to augment querying a LLM (or SLM, in this case). Uses `ollama` and the `phi3:mini` model. Should be able to parse HTML, PDF, and text files, but I've only tried with HTML so far.

Runs on a Raspberry Pi 5, but is painfully slow at the moment:
- takes a couple of minutes to tokenise the input files (a few hundred HTML files in my case)
- can take several minutes to return an answer, depending on the query

I'd like to improve this performance one day. The idea of having an "at-home" chatbot able to pull info from my personal files sounds appealing.

Outputs the full input and API response for debugging purposes.

Built using ChatGPT (4o) and GitHub Copilot, as I've preciously only written ~20 lines of basic python code, so there's probably plenty scope for optimisation.

To get started:
1. Install ollama
2. Pull the Phi3-Mini model
3. Add a `.env` file. Set `DOCS_LOCATION=<path to your source files>`
4. Install the script dependencies: `pip install -r requirements.txt`
5. Run `python dingus.py`

