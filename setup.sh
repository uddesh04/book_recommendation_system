mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"19bec004@iiitdwd.ac.in\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
