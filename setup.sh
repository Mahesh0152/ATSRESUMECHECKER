mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port=$PORT\n\
enableCORS=false\n\
\n\
" > ~/.streamlit/congfig.toml
