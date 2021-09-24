mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"1171200748@student.mmu.edu.my\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml