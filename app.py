import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
import uuid
import os
from datetime import datetime
import pytz

load_dotenv(override=True)

def get_current_date():
   warsaw_tz = pytz.timezone('Europe/Warsaw')
   current_date = datetime.now(warsaw_tz)
   return current_date

def get_tool_display_name(tool_name):
   tool_names = {
       "tavily_search_results_json": "Wyszukiwarka",
       "youtube_search": "YouTube",
       "wikipedia": "Wikipedia"
   }
   return tool_names.get(tool_name, tool_name)

def get_youtube_titles(urls):
   titles = []
   for url in urls:
       if 'youtube.com/watch?v=' in url:
           video_id = url.split('watch?v=')[1].split('&')[0]
           titles.append(f"🎥 [Zobacz film na YouTube](https://www.youtube.com/watch?v={video_id})")
   return titles

def build_agent():
   current_date = get_current_date()
   
   system_message = SystemMessage(
       content=f"""Jesteś pomocnym asystentem AI, który:
       1. Odpowiada po polsku w naturalny i przyjazny sposób
       2. BARDZO WAŻNE: Dzisiaj jest {current_date.strftime("%d.%m.%Y")} - zawsze używaj tej daty jako punktu odniesienia
       3. Zawsze zaznaczaj datę w swoich odpowiedziach i upewnij się, że informacje są aktualne
       4. Jeśli nie masz dostępu do najnowszych informacji z dzisiaj, wyraźnie to zaznacz
       5. Potrafi korzystać z różnych źródeł do weryfikacji informacji
       6. Informuje o źródłach i ich datach
       7. W przypadku niejasności prosi o doprecyzowanie"""
   )

   llm = ChatOpenAI(
       model="gpt-4-turbo-preview",
       temperature=0.7,
       streaming=True
   )
   
   tools = [
       TavilySearchResults(
           max_results=3,
           kwargs={"include_domains": ["wyborcza.pl", "wp.pl", "onet.pl", "interia.pl", "rmf24.pl", "polsatnews.pl"]}
       ),
       YouTubeSearchTool(),
       WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
   ]

   return create_react_agent(llm, tools, state_modifier=system_message)

def init_chat():
   if "messages" not in st.session_state:
       st.session_state.messages = []
   
   if "agent" not in st.session_state:
       st.session_state.agent = build_agent()
       st.session_state.thread_id = str(uuid.uuid4())

def clean_tool_content(content):
   if isinstance(content, str):
       # Jeśli treść to lista linków YouTube
       if content.startswith('[') and 'youtube.com' in content:
           try:
               youtube_links = eval(content)
               titles = get_youtube_titles(youtube_links)
               return "\n\nOto polecane filmy z YouTube:\n" + "\n".join(titles)
           except:
               return None
       # Jeśli treść to surowy JSON
       if content.strip().startswith('[{'):
           return None
   return content

def main():
   st.set_page_config(
       page_title="AI Assistant",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   
   st.markdown("""
       <style>
       .stButton>button {
           width: 100%;
           border-radius: 5px;
           height: 3em;
       }
       .sidebar .element-container {
           margin-bottom: 10px;
       }
       </style>
   """, unsafe_allow_html=True)

   st.title("AI Assistant")
   
   init_chat()

   # Sidebar
   with st.sidebar:
       st.header("O Asystencie")
       st.write("Ten asystent AI może:")
       
       features = {
           "🔍": "Wyszukiwać informacje",
           "💭": "Odpowiadać na pytania",
           "🛠️": "Pomagać w rozwiązywaniu problemów",
           "📝": "Tworzyć i analizować treści"
       }
       
       for emoji, feature in features.items():
           st.write(f"{emoji} {feature}")
           
       st.markdown("---")
       
       if st.button("🗑️ Wyczyść czat"):
           st.session_state.messages = []
           st.rerun()
           
       current_date = get_current_date()
       st.markdown("---")
       st.write(f"🗓️ Data: {current_date.strftime('%d.%m.%Y')}")

   # Display chat history
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])

   # Chat input
   if user_input := st.chat_input("Wpisz wiadomość..."):
       with st.chat_message("user"):
           st.write(user_input)
       
       with st.chat_message("assistant"):
           message_placeholder = st.empty()
           full_response = ""
           
           try:
               for chunk, metadata in st.session_state.agent.stream(
                   {"messages": [("human", user_input)]},
                   config={"configurable": {"thread_id": st.session_state.thread_id}},
                   stream_mode="messages"
               ):
                   if hasattr(chunk, 'content'):
                       content = clean_tool_content(chunk.content)
                       if content:
                           full_response += content
                           message_placeholder.markdown(full_response + "▌")
                   
                   if (hasattr(chunk, 'name') and hasattr(chunk, 'content') 
                       and chunk.name not in ["None", "", None]):
                       content = clean_tool_content(chunk.content)
                       if content and not ('youtube.com' in str(chunk.content)):
                           friendly_name = get_tool_display_name(chunk.name)
                           with st.expander(f"🔍 Sprawdzam w: {friendly_name}"):
                               st.info(content)
               
               message_placeholder.markdown(full_response)
               
               if full_response:
                   st.session_state.messages.append({"role": "user", "content": user_input})
                   st.session_state.messages.append({"role": "assistant", "content": full_response})
               
           except Exception as e:
               st.error(f"Wystąpił błąd: {str(e)}")

if __name__ == "__main__":
   main()