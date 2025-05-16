import streamlit as st

# Definition of the names of the pages and of what is present into them
page1 = st.Page('page1.py', title="Dynamics KPI")
page2 = st.Page('page2.py', title="Dynamics GRF")
# We have only defined the pages as objects, and then now we need to define the sidebar that allows to navigate into them (with a "navigation" object)

pg = st.navigation([page1,page2]) # In the navigation object we need to put a list (in []) of the pages in which we need to navigate
st.set_page_config(page_title="Home Page Dynamics")

# The key parameters are able to save the information in the sessions state that we are running

# We need to run the pages that we are recalling with the navigation, with this command
pg.run()
# The command "page-run" ("pg.run()"") should/must be put in the main page (the streamlit.app one), maybe where a sidebar is present (to navigate within pages)